/**
 * Model 实现 - 模型加载和前向传播
 */

#include "pocket.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <stdexcept>

namespace pocket {

// 前向声明算子
namespace ops {
    void rmsnorm(const float* input, const float* weight, float* output, size_t size, float eps);
    void apply_rope_inplace(float* q, float* k, size_t num_q_heads, size_t num_kv_heads, size_t head_dim, pos_t position);
    void multi_head_attention_with_cache(
        const float* x, const float* q_weight, const float* k_weight,
        const float* v_weight, const float* o_weight, float* output,
        float* k_cache, float* v_cache, size_t current_pos,
        size_t hidden_size, size_t num_heads, size_t num_kv_heads,
        size_t head_dim, float* workspace);
    void swiglu_ffn_single(
        const float* x, const float* gate_weight, const float* up_weight,
        const float* down_weight, float* output,
        size_t hidden_size, size_t intermediate_size, float* workspace);
    void matvec(const float* A, const float* x, float* y, size_t m, size_t k);
}

bool Model::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // 1. 验证 magic number
    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "PLLM", 4) != 0) {
        return false;
    }

    // 2. 读取版本
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        return false;
    }

    // 3. 读取 header size
    uint32_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    // 4. 读取 config (JSON)
    uint32_t config_size;
    file.read(reinterpret_cast<char*>(&config_size), sizeof(config_size));

    std::string config_json(config_size, '\0');
    file.read(&config_json[0], config_size);

    // 解析 JSON (简化版 - 实际应使用 JSON 库)
    // TODO: 完整 JSON 解析
    config_.vocab_size = 6400;
    config_.hidden_size = 768;
    config_.num_layers = 12;
    config_.num_heads = 12;
    config_.num_kv_heads = 4;
    config_.intermediate_size = 2048;
    config_.max_seq_len = 2048;

    // 5. 读取 vocab
    uint32_t vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    // TODO: 读取词表

    // 6. 读取 tensor metadata
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    tensors_.resize(num_tensors);
    for (uint32_t i = 0; i < num_tensors; i++) {
        Tensor& tensor = tensors_[i];

        // name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        tensor.name.resize(name_len);
        file.read(&tensor.name[0], name_len);

        // shape
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        tensor.shape.resize(ndim);
        file.read(reinterpret_cast<char*>(tensor.shape.data()), ndim * sizeof(uint64_t));

        // qtype
        uint8_t qtype;
        file.read(reinterpret_cast<char*>(&qtype), sizeof(qtype));
        tensor.qtype = static_cast<QuantType>(qtype);

        // offset & size
        file.read(reinterpret_cast<char*>(&tensor.offset), sizeof(tensor.offset));
        uint64_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    }

    // 7. 读取 weights data
    // 获取文件当前位置作为 weights 起始
    size_t weights_start = file.tellg();

    // 计算总大小
    size_t total_size = 0;
    for (const auto& tensor : tensors_) {
        total_size = std::max(total_size, tensor.offset + tensor.nbytes());
    }

    // 读取所有权重
    weights_data_.resize(total_size);
    file.read(reinterpret_cast<char*>(weights_data_.data()), total_size);

    // 设置每个 tensor 的 data 指针
    for (auto& tensor : tensors_) {
        tensor.data = weights_data_.data() + tensor.offset;
    }

    file.close();
    return true;
}

const Tensor* Model::get_tensor(const std::string& name) const {
    for (const auto& tensor : tensors_) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

std::vector<float> Model::forward(
    const std::vector<token_t>& input_ids,
    KVCache* kv_cache
) {
    size_t seq_len = input_ids.size();
    size_t hidden_size = config_.hidden_size;
    size_t vocab_size = config_.vocab_size;

    // 分配工作空间
    std::vector<float> hidden_states(seq_len * hidden_size);
    std::vector<float> residual(seq_len * hidden_size);
    std::vector<float> workspace(seq_len * hidden_size * 8); // 临时缓冲区

    // 1. Embedding
    embedding(input_ids, hidden_states.data());

    // 2. Transformer layers
    for (int layer_idx = 0; layer_idx < config_.num_layers; layer_idx++) {
        // 保存 residual
        std::memcpy(residual.data(), hidden_states.data(), seq_len * hidden_size * sizeof(float));

        // RMSNorm
        const Tensor* ln_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
        ops::rmsnorm(hidden_states.data(), (float*)ln_weight->data, hidden_states.data(),
                     seq_len * hidden_size, config_.rms_norm_eps);

        // Attention
        const Tensor* q_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight");
        const Tensor* k_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight");
        const Tensor* v_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight");
        const Tensor* o_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight");

        // TODO: 完整的 attention 实现
        // 这里简化处理

        // Residual connection
        for (size_t i = 0; i < seq_len * hidden_size; i++) {
            hidden_states[i] += residual[i];
        }

        // 保存新的 residual
        std::memcpy(residual.data(), hidden_states.data(), seq_len * hidden_size * sizeof(float));

        // Post-attention RMSNorm
        const Tensor* post_ln_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");
        ops::rmsnorm(hidden_states.data(), (float*)post_ln_weight->data, hidden_states.data(),
                     seq_len * hidden_size, config_.rms_norm_eps);

        // FFN
        const Tensor* gate_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight");
        const Tensor* up_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight");
        const Tensor* down_weight = get_tensor("model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight");

        // TODO: FFN 实现

        // Residual connection
        for (size_t i = 0; i < seq_len * hidden_size; i++) {
            hidden_states[i] += residual[i];
        }
    }

    // 3. Final RMSNorm
    const Tensor* final_ln_weight = get_tensor("model.norm.weight");
    ops::rmsnorm(hidden_states.data(), (float*)final_ln_weight->data, hidden_states.data(),
                 seq_len * hidden_size, config_.rms_norm_eps);

    // 4. LM Head
    const Tensor* lm_head_weight = get_tensor("lm_head.weight");
    std::vector<float> logits(vocab_size);

    // 只计算最后一个 token 的 logits
    ops::matvec((float*)lm_head_weight->data,
                hidden_states.data() + (seq_len - 1) * hidden_size,
                logits.data(), vocab_size, hidden_size);

    return logits;
}

void Model::embedding(const std::vector<token_t>& input_ids, float* output) {
    const Tensor* embed_weight = get_tensor("model.embed_tokens.weight");
    float* embed_data = (float*)embed_weight->data;

    size_t hidden_size = config_.hidden_size;

    for (size_t i = 0; i < input_ids.size(); i++) {
        token_t token_id = input_ids[i];
        float* token_embed = embed_data + token_id * hidden_size;
        std::memcpy(output + i * hidden_size, token_embed, hidden_size * sizeof(float));
    }
}

void Model::rms_norm(const float* input, const float* weight, float* output, size_t size) {
    ops::rmsnorm(input, weight, output, size, config_.rms_norm_eps);
}

void Model::rope(float* q, float* k, pos_t pos, size_t head_dim) {
    ops::apply_rope_inplace(q, k, config_.num_heads, config_.num_kv_heads, head_dim, pos);
}

void print_model_info(const Model& model) {
    const auto& config = model.config();

    std::cout << "Model Information:" << std::endl;
    std::cout << "  Architecture: " << config.architecture << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  Hidden size: " << config.hidden_size << std::endl;
    std::cout << "  Num layers: " << config.num_layers << std::endl;
    std::cout << "  Num heads: " << config.num_heads << std::endl;
    std::cout << "  Num KV heads: " << config.num_kv_heads << std::endl;
    std::cout << "  Intermediate size: " << config.intermediate_size << std::endl;
    std::cout << "  Max seq length: " << config.max_seq_len << std::endl;

    // 计算参数量
    size_t total_params = 0;
    total_params += config.vocab_size * config.hidden_size; // embedding
    total_params += config.num_layers * (
        4 * config.hidden_size * config.hidden_size + // attention weights
        3 * config.hidden_size * config.intermediate_size // FFN weights
    );
    total_params += config.vocab_size * config.hidden_size; // lm_head

    std::cout << "  Total parameters: " << total_params / 1e6 << "M" << std::endl;
}

std::string version() {
    return "Pocket-0.1";
}

} // namespace pocket
