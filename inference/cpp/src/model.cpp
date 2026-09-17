/**
 * Model 实现 - 模型加载（.pllm 新格式）和前向传播
 *
 * 本实现与 tools/export_to_pllm.py 的输出格式严格匹配：
 *   compact_magic | version | header_size | config_size+config_json |
 *   vocab_size+vocab | num_tensors | per-tensor(metadata) | data
 *
 * 权重统一反量化为 fp32，存储于 dequant_data_；Tensor.data 指向反量化数据。
 * 权重布局为 [out, in]，使用 matvec（x @ W^T）方式计算。
 * RoPE 采用 rotate_half 非交错语义（cos/sin 对称重复），与 pocket_hf.py 一致。
 */

#include "pocket.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace pocket {

// 前向声明算子
namespace ops {
    void matvec(const float* A, const float* x, float* y, size_t m, size_t k);
    void rmsnorm(const float* input, const float* weight, float* output, size_t size, float eps);
    void softmax(float* x, size_t size);
}

// ============================================================================
// 简单扁平 JSON 提取工具（仅用于 header config，数值类型）
// ============================================================================
namespace {

// 取出 "key" 冒号后的原始值子串（数字或 null）
std::string json_extract(const std::string& json, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return "";
    size_t colon = json.find(':', pos + pat.size());
    if (colon == std::string::npos) return "";
    size_t start = colon + 1;
    // 跳过空格
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) start++;
    size_t end = start;
    if (start < json.size() && (json[start] == '-' || json[start] == '.' || std::isdigit((unsigned char)json[start]))) {
        while (end < json.size()) {
            char c = json[end];
            if (c == ',' || c == '}' || c == ']') break;
            end++;
        }
    }
    return json.substr(start, end - start);
}

long long json_int(const std::string& json, const std::string& key, long long dflt) {
    std::string v = json_extract(json, key);
    if (v.empty() || v == "null") return dflt;
    try {
        return std::stoll(v);
    } catch (...) { return dflt; }
}

double json_double(const std::string& json, const std::string& key, double dflt) {
    std::string v = json_extract(json, key);
    if (v.empty() || v == "null") return dflt;
    try {
        return std::stod(v);
    } catch (...) { return dflt; }
}

// fp16 -> fp32
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            // 非规格化
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// 将单个 tensor 的原始字节反量化为 fp32 数组
// data: 原始字节; qtype; n_total = 反量化后的元素总数（块填充后）
std::vector<float> dequantize(const uint8_t* ptr, QuantType qtype, size_t n_total) {
    std::vector<float> out(n_total);
    switch (qtype) {
        case QuantType::F32: {
            std::memcpy(out.data(), ptr, n_total * 4);
            break;
        }
        case QuantType::F16: {
            for (size_t i = 0; i < n_total; i++) {
                uint16_t h;
                std::memcpy(&h, ptr + i * 2, 2);
                out[i] = fp16_to_fp32(h);
            }
            break;
        }
        case QuantType::Q8_0:
        case QuantType::Q4_0: {
            constexpr size_t BLOCK = 32;
            size_t n_blocks = (n_total + BLOCK - 1) / BLOCK;
            for (size_t b = 0; b < n_blocks; b++) {
                // scale (fp16)
                uint16_t scale_bits;
                std::memcpy(&scale_bits, ptr + b * (2 + BLOCK), 2);
                float scale = fp16_to_fp32(scale_bits);
                const uint8_t* block_ptr = ptr + b * (2 + BLOCK) + 2;
                for (size_t j = 0; j < BLOCK; j++) {
                    size_t idx = b * BLOCK + j;
                    if (idx >= n_total) break;
                    int8_t v;
                    if (qtype == QuantType::Q8_0) {
                        v = (int8_t)block_ptr[j];
                    } else {
                        // 一个字节包含两个 4-bit 值
                        uint8_t byte = block_ptr[j / 2];
                        int8_t q;
                        if (j % 2 == 0) {           // low nibble
                            q = (int8_t)((byte & 0x0F) << 4) >> 4;
                        } else {                     // high nibble
                            q = (int8_t)byte >> 4;
                        }
                        v = q;
                    }
                    out[idx] = scale * (float)v;
                }
            }
            break;
        }
        default:
            out.assign(n_total, 0.0f);
    }
    return out;
}

// rotate_half 版 RoPE（非交错），与 pocket_hf.py 一致
// vec[head_dim], pos 位置
void apply_rope_pocket(float* vec, size_t head_dim, pos_t pos, float theta) {
    size_t half = head_dim / 2;
    for (size_t i = 0; i < half; i++) {
        double freq = 1.0 / std::pow(theta, (2.0 * i) / head_dim);
        double angle = pos * freq;
        float c = (float)std::cos(angle);
        float s = (float)std::sin(angle);
        float x1 = vec[i];
        float x2 = vec[i + half];
        vec[i]       = x1 * c - x2 * s;
        vec[i + half] = x2 * c + x1 * s;
    }
}

} // namespace

// ============================================================================
// Model::load - 解析新 .pllm 格式
// ============================================================================
bool Model::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Model] 无法打开文件: " << path << std::endl;
        return false;
    }

    // 1. Magic
    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "PLLM", 4) != 0) {
        std::cerr << "[Model] 非法的 magic（不是 .pllm 文件）" << std::endl;
        return false;
    }

    // 2. Version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "[Model] 不支持的文件版本: " << version << std::endl;
        return false;
    }

    // 3. header_size（config_section + vocab_section 字节数）
    uint32_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    // 4. config_section: config_size(uint32) + config_json
    uint32_t config_size;
    file.read(reinterpret_cast<char*>(&config_size), sizeof(config_size));
    std::string config_json(config_size, '\0');
    file.read(&config_json[0], config_size);

    // 解析模型配置
    config_.vocab_size          = (int32_t)json_int(config_json, "vocab_size", 6400);
    config_.hidden_size         = (int32_t)json_int(config_json, "hidden_size", 768);
    config_.num_layers          = (int32_t)json_int(config_json, "num_hidden_layers", 12);
    config_.num_heads           = (int32_t)json_int(config_json, "num_attention_heads", 12);
    config_.num_kv_heads        = (int32_t)json_int(config_json, "num_key_value_heads", config_.num_heads);
    config_.intermediate_size   = (int32_t)json_int(config_json, "intermediate_size", 2048);
    config_.max_seq_len         = (int32_t)json_int(config_json, "max_position_embeddings", 2048);
    config_.rms_norm_eps        = (float)json_double(config_json, "rms_norm_eps", 1e-6);
    config_.rope_theta          = (float)json_double(config_json, "rope_theta", 10000.0);
    config_.bos_token_id        = (token_t)json_int(config_json, "bos_token_id", 1);
    config_.eos_token_id        = (token_t)json_int(config_json, "eos_token_id", 2);
    config_.pad_token_id        = (token_t)json_int(config_json, "pad_token_id", 0);

    // 5. vocab_section: vocab_size(uint32) + 每个 token len(uint32)+utf8
    uint32_t vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    vocab_.resize(vocab_size);
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint32_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string token(len, '\0');
        file.read(&token[0], len);
        vocab_[i] = std::move(token);
    }

    // 6. tensor metadata
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
    tensors_.resize(num_tensors);
    for (uint32_t i = 0; i < num_tensors; i++) {
        Tensor& tensor = tensors_[i];
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        tensor.name.resize(name_len);
        file.read(&tensor.name[0], name_len);

        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        tensor.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            tensor.shape[d] = static_cast<size_t>(dim);
        }

        uint32_t qtype;
        file.read(reinterpret_cast<char*>(&qtype), sizeof(qtype));
        tensor.qtype = static_cast<QuantType>(qtype);

        uint64_t offset, size;
        file.read(reinterpret_cast<char*>(&offset), sizeof(offset));
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        tensor.offset = static_cast<size_t>(offset);
    }

    // 7. data 区：直接定位到数据起始（当前文件指针）
    size_t data_start = static_cast<size_t>(file.tellg());

    // 读取整块数据
    // offset 相对 data 区起点连续累计；每个 tensor 按 offset seek 后单独反量化。
    dequant_data_.resize(num_tensors);

    for (uint32_t i = 0; i < num_tensors; i++) {
        const Tensor& tensor = tensors_[i];
        file.seekg(data_start + tensor.offset);
        std::vector<uint8_t> raw(tensor.nbytes());
        if (!raw.empty()) {
            file.read(reinterpret_cast<char*>(raw.data()), raw.size());
            // n_total = 块对齐后的元素数（与导出端一致）
            size_t n_total = tensor.numel();
            if (tensor.qtype == QuantType::Q8_0 || tensor.qtype == QuantType::Q4_0) {
                n_total = ((n_total + 31) / 32) * 32;
            }
            dequant_data_[i] = dequantize(raw.data(), tensor.qtype, n_total);
            // 裁剪掉块填充的多余元素
            dequant_data_[i].resize(tensor.numel());
        }
        // 指向反量化数据
        const_cast<Tensor&>(tensor).data = dequant_data_[i].data();
    }

    file.close();
    std::cout << "[Model] 加载成功: " << path
              << " (" << num_tensors << " 张量, config vocab=" << config_.vocab_size
              << " hidden=" << config_.hidden_size
              << " layers=" << config_.num_layers << ")" << std::endl;
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

// 权重便捷访问（[out, in] 布局）
static const float* W(const Model* m, const std::string& name) {
    const Tensor* t = m->get_tensor(name);
    if (!t) {
        std::cerr << "[Model] 缺少权重: " << name << std::endl;
        throw std::runtime_error("missing tensor");
    }
    return reinterpret_cast<const float*>(t->data);
}

// ============================================================================
// Model::forward - 前向传播（逐 token，复用 KV cache）
// ============================================================================
std::vector<float> Model::forward(
    const std::vector<token_t>& input_ids,
    KVCache* kv_cache
) {
    if (input_ids.empty() || tensors_.empty()) {
        throw std::runtime_error("Model 未加载或输入为空");
    }

    const size_t hidden        = (size_t)config_.hidden_size;
    const size_t num_layers    = (size_t)config_.num_layers;
    const size_t num_heads     = (size_t)config_.num_heads;
    const size_t num_kv_heads  = (size_t)config_.num_kv_heads;
    const size_t head_dim      = hidden / num_heads;
    const size_t n_rep         = num_heads / num_kv_heads;
    const size_t max_seq       = (size_t)config_.max_seq_len;
    const size_t inter         = (size_t)config_.intermediate_size;

    const size_t seq_len = input_ids.size();

    // 未传入 cache 时，用临时 KV cache 走同一套逻辑（无缓存全量前向）
    KVCache local_kv;
    if (!kv_cache) kv_cache = &local_kv;

    // 已在 cache 中的位置数（复用部分）
    size_t start = 0;
    {
        start = kv_cache->current_seq_len < seq_len ? kv_cache->current_seq_len : seq_len;
        size_t per_layer = max_seq * num_kv_heads * head_dim;
        size_t need = num_layers * per_layer;
        if (kv_cache->k_cache.size() < need) kv_cache->k_cache.resize(need, 0.0f);
        if (kv_cache->v_cache.size() < need) kv_cache->v_cache.assign(need, 0.0f);
    }

    const size_t n_new = seq_len - start;  // 本次需要处理的新 token 数

    // 权重指针
    const float* embed_w   = W(this, "model.embed_tokens.weight");
    const float* final_norm_w = W(this, "model.norm.weight");
    const float* lm_head_w = W(this, "lm_head.weight");

    // hidden states：仅保留新 token 行，逐 token 前向
    std::vector<float> hids(n_new * hidden);         // 当前层的输入（残差累积）
    std::vector<float> norm_buf(n_new * hidden);       // input RMSNorm 输出
    std::vector<float> attn_out(n_new * hidden);       // attention 输出
    std::vector<float> ffn_in(n_new * hidden);          // post-norm 输出（FFN 输入）
    std::vector<float> gate_buf(n_new * inter);         // gate
    std::vector<float> up_buf(n_new * inter);           // up
    std::vector<float> gated(n_new * inter);            // silu(gate) * up
    std::vector<float> q(num_heads * head_dim);
    std::vector<float> k(num_kv_heads * head_dim);
    std::vector<float> v(num_kv_heads * head_dim);
    std::vector<float> scores(max_seq);
    std::vector<float> attn_vec(num_heads * head_dim);  // 拼接后的多头输出

    for (size_t l = 0; l < num_layers; l++) {
        const std::string ln = "model.layers." + std::to_string(l) + ".";
        const float* q_w = W(this, ln + "self_attn.q_proj.weight");
        const float* k_w = W(this, ln + "self_attn.k_proj.weight");
        const float* v_w = W(this, ln + "self_attn.v_proj.weight");
        const float* o_w = W(this, ln + "self_attn.o_proj.weight");
        const float* ln_in_w  = W(this, ln + "input_layernorm.weight");
        const float* ln_post_w = W(this, ln + "post_attention_layernorm.weight");
        const float* g_w = W(this, ln + "mlp.gate_proj.weight");
        const float* u_w = W(this, ln + "mlp.up_proj.weight");
        const float* d_w = W(this, ln + "mlp.down_proj.weight");

        // QK-norm（可选，MiniMind 风格）：权重存在才启用，不影响没有该结构的模型
        const Tensor* q_norm_t = get_tensor(ln + "self_attn.q_norm.weight");
        const Tensor* k_norm_t = get_tensor(ln + "self_attn.k_norm.weight");
        const float* q_norm_w = q_norm_t ? reinterpret_cast<const float*>(q_norm_t->data) : nullptr;
        const float* k_norm_w = k_norm_t ? reinterpret_cast<const float*>(k_norm_t->data) : nullptr;

        const float* k_cache = kv_cache ? kv_cache->k_cache.data() + l * max_seq * num_kv_heads * head_dim : nullptr;
        const float* v_cache = kv_cache ? kv_cache->v_cache.data() + l * max_seq * num_kv_heads * head_dim : nullptr;

        // 第一层：从 embedding 取 hidden；后续层：上一次循环写入的 hidden 保留
        // （embedding 只做一次，在第一个层之前）
        for (size_t nt = 0; nt < n_new; nt++) {
            // hids[nt] 已在上一轮 layer 结束时写入（除了第一层）
            if (l == 0) {
                token_t tok = input_ids[start + nt];
                if (tok < 0 || (size_t)tok >= config_.vocab_size) tok = 0;
                std::memcpy(&hids[nt * hidden], embed_w + (size_t)tok * hidden, hidden * sizeof(float));
            }

            // ===== Input RMSNorm =====
            ops::rmsnorm(&hids[nt * hidden], ln_in_w, &norm_buf[nt * hidden], hidden, config_.rms_norm_eps);
        }

        // ===== Attention（逐 token）=====
        for (size_t nt = 0; nt < n_new; nt++) {
            pos_t cur_pos = (pos_t)(start + nt);
            const float* x = &norm_buf[nt * hidden];

            // Q/K/V 投影（matvec: [out] = W[out][in] @ x，即 x @ W^T）
            ops::matvec(q_w, x, q.data(), num_heads * head_dim, hidden);
            ops::matvec(k_w, x, k.data(), num_kv_heads * head_dim, hidden);
            ops::matvec(v_w, x, v.data(), num_kv_heads * head_dim, hidden);

            // QK-norm（按 head 归一化，必须在 RoPE 之前）
            if (q_norm_w) {
                for (size_t h = 0; h < num_heads; h++)
                    ops::rmsnorm(q.data() + h * head_dim, q_norm_w, q.data() + h * head_dim,
                                 head_dim, config_.rms_norm_eps);
            }
            if (k_norm_w) {
                for (size_t h = 0; h < num_kv_heads; h++)
                    ops::rmsnorm(k.data() + h * head_dim, k_norm_w, k.data() + h * head_dim,
                                 head_dim, config_.rms_norm_eps);
            }

            // RoPE（rotate_half 语义）
            for (size_t h = 0; h < num_heads; h++) apply_rope_pocket(q.data() + h * head_dim, head_dim, cur_pos, config_.rope_theta);
            for (size_t h = 0; h < num_kv_heads; h++) apply_rope_pocket(k.data() + h * head_dim, head_dim, cur_pos, config_.rope_theta);

            // 写入 KV cache（若启用）
            if (kv_cache) {
                for (size_t h = 0; h < num_kv_heads; h++) {
                    std::memcpy(const_cast<float*>(k_cache) + ((cur_pos * num_kv_heads) + h) * head_dim,
                                k.data() + h * head_dim, head_dim * sizeof(float));
                    std::memcpy(const_cast<float*>(v_cache) + ((cur_pos * num_kv_heads) + h) * head_dim,
                                v.data() + h * head_dim, head_dim * sizeof(float));
                }
            }

            // 注意力的 K/V 源（cache 或本层本 token）
            const float* K = k_cache ? k_cache : k.data();
            const float* V = v_cache ? v_cache : v.data();
            size_t cache_len = start + nt + 1;  // 可见位置数

            const float scale = 1.0f / std::sqrt((float)head_dim);
            for (size_t qh = 0; qh < num_heads; qh++) {
                const float* q_h = q.data() + qh * head_dim;
                size_t kvh = qh / n_rep;

                // scores
                for (size_t t = 0; t < cache_len; t++) {
                    const float* K_t = K + ((t * num_kv_heads) + kvh) * head_dim;
                    float sc = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) sc += q_h[d] * K_t[d];
                    scores[t] = sc * scale;
                }
                // softmax
                ops::softmax(scores.data(), cache_len);

                // attn @ V
                float* out_h = attn_vec.data() + qh * head_dim;
                std::memset(out_h, 0, head_dim * sizeof(float));
                for (size_t t = 0; t < cache_len; t++) {
                    const float* V_t = V + ((t * num_kv_heads) + kvh) * head_dim;
                    float w = scores[t];
                    for (size_t d = 0; d < head_dim; d++) out_h[d] += w * V_t[d];
                }
            }

            // 输出投影: attn_out = attn_vec @ W_o^T
            ops::matvec(o_w, attn_vec.data(), &attn_out[nt * hidden], hidden, num_heads * head_dim);
        }

        // 残差
        for (size_t nt = 0; nt < n_new; nt++)
            for (size_t d = 0; d < hidden; d++)
                hids[nt * hidden + d] += attn_out[nt * hidden + d];

        // Post-norm
        for (size_t nt = 0; nt < n_new; nt++)
            ops::rmsnorm(&hids[nt * hidden], ln_post_w, &ffn_in[nt * hidden], hidden, config_.rms_norm_eps);

        // FFN: gate / up 投影
        for (size_t nt = 0; nt < n_new; nt++) {
            const float* x = &ffn_in[nt * hidden];
            ops::matvec(g_w, x, &gate_buf[nt * inter], inter, hidden);
            ops::matvec(u_w, x, &up_buf[nt * inter],     inter, hidden);
        }
        // silu(gate) * up
        for (size_t nt = 0; nt < n_new; nt++)
            for (size_t d = 0; d < inter; d++) {
                float g = gate_buf[nt * inter + d];
                gated[nt * inter + d] = (g / (1.0f + std::exp(-g))) * up_buf[nt * inter + d];
            }
        // down 投影回 hidden 维度
        for (size_t nt = 0; nt < n_new; nt++)
            ops::matvec(d_w, &gated[nt * inter], &ffn_in[nt * hidden], hidden, inter);

        // 残差
        for (size_t nt = 0; nt < n_new; nt++)
            for (size_t d = 0; d < hidden; d++)
                hids[nt * hidden + d] += ffn_in[nt * hidden + d];
    }

    // 更新 cache 位置计数
    kv_cache->current_seq_len = seq_len;

    // 最终 RMSNorm + LM head（最后一个 token）
    std::vector<float> final_vec(1 * hidden);
    ops::rmsnorm(&hids[(n_new - 1) * hidden], final_norm_w, final_vec.data(), hidden, config_.rms_norm_eps);

    std::vector<float> logits((size_t)config_.vocab_size);
    ops::matvec(lm_head_w, final_vec.data(), logits.data(), (size_t)config_.vocab_size, hidden);
    return logits;
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
    std::cout << "  Total parameters: "
              << (4 * config.hidden_size * config.hidden_size * config.num_layers
                  + 3 * config.hidden_size * config.intermediate_size * config.num_layers
                  + 2 * config.vocab_size * config.hidden_size) / 1000000.0
              << "M" << std::endl;
}

std::string version() {
    return "Pocket-0.1";
}

}