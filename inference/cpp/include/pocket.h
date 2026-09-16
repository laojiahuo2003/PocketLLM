/**
 * Pocket-0.1 C++ Inference Engine
 *
 * 轻量级、高性能的 LLM 推理引擎，专为移动端设计
 *
 * 核心特性：
 * - 纯 C++11，无外部依赖
 * - 支持量化（FP32/FP16/INT8）
 * - ARM NEON 优化
 * - KV Cache 优化
 * - 内存高效
 */

#ifndef POCKET_H
#define POCKET_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

namespace pocket {

// ============================================================================
// 基础类型定义
// ============================================================================

using token_t = int32_t;
using pos_t = int32_t;

// 量化类型
enum class QuantType {
    F32,    // 32-bit float
    F16,    // 16-bit float
    Q8_0,   // 8-bit quantization (block-wise)
    Q4_0,   // 4-bit quantization (block-wise)
};

// 设备类型
enum class DeviceType {
    CPU,
    CUDA,   // 未来支持
    METAL,  // 未来支持
};

// ============================================================================
// Tensor - 张量
// ============================================================================

/**
 * 轻量级张量
 *
 * 只存储元数据，实际数据由 Model 统一管理
 */
struct Tensor {
    std::string name;
    std::vector<size_t> shape;  // [dim0, dim1, ...]
    QuantType qtype;
    void* data;                  // 指向实际数据
    size_t offset;               // 在文件中的偏移量

    // 获取元素总数
    size_t numel() const {
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    // 获取字节大小
    size_t nbytes() const;

    // 获取维度
    size_t ndim() const { return shape.size(); }
};

// ============================================================================
// ModelConfig - 模型配置
// ============================================================================

struct ModelConfig {
    // 架构
    std::string architecture = "pocket";
    int32_t version = 1;

    // 模型超参数
    int32_t vocab_size = 6400;
    int32_t hidden_size = 768;
    int32_t num_layers = 12;
    int32_t num_heads = 12;
    int32_t num_kv_heads = 4;        // GQA
    int32_t intermediate_size = 2048;
    int32_t max_seq_len = 2048;

    // 归一化
    float rms_norm_eps = 1e-6f;

    // RoPE
    float rope_theta = 10000.0f;

    // 特殊 token
    token_t bos_token_id = 1;
    token_t eos_token_id = 2;
    token_t pad_token_id = 0;

    // 量化
    QuantType quant_type = QuantType::F32;
};

// ============================================================================
// Tokenizer - 分词器
// ============================================================================

/**
 * 简化的 BPE 分词器
 */
class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // 从文件加载
    bool load(const std::string& path);

    // 编码
    std::vector<token_t> encode(const std::string& text, bool add_special_tokens = true);

    // 解码
    std::string decode(const std::vector<token_t>& tokens, bool skip_special_tokens = true);
    std::string decode(token_t token);

    // 词表大小
    size_t vocab_size() const { return vocab_.size(); }

    // 特殊 token
    token_t bos_token() const { return bos_token_; }
    token_t eos_token() const { return eos_token_; }
    token_t pad_token() const { return pad_token_; }

private:
    std::vector<std::string> vocab_;
    token_t bos_token_ = 1;
    token_t eos_token_ = 2;
    token_t pad_token_ = 0;

    // BPE merge rules (简化版)
    // TODO: 完整实现
};

// ============================================================================
// KVCache - KV 缓存
// ============================================================================

/**
 * KV Cache 用于加速推理
 */
struct KVCache {
    std::vector<float> k_cache;  // [num_layers, max_seq_len, num_kv_heads, head_dim]
    std::vector<float> v_cache;  // [num_layers, max_seq_len, num_kv_heads, head_dim]
    size_t current_seq_len = 0;

    void clear() {
        k_cache.clear();
        v_cache.clear();
        current_seq_len = 0;
    }
};

// ============================================================================
// Model - 模型
// ============================================================================

/**
 * Pocket 模型
 *
 * 管理模型权重和推理
 */
class Model {
public:
    Model() = default;
    ~Model() = default;

    // 从文件加载模型
    bool load(const std::string& path);

    // 前向传播
    std::vector<float> forward(
        const std::vector<token_t>& input_ids,
        KVCache* kv_cache = nullptr
    );

    // 获取配置
    const ModelConfig& config() const { return config_; }

    // 获取张量
    const Tensor* get_tensor(const std::string& name) const;

private:
    ModelConfig config_;
    std::vector<Tensor> tensors_;
    std::vector<uint8_t> weights_data_;  // 所有权重数据

    // 内部前向传播函数
    void embedding(const std::vector<token_t>& input_ids, float* output);
    void rms_norm(const float* input, const float* weight, float* output, size_t size);
    void rope(float* q, float* k, pos_t pos, size_t head_dim);
    void attention(
        const float* input,
        const float* q_weight,
        const float* k_weight,
        const float* v_weight,
        const float* o_weight,
        float* output,
        KVCache* kv_cache,
        int layer_idx
    );
    void swiglu_ffn(
        const float* input,
        const float* gate_weight,
        const float* up_weight,
        const float* down_weight,
        float* output
    );
};

// ============================================================================
// Sampler - 采样器
// ============================================================================

/**
 * 采样策略
 */
struct SamplerConfig {
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 50;
    float repetition_penalty = 1.0f;
    uint32_t seed = 0;
};

/**
 * 采样器
 */
class Sampler {
public:
    explicit Sampler(const SamplerConfig& config) : config_(config) {}

    // 从 logits 采样下一个 token
    token_t sample(const std::vector<float>& logits);

    // Greedy search
    token_t argmax(const std::vector<float>& logits);

    // Top-k sampling
    token_t sample_top_k(std::vector<float> logits, int k);

    // Top-p (nucleus) sampling
    token_t sample_top_p(std::vector<float> logits, float p);

private:
    SamplerConfig config_;
};

// ============================================================================
// Generator - 生成器
// ============================================================================

/**
 * 文本生成配置
 */
struct GenerateConfig {
    int max_new_tokens = 100;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 50;
    float repetition_penalty = 1.0f;
    bool use_cache = true;
    uint32_t seed = 0;

    // 回调函数（用于流式输出）
    using callback_t = void (*)(token_t token, void* user_data);
    callback_t callback = nullptr;
    void* callback_data = nullptr;
};

/**
 * 文本生成器
 */
class Generator {
public:
    Generator(Model* model, Tokenizer* tokenizer)
        : model_(model), tokenizer_(tokenizer) {}

    // 生成文本
    std::string generate(
        const std::string& prompt,
        const GenerateConfig& config = GenerateConfig()
    );

    // 生成 token 序列
    std::vector<token_t> generate_tokens(
        const std::vector<token_t>& prompt_tokens,
        const GenerateConfig& config = GenerateConfig()
    );

private:
    Model* model_;
    Tokenizer* tokenizer_;
    KVCache kv_cache_;
};

// ============================================================================
// 工具函数
// ============================================================================

// 打印模型信息
void print_model_info(const Model& model);

// 打印张量信息
void print_tensor_info(const Tensor& tensor);

// 获取版本信息
std::string version();

} // namespace pocket

#endif // POCKET_H
