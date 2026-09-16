/**
 * Attention 注意力机制
 *
 * 实现 Grouped Query Attention (GQA)
 */

#include "pocket.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace pocket {
namespace ops {

// 前向声明
void matmul_transposed(const float* A, const float* B, float* C, size_t m, size_t n, size_t k);
void matvec(const float* A, const float* x, float* y, size_t m, size_t k);

/**
 * Softmax
 */
void softmax(float* x, size_t size) {
    // 找最大值（数值稳定性）
    float max_val = x[0];
    for (size_t i = 1; i < size; i++) {
        max_val = std::max(max_val, x[i]);
    }

    // exp 和求和
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // 归一化
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

/**
 * 重复 KV heads（用于 GQA）
 *
 * 将 KV heads 复制 n_rep 次以匹配 Q heads 数量
 */
void repeat_kv(
    const float* input,
    float* output,
    size_t seq_len,
    size_t num_kv_heads,
    size_t head_dim,
    size_t n_rep
) {
    if (n_rep == 1) {
        // 不需要重复，直接复制
        size_t size = seq_len * num_kv_heads * head_dim;
        memcpy(output, input, size * sizeof(float));
        return;
    }

    // 重复每个 KV head
    for (size_t s = 0; s < seq_len; s++) {
        for (size_t kv_h = 0; kv_h < num_kv_heads; kv_h++) {
            const float* src = input + (s * num_kv_heads + kv_h) * head_dim;

            for (size_t r = 0; r < n_rep; r++) {
                size_t q_h = kv_h * n_rep + r;
                float* dst = output + (s * num_kv_heads * n_rep + q_h) * head_dim;
                memcpy(dst, src, head_dim * sizeof(float));
            }
        }
    }
}

/**
 * Scaled Dot-Product Attention (单头)
 *
 * Q: [seq_len_q, head_dim]
 * K: [seq_len_k, head_dim]
 * V: [seq_len_k, head_dim]
 * output: [seq_len_q, head_dim]
 */
void scaled_dot_product_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    size_t seq_len_q,
    size_t seq_len_k,
    size_t head_dim,
    float* attn_weights  // 临时缓冲区 [seq_len_q, seq_len_k]
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // 对每个 query 位置
    for (size_t i = 0; i < seq_len_q; i++) {
        const float* q = Q + i * head_dim;

        // 计算 attention scores: scores = Q @ K^T
        for (size_t j = 0; j < seq_len_k; j++) {
            const float* k = K + j * head_dim;
            float score = 0.0f;

            for (size_t d = 0; d < head_dim; d++) {
                score += q[d] * k[d];
            }

            attn_weights[i * seq_len_k + j] = score * scale;
        }

        // 因果掩码（只能看到当前及之前的位置）
        for (size_t j = i + 1; j < seq_len_k; j++) {
            attn_weights[i * seq_len_k + j] = -INFINITY;
        }

        // Softmax
        softmax(attn_weights + i * seq_len_k, seq_len_k);

        // 计算输出: output = attention_weights @ V
        float* out = output + i * head_dim;
        memset(out, 0, head_dim * sizeof(float));

        for (size_t j = 0; j < seq_len_k; j++) {
            float weight = attn_weights[i * seq_len_k + j];
            const float* v = V + j * head_dim;

            for (size_t d = 0; d < head_dim; d++) {
                out[d] += weight * v[d];
            }
        }
    }
}

/**
 * Multi-Head Attention
 *
 * 输入:
 *   x: [seq_len, hidden_size]
 *   q_weight: [hidden_size, num_heads * head_dim]
 *   k_weight: [hidden_size, num_kv_heads * head_dim]
 *   v_weight: [hidden_size, num_kv_heads * head_dim]
 *   o_weight: [num_heads * head_dim, hidden_size]
 *
 * 输出:
 *   output: [seq_len, hidden_size]
 */
void multi_head_attention(
    const float* x,
    const float* q_weight,
    const float* k_weight,
    const float* v_weight,
    const float* o_weight,
    float* output,
    size_t seq_len,
    size_t hidden_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float* workspace  // 临时工作空间
) {
    size_t n_rep = num_heads / num_kv_heads;  // GQA 重复因子

    // 分配工作空间
    float* Q = workspace;
    float* K = Q + seq_len * num_heads * head_dim;
    float* V = K + seq_len * num_kv_heads * head_dim;
    float* K_rep = V + seq_len * num_kv_heads * head_dim;
    float* V_rep = K_rep + seq_len * num_heads * head_dim;
    float* attn_output = V_rep + seq_len * num_heads * head_dim;
    float* attn_weights = attn_output + seq_len * num_heads * head_dim;

    // 1. 线性投影 Q, K, V
    // Q = x @ q_weight
    matmul_transposed(x, q_weight, Q, seq_len, num_heads * head_dim, hidden_size);

    // K = x @ k_weight
    matmul_transposed(x, k_weight, K, seq_len, num_kv_heads * head_dim, hidden_size);

    // V = x @ v_weight
    matmul_transposed(x, v_weight, V, seq_len, num_kv_heads * head_dim, hidden_size);

    // 2. 重复 KV heads（GQA）
    repeat_kv(K, K_rep, seq_len, num_kv_heads, head_dim, n_rep);
    repeat_kv(V, V_rep, seq_len, num_kv_heads, head_dim, n_rep);

    // 3. 对每个 head 计算 attention
    for (size_t h = 0; h < num_heads; h++) {
        const float* Q_h = Q + h * seq_len * head_dim;
        const float* K_h = K_rep + h * seq_len * head_dim;
        const float* V_h = V_rep + h * seq_len * head_dim;
        float* output_h = attn_output + h * seq_len * head_dim;

        scaled_dot_product_attention(
            Q_h, K_h, V_h, output_h,
            seq_len, seq_len, head_dim,
            attn_weights
        );
    }

    // 4. 合并 heads 并投影
    // output = attn_output @ o_weight
    matmul_transposed(attn_output, o_weight, output, seq_len, hidden_size, num_heads * head_dim);
}

/**
 * 带 KV Cache 的 Attention（用于生成）
 *
 * 只处理新的 token，复用之前的 KV
 */
void multi_head_attention_with_cache(
    const float* x,           // [1, hidden_size] - 只有一个新 token
    const float* q_weight,
    const float* k_weight,
    const float* v_weight,
    const float* o_weight,
    float* output,            // [1, hidden_size]
    float* k_cache,           // [max_seq_len, num_kv_heads, head_dim]
    float* v_cache,           // [max_seq_len, num_kv_heads, head_dim]
    size_t current_pos,       // 当前位置
    size_t hidden_size,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float* workspace
) {
    size_t n_rep = num_heads / num_kv_heads;

    // 分配工作空间
    float* Q = workspace;
    float* K_new = Q + num_heads * head_dim;
    float* V_new = K_new + num_kv_heads * head_dim;
    float* K_rep = V_new + num_kv_heads * head_dim;
    float* V_rep = K_rep + (current_pos + 1) * num_heads * head_dim;
    float* attn_output = V_rep + (current_pos + 1) * num_heads * head_dim;
    float* attn_weights = attn_output + num_heads * head_dim;

    // 1. 计算新 token 的 Q, K, V
    matvec(q_weight, x, Q, num_heads * head_dim, hidden_size);
    matvec(k_weight, x, K_new, num_kv_heads * head_dim, hidden_size);
    matvec(v_weight, x, V_new, num_kv_heads * head_dim, hidden_size);

    // 2. 更新 KV cache
    for (size_t h = 0; h < num_kv_heads; h++) {
        memcpy(
            k_cache + (current_pos * num_kv_heads + h) * head_dim,
            K_new + h * head_dim,
            head_dim * sizeof(float)
        );
        memcpy(
            v_cache + (current_pos * num_kv_heads + h) * head_dim,
            V_new + h * head_dim,
            head_dim * sizeof(float)
        );
    }

    // 3. 获取完整的 K, V（包括历史）
    size_t seq_len = current_pos + 1;

    repeat_kv(k_cache, K_rep, seq_len, num_kv_heads, head_dim, n_rep);
    repeat_kv(v_cache, V_rep, seq_len, num_kv_heads, head_dim, n_rep);

    // 4. 计算 attention（只对当前 query）
    for (size_t h = 0; h < num_heads; h++) {
        const float* Q_h = Q + h * head_dim;
        const float* K_h = K_rep + h * seq_len * head_dim;
        const float* V_h = V_rep + h * seq_len * head_dim;
        float* output_h = attn_output + h * head_dim;

        scaled_dot_product_attention(
            Q_h, K_h, V_h, output_h,
            1, seq_len, head_dim,
            attn_weights
        );
    }

    // 5. 投影输出
    matvec(o_weight, attn_output, output, hidden_size, num_heads * head_dim);
}

} // namespace ops
} // namespace pocket
