/**
 * SwiGLU Feed-Forward Network
 *
 * SwiGLU 是一种改进的激活函数，用于 FFN
 * 公式: FFN(x) = (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down
 * 其中 Swish(x) = x * sigmoid(x) = x * σ(x)
 */

#include "pocket.h"
#include <cmath>
#include <cstring>

#ifdef POCKET_ARM_NEON
#include <arm_neon.h>
#endif

namespace pocket {
namespace ops {

// 前向声明
void matvec(const float* A, const float* x, float* y, size_t m, size_t k);

/**
 * SiLU (Swish) 激活函数
 *
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * SiLU 的向量版本
 */
void silu_inplace(float* x, size_t size) {
#ifdef POCKET_ARM_NEON
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);

        // sigmoid(x) = 1 / (1 + exp(-x))
        float32x4_t neg_v = vnegq_f32(v);

        // 这里简化处理，实际可以用更快的近似
        float sigmoid[4];
        for (int j = 0; j < 4; j++) {
            sigmoid[j] = 1.0f / (1.0f + expf(-x[i + j]));
        }
        float32x4_t sig_v = vld1q_f32(sigmoid);

        // silu(x) = x * sigmoid(x)
        float32x4_t result = vmulq_f32(v, sig_v);
        vst1q_f32(x + i, result);
    }

    for (; i < size; i++) {
        x[i] = silu(x[i]);
    }
#else
    for (size_t i = 0; i < size; i++) {
        x[i] = silu(x[i]);
    }
#endif
}

/**
 * 逐元素乘法
 */
void elementwise_mul(
    const float* a,
    const float* b,
    float* output,
    size_t size
) {
#ifdef POCKET_ARM_NEON
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t result = vmulq_f32(a_vec, b_vec);
        vst1q_f32(output + i, result);
    }

    for (; i < size; i++) {
        output[i] = a[i] * b[i];
    }
#else
    for (size_t i = 0; i < size; i++) {
        output[i] = a[i] * b[i];
    }
#endif
}

/**
 * SwiGLU FFN
 *
 * 输入:
 *   x: [seq_len, hidden_size]
 *   gate_weight: [hidden_size, intermediate_size]
 *   up_weight: [hidden_size, intermediate_size]
 *   down_weight: [intermediate_size, hidden_size]
 *
 * 输出:
 *   output: [seq_len, hidden_size]
 */
void swiglu_ffn(
    const float* x,
    const float* gate_weight,
    const float* up_weight,
    const float* down_weight,
    float* output,
    size_t seq_len,
    size_t hidden_size,
    size_t intermediate_size,
    float* workspace  // 临时工作空间
) {
    // 分配工作空间
    float* gate = workspace;
    float* up = gate + seq_len * intermediate_size;
    float* gated = up + seq_len * intermediate_size;

    // 对每个 token
    for (size_t i = 0; i < seq_len; i++) {
        const float* x_i = x + i * hidden_size;
        float* gate_i = gate + i * intermediate_size;
        float* up_i = up + i * intermediate_size;
        float* gated_i = gated + i * intermediate_size;
        float* output_i = output + i * hidden_size;

        // 1. gate = x @ W_gate
        matvec(gate_weight, x_i, gate_i, intermediate_size, hidden_size);

        // 2. up = x @ W_up
        matvec(up_weight, x_i, up_i, intermediate_size, hidden_size);

        // 3. gate = silu(gate)
        silu_inplace(gate_i, intermediate_size);

        // 4. gated = gate ⊙ up (逐元素乘法)
        elementwise_mul(gate_i, up_i, gated_i, intermediate_size);

        // 5. output = gated @ W_down
        matvec(down_weight, gated_i, output_i, hidden_size, intermediate_size);
    }
}

/**
 * 单个 token 的 SwiGLU FFN（用于生成）
 */
void swiglu_ffn_single(
    const float* x,           // [hidden_size]
    const float* gate_weight,
    const float* up_weight,
    const float* down_weight,
    float* output,            // [hidden_size]
    size_t hidden_size,
    size_t intermediate_size,
    float* workspace
) {
    float* gate = workspace;
    float* up = gate + intermediate_size;
    float* gated = up + intermediate_size;

    // 1. gate = x @ W_gate
    matvec(gate_weight, x, gate, intermediate_size, hidden_size);

    // 2. up = x @ W_up
    matvec(up_weight, x, up, intermediate_size, hidden_size);

    // 3. gate = silu(gate)
    silu_inplace(gate, intermediate_size);

    // 4. gated = gate ⊙ up
    elementwise_mul(gate, up, gated, intermediate_size);

    // 5. output = gated @ W_down
    matvec(down_weight, gated, output, hidden_size, intermediate_size);
}

/**
 * 标准 FFN（带 ReLU/GELU）
 *
 * 如果不使用 SwiGLU，可以使用这个
 */
void standard_ffn(
    const float* x,
    const float* w1,
    const float* w2,
    float* output,
    size_t seq_len,
    size_t hidden_size,
    size_t intermediate_size,
    float* workspace,
    bool use_gelu = false
) {
    float* hidden = workspace;

    for (size_t i = 0; i < seq_len; i++) {
        const float* x_i = x + i * hidden_size;
        float* hidden_i = hidden + i * intermediate_size;
        float* output_i = output + i * hidden_size;

        // 1. hidden = x @ W1
        matvec(w1, x_i, hidden_i, intermediate_size, hidden_size);

        // 2. 激活函数
        if (use_gelu) {
            // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            for (size_t j = 0; j < intermediate_size; j++) {
                float x = hidden_i[j];
                float x3 = x * x * x;
                float inner = 0.7978845608f * (x + 0.044715f * x3);
                hidden_i[j] = 0.5f * x * (1.0f + tanhf(inner));
            }
        } else {
            // ReLU
            for (size_t j = 0; j < intermediate_size; j++) {
                hidden_i[j] = std::max(0.0f, hidden_i[j]);
            }
        }

        // 3. output = hidden @ W2
        matvec(w2, hidden_i, output_i, hidden_size, intermediate_size);
    }
}

} // namespace ops
} // namespace pocket
