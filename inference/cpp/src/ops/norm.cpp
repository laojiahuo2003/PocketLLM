/**
 * RMSNorm (Root Mean Square Layer Normalization)
 *
 * RMSNorm 是 LayerNorm 的简化版本，只做缩放，不做平移
 * 计算公式: y = (x / RMS(x)) * weight
 * 其中 RMS(x) = sqrt(mean(x^2) + eps)
 */

#include "pocket.h"
#include <cmath>

#ifdef POCKET_ARM_NEON
#include <arm_neon.h>
#endif

namespace pocket {
namespace ops {

/**
 * 标量版本 RMSNorm
 */
void rmsnorm_scalar(
    const float* input,
    const float* weight,
    float* output,
    size_t size,
    float eps
) {
    // 1. 计算平方和
    float sum_squares = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum_squares += input[i] * input[i];
    }

    // 2. 计算 RMS
    float rms = sqrtf(sum_squares / size + eps);

    // 3. 归一化并乘以权重
    for (size_t i = 0; i < size; i++) {
        output[i] = (input[i] / rms) * weight[i];
    }
}

#ifdef POCKET_ARM_NEON
/**
 * ARM NEON 优化版本
 */
void rmsnorm_neon(
    const float* input,
    const float* weight,
    float* output,
    size_t size,
    float eps
) {
    // 1. 计算平方和（向量化）
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 3 < size; i += 4) {
        float32x4_t x_vec = vld1q_f32(input + i);
        sum_vec = vmlaq_f32(sum_vec, x_vec, x_vec);  // sum += x * x
    }

    // 水平求和
    float sum_squares = vaddvq_f32(sum_vec);

    // 处理剩余元素
    for (; i < size; i++) {
        sum_squares += input[i] * input[i];
    }

    // 2. 计算 RMS
    float rms = sqrtf(sum_squares / size + eps);
    float inv_rms = 1.0f / rms;

    // 3. 归一化并乘以权重（向量化）
    float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
    i = 0;

    for (; i + 3 < size; i += 4) {
        float32x4_t x_vec = vld1q_f32(input + i);
        float32x4_t w_vec = vld1q_f32(weight + i);
        float32x4_t y_vec = vmulq_f32(vmulq_f32(x_vec, inv_rms_vec), w_vec);
        vst1q_f32(output + i, y_vec);
    }

    // 处理剩余元素
    for (; i < size; i++) {
        output[i] = (input[i] * inv_rms) * weight[i];
    }
}
#endif

/**
 * RMSNorm 主接口
 */
void rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    size_t size,
    float eps
) {
#ifdef POCKET_ARM_NEON
    rmsnorm_neon(input, weight, output, size, eps);
#else
    rmsnorm_scalar(input, weight, output, size, eps);
#endif
}

/**
 * 批量 RMSNorm
 *
 * 对多个向量分别做 RMSNorm
 * input: [batch_size, size]
 * weight: [size]
 * output: [batch_size, size]
 */
void batch_rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    size_t batch_size,
    size_t size,
    float eps
) {
    for (size_t b = 0; b < batch_size; b++) {
        rmsnorm(
            input + b * size,
            weight,
            output + b * size,
            size,
            eps
        );
    }
}

} // namespace ops
} // namespace pocket
