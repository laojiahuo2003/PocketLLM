/**
 * RoPE (Rotary Position Embedding)
 *
 * 旋转位置编码，通过旋转矩阵为 Q 和 K 添加位置信息
 */

#include "pocket.h"
#include <cmath>
#include <vector>

#ifdef POCKET_ARM_NEON
#include <arm_neon.h>
#endif

namespace pocket {
namespace ops {

/**
 * 预计算 RoPE 的 cos 和 sin 值
 */
struct RoPECache {
    std::vector<float> cos_cached;
    std::vector<float> sin_cached;
    size_t max_seq_len;
    size_t head_dim;
    float theta;

    RoPECache(size_t max_seq_len, size_t head_dim, float theta = 10000.0f)
        : max_seq_len(max_seq_len), head_dim(head_dim), theta(theta) {

        cos_cached.resize(max_seq_len * head_dim);
        sin_cached.resize(max_seq_len * head_dim);

        // 计算频率
        for (size_t i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);

            for (size_t pos = 0; pos < max_seq_len; pos++) {
                float angle = pos * freq;

                // 每个维度对应两个位置（实部和虚部）
                cos_cached[pos * head_dim + i * 2] = cosf(angle);
                cos_cached[pos * head_dim + i * 2 + 1] = cosf(angle);
                sin_cached[pos * head_dim + i * 2] = sinf(angle);
                sin_cached[pos * head_dim + i * 2 + 1] = sinf(angle);
            }
        }
    }
};

// 全局 RoPE 缓存
static RoPECache* g_rope_cache = nullptr;

/**
 * 初始化 RoPE 缓存
 */
void init_rope_cache(size_t max_seq_len, size_t head_dim, float theta) {
    if (g_rope_cache == nullptr) {
        g_rope_cache = new RoPECache(max_seq_len, head_dim, theta);
    }
}

/**
 * 旋转半个向量
 */
inline void rotate_half(float* x, size_t head_dim) {
    for (size_t i = 0; i < head_dim / 2; i++) {
        float tmp = x[i];
        x[i] = -x[i + head_dim / 2];
        x[i + head_dim / 2] = tmp;
    }
}

/**
 * 应用 RoPE 到单个向量
 */
void apply_rope_single(
    float* vec,
    const float* cos,
    const float* sin,
    size_t head_dim
) {
    std::vector<float> rotated(head_dim);

    // 旋转操作
    for (size_t i = 0; i < head_dim / 2; i++) {
        size_t idx1 = i * 2;
        size_t idx2 = i * 2 + 1;

        // 复数乘法
        rotated[idx1] = vec[idx1] * cos[idx1] - vec[idx2] * sin[idx1];
        rotated[idx2] = vec[idx1] * sin[idx2] + vec[idx2] * cos[idx2];
    }

    // 拷贝回去
    for (size_t i = 0; i < head_dim; i++) {
        vec[i] = rotated[i];
    }
}

#ifdef POCKET_ARM_NEON
/**
 * ARM NEON 优化版本
 */
void apply_rope_single_neon(
    float* vec,
    const float* cos,
    const float* sin,
    size_t head_dim
) {
    for (size_t i = 0; i < head_dim; i += 4) {
        float32x4_t v = vld1q_f32(vec + i);
        float32x4_t c = vld1q_f32(cos + i);
        float32x4_t s = vld1q_f32(sin + i);

        // 创建旋转后的向量
        float32x4_t v_rot = {-vec[i + 2], -vec[i + 3], vec[i], vec[i + 1]};

        // result = v * cos + v_rot * sin
        float32x4_t result = vmlaq_f32(vmulq_f32(v, c), v_rot, s);
        vst1q_f32(vec + i, result);
    }
}
#endif

/**
 * 应用 RoPE
 *
 * q: [batch_size, num_heads, seq_len, head_dim]
 * k: [batch_size, num_kv_heads, seq_len, head_dim]
 * position: 当前位置
 */
void apply_rope(
    float* q,
    float* k,
    size_t batch_size,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t seq_len,
    size_t head_dim,
    pos_t position
) {
    if (g_rope_cache == nullptr) {
        init_rope_cache(2048, head_dim, 10000.0f);
    }

    // 对 Q 应用 RoPE
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_q_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                pos_t pos = position + s;
                size_t offset = ((b * num_q_heads + h) * seq_len + s) * head_dim;

                const float* cos = &g_rope_cache->cos_cached[pos * head_dim];
                const float* sin = &g_rope_cache->sin_cached[pos * head_dim];

#ifdef POCKET_ARM_NEON
                apply_rope_single_neon(q + offset, cos, sin, head_dim);
#else
                apply_rope_single(q + offset, cos, sin, head_dim);
#endif
            }
        }
    }

    // 对 K 应用 RoPE
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < num_kv_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                pos_t pos = position + s;
                size_t offset = ((b * num_kv_heads + h) * seq_len + s) * head_dim;

                const float* cos = &g_rope_cache->cos_cached[pos * head_dim];
                const float* sin = &g_rope_cache->sin_cached[pos * head_dim];

#ifdef POCKET_ARM_NEON
                apply_rope_single_neon(k + offset, cos, sin, head_dim);
#else
                apply_rope_single(k + offset, cos, sin, head_dim);
#endif
            }
        }
    }
}

/**
 * 简化版 RoPE（用于单个位置）
 */
void apply_rope_inplace(
    float* q,
    float* k,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    pos_t position
) {
    if (g_rope_cache == nullptr) {
        init_rope_cache(2048, head_dim, 10000.0f);
    }

    const float* cos = &g_rope_cache->cos_cached[position * head_dim];
    const float* sin = &g_rope_cache->sin_cached[position * head_dim];

    // 对所有 Q heads 应用
    for (size_t h = 0; h < num_q_heads; h++) {
        apply_rope_single(q + h * head_dim, cos, sin, head_dim);
    }

    // 对所有 K heads 应用
    for (size_t h = 0; h < num_kv_heads; h++) {
        apply_rope_single(k + h * head_dim, cos, sin, head_dim);
    }
}

} // namespace ops
} // namespace pocket
