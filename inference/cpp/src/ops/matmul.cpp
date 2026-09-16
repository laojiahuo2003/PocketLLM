/**
 * 矩阵乘法算子
 *
 * 实现高效的矩阵乘法，支持：
 * - 普通矩阵乘法
 * - ARM NEON 优化
 * - 多线程并行
 */

#include "pocket.h"
#include <cstring>
#include <thread>
#include <vector>

#ifdef POCKET_ARM_NEON
#include <arm_neon.h>
#endif

namespace pocket {
namespace ops {

/**
 * 标量版本的矩阵乘法
 *
 * C = A @ B
 * A: [m, k]
 * B: [k, n]
 * C: [m, n]
 */
void matmul_scalar(
    const float* A,
    const float* B,
    float* C,
    size_t m,
    size_t n,
    size_t k
) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

#ifdef POCKET_ARM_NEON
/**
 * ARM NEON 优化的矩阵乘法
 */
void matmul_neon(
    const float* A,
    const float* B,
    float* C,
    size_t m,
    size_t n,
    size_t k
) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t p = 0;

            // 向量化内积（每次处理 4 个元素）
            for (; p + 3 < k; p += 4) {
                float32x4_t a_vec = vld1q_f32(A + i * k + p);
                float32x4_t b_vec = vld1q_f32(B + p * n + j);
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
            }

            // 水平求和
            float sum = vaddvq_f32(sum_vec);

            // 处理剩余元素
            for (; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }

            C[i * n + j] = sum;
        }
    }
}
#endif

/**
 * 矩阵向量乘法（特殊优化）
 *
 * y = A @ x
 * A: [m, k]
 * x: [k]
 * y: [m]
 */
void matvec(
    const float* A,
    const float* x,
    float* y,
    size_t m,
    size_t k
) {
    for (size_t i = 0; i < m; i++) {
        float sum = 0.0f;

#ifdef POCKET_ARM_NEON
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        size_t j = 0;

        for (; j + 3 < k; j += 4) {
            float32x4_t a_vec = vld1q_f32(A + i * k + j);
            float32x4_t x_vec = vld1q_f32(x + j);
            sum_vec = vmlaq_f32(sum_vec, a_vec, x_vec);
        }

        sum = vaddvq_f32(sum_vec);

        for (; j < k; j++) {
            sum += A[i * k + j] * x[j];
        }
#else
        for (size_t j = 0; j < k; j++) {
            sum += A[i * k + j] * x[j];
        }
#endif

        y[i] = sum;
    }
}

/**
 * 多线程矩阵乘法
 */
void matmul_parallel(
    const float* A,
    const float* B,
    float* C,
    size_t m,
    size_t n,
    size_t k,
    size_t num_threads
) {
    if (num_threads <= 1 || m < 4) {
#ifdef POCKET_ARM_NEON
        matmul_neon(A, B, C, m, n, k);
#else
        matmul_scalar(A, B, C, m, n, k);
#endif
        return;
    }

    std::vector<std::thread> threads;
    size_t rows_per_thread = (m + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; t++) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = std::min(start_row + rows_per_thread, m);

        if (start_row >= m) break;

        threads.emplace_back([=]() {
            size_t local_m = end_row - start_row;
#ifdef POCKET_ARM_NEON
            matmul_neon(
                A + start_row * k,
                B,
                C + start_row * n,
                local_m, n, k
            );
#else
            matmul_scalar(
                A + start_row * k,
                B,
                C + start_row * n,
                local_m, n, k
            );
#endif
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

/**
 * 批量矩阵乘法
 *
 * C[b] = A[b] @ B[b] for b in [0, batch_size)
 */
void batch_matmul(
    const float* A,
    const float* B,
    float* C,
    size_t batch_size,
    size_t m,
    size_t n,
    size_t k
) {
    size_t a_stride = m * k;
    size_t b_stride = k * n;
    size_t c_stride = m * n;

    for (size_t b = 0; b < batch_size; b++) {
#ifdef POCKET_ARM_NEON
        matmul_neon(
            A + b * a_stride,
            B + b * b_stride,
            C + b * c_stride,
            m, n, k
        );
#else
        matmul_scalar(
            A + b * a_stride,
            B + b * b_stride,
            C + b * c_stride,
            m, n, k
        );
#endif
    }
}

/**
 * 转置矩阵乘法
 *
 * C = A @ B^T
 * A: [m, k]
 * B: [n, k]  (转置后变为 [k, n])
 * C: [m, n]
 */
void matmul_transposed(
    const float* A,
    const float* B,
    float* C,
    size_t m,
    size_t n,
    size_t k
) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;

#ifdef POCKET_ARM_NEON
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            size_t p = 0;

            for (; p + 3 < k; p += 4) {
                float32x4_t a_vec = vld1q_f32(A + i * k + p);
                float32x4_t b_vec = vld1q_f32(B + j * k + p);
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
            }

            sum = vaddvq_f32(sum_vec);

            for (; p < k; p++) {
                sum += A[i * k + p] * B[j * k + p];
            }
#else
            for (size_t p = 0; p < k; p++) {
                sum += A[i * k + p] * B[j * k + p];
            }
#endif

            C[i * n + j] = sum;
        }
    }
}

} // namespace ops
} // namespace pocket
