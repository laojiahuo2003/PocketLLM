/**
 * Tensor 实现
 */

#include "pocket.h"
#include <cstring>
#include <stdexcept>

namespace pocket {

size_t Tensor::nbytes() const {
    size_t n = numel();

    switch (qtype) {
        case QuantType::F32:
            return n * sizeof(float);

        case QuantType::F16:
            return n * sizeof(uint16_t);

        case QuantType::Q8_0:
            // 每 32 个元素一个块：1 个 fp16 scale + 32 个 int8
            return ((n + 31) / 32) * (2 + 32 * sizeof(int8_t));

        case QuantType::Q4_0:
            // 每 32 个元素一个块：1 个 fp16 scale + 16 个 uint8 (每个存 2 个 4-bit)
            return ((n + 31) / 32) * (2 + 16);

        default:
            throw std::runtime_error("Unknown quantization type");
    }
}

} // namespace pocket
