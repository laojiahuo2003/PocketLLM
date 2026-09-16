/**
 * PocketLLM 推理引擎 - 张量定义
 *
 * 轻量级张量实现，支持多种数据类型和量化
 */

#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>

namespace pllm {

/**
 * 数据类型
 */
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT8,
    INT4,
    UINT8
};

/**
 * 张量类
 */
class Tensor {
public:
    /**
     * 构造函数
     * @param shape 张量形状
     * @param dtype 数据类型
     */
    Tensor(const std::vector<int>& shape, DataType dtype = DataType::FLOAT32)
        : shape_(shape), dtype_(dtype) {
        size_t total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }
        num_elements_ = total_size;

        // 分配内存
        size_t bytes = num_elements_ * GetElementSize(dtype);
        data_ = std::shared_ptr<void>(malloc(bytes), free);
    }

    /**
     * 从现有数据构造
     */
    Tensor(const std::vector<int>& shape, void* data, DataType dtype = DataType::FLOAT32)
        : shape_(shape), dtype_(dtype) {
        size_t total_size = 1;
        for (int dim : shape) {
            total_size *= dim;
        }
        num_elements_ = total_size;

        // 共享数据指针（不拥有内存）
        data_ = std::shared_ptr<void>(data, [](void*){});
    }

    /**
     * 获取形状
     */
    const std::vector<int>& GetShape() const { return shape_; }

    /**
     * 获取数据类型
     */
    DataType GetDataType() const { return dtype_; }

    /**
     * 获取元素数量
     */
    size_t GetNumElements() const { return num_elements_; }

    /**
     * 获取数据指针
     */
    template<typename T>
    T* Data() {
        return static_cast<T*>(data_.get());
    }

    template<typename T>
    const T* Data() const {
        return static_cast<const T*>(data_.get());
    }

    /**
     * 获取原始数据指针
     */
    void* RawData() { return data_.get(); }
    const void* RawData() const { return data_.get(); }

    /**
     * 获取维度
     */
    int GetDim(int index) const {
        if (index < 0 || index >= static_cast<int>(shape_.size())) {
            throw std::out_of_range("Dimension index out of range");
        }
        return shape_[index];
    }

    /**
     * 获取秩（维度数）
     */
    int GetRank() const { return static_cast<int>(shape_.size()); }

    /**
     * 重塑形状（不改变数据）
     */
    void Reshape(const std::vector<int>& new_shape) {
        size_t new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != num_elements_) {
            throw std::invalid_argument("New shape must have same number of elements");
        }
        shape_ = new_shape;
    }

    /**
     * 克隆张量（深拷贝）
     */
    Tensor Clone() const {
        Tensor cloned(shape_, dtype_);
        size_t bytes = num_elements_ * GetElementSize(dtype_);
        memcpy(cloned.RawData(), RawData(), bytes);
        return cloned;
    }

private:
    std::vector<int> shape_;
    DataType dtype_;
    size_t num_elements_;
    std::shared_ptr<void> data_;

    /**
     * 获取单个元素的字节数
     */
    static size_t GetElementSize(DataType dtype) {
        switch (dtype) {
            case DataType::FLOAT32: return 4;
            case DataType::FLOAT16: return 2;
            case DataType::INT8: return 1;
            case DataType::INT4: return 1;  // 2个INT4打包成1个字节
            case DataType::UINT8: return 1;
            default: throw std::invalid_argument("Unknown data type");
        }
    }
};

} // namespace pllm
