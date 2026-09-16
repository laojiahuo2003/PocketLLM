/**
 * PocketLLM 推理引擎 - 算子接口
 *
 * 所有算子的抽象基类
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace pllm {

// 前向声明
class Tensor;

/**
 * 算子属性
 */
using OpAttributes = std::unordered_map<std::string, std::string>;

/**
 * 算子抽象基类
 */
class Op {
public:
    virtual ~Op() = default;

    /**
     * 前向传播
     * @param inputs 输入张量列表
     * @return 输出张量列表
     */
    virtual std::vector<Tensor> Forward(const std::vector<Tensor>& inputs) = 0;

    /**
     * 获取算子类型名称
     */
    virtual std::string GetOpType() const = 0;

    /**
     * 设置属性
     */
    void SetAttributes(const OpAttributes& attrs) {
        attributes_ = attrs;
    }

    /**
     * 获取属性
     */
    const OpAttributes& GetAttributes() const {
        return attributes_;
    }

protected:
    OpAttributes attributes_;

    /**
     * 辅助方法：从属性中获取整数
     */
    int GetIntAttr(const std::string& key, int default_value = 0) const {
        auto it = attributes_.find(key);
        if (it != attributes_.end()) {
            return std::stoi(it->second);
        }
        return default_value;
    }

    /**
     * 辅助方法：从属性中获取浮点数
     */
    float GetFloatAttr(const std::string& key, float default_value = 0.0f) const {
        auto it = attributes_.find(key);
        if (it != attributes_.end()) {
            return std::stof(it->second);
        }
        return default_value;
    }

    /**
     * 辅助方法：从属性中获取字符串
     */
    std::string GetStringAttr(const std::string& key, const std::string& default_value = "") const {
        auto it = attributes_.find(key);
        if (it != attributes_.end()) {
            return it->second;
        }
        return default_value;
    }
};

using OpPtr = std::shared_ptr<Op>;

/**
 * 算子创建函数指针
 */
using OpCreator = std::function<OpPtr(const OpAttributes&)>;

} // namespace pllm
