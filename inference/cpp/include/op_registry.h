/**
 * PocketLLM 推理引擎 - 算子注册表
 *
 * 用于注册和创建算子实例
 */

#pragma once

#include "op.h"
#include <unordered_map>
#include <memory>
#include <stdexcept>

namespace pllm {

/**
 * 算子注册表（单例）
 */
class OpRegistry {
public:
    /**
     * 获取单例实例
     */
    static OpRegistry& Instance() {
        static OpRegistry instance;
        return instance;
    }

    /**
     * 注册算子
     * @param op_name 算子名称
     * @param creator 创建函数
     */
    void Register(const std::string& op_name, OpCreator creator) {
        registry_[op_name] = creator;
    }

    /**
     * 创建算子实例
     * @param op_name 算子名称
     * @param attrs 算子属性
     * @return 算子实例
     */
    OpPtr Create(const std::string& op_name, const OpAttributes& attrs = {}) {
        auto it = registry_.find(op_name);
        if (it == registry_.end()) {
            throw std::runtime_error("Unknown op type: " + op_name);
        }
        OpPtr op = it->second(attrs);
        op->SetAttributes(attrs);
        return op;
    }

    /**
     * 检查算子是否已注册
     */
    bool IsRegistered(const std::string& op_name) const {
        return registry_.find(op_name) != registry_.end();
    }

    /**
     * 获取所有已注册的算子名称
     */
    std::vector<std::string> GetRegisteredOps() const {
        std::vector<std::string> ops;
        for (const auto& pair : registry_) {
            ops.push_back(pair.first);
        }
        return ops;
    }

private:
    OpRegistry() = default;
    OpRegistry(const OpRegistry&) = delete;
    OpRegistry& operator=(const OpRegistry&) = delete;

    std::unordered_map<std::string, OpCreator> registry_;
};

/**
 * 算子注册辅助宏
 *
 * 用法:
 *   REGISTER_OP("MatMul", MatMulOp);
 */
#define REGISTER_OP(op_name, op_class)                                  \
    namespace {                                                         \
    struct op_class##Registrar {                                        \
        op_class##Registrar() {                                         \
            OpRegistry::Instance().Register(                            \
                op_name,                                                \
                [](const OpAttributes& attrs) -> OpPtr {                \
                    return std::make_shared<op_class>();                \
                }                                                       \
            );                                                          \
        }                                                               \
    };                                                                  \
    static op_class##Registrar g_##op_class##_registrar;               \
    }

} // namespace pllm
