/**
 * PocketLLM 推理引擎 - 计算图
 *
 * 表示模型的计算图结构
 */

#pragma once

#include "op.h"
#include "tensor.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace pllm {

/**
 * 图节点
 */
struct GraphNode {
    std::string id;                     // 节点唯一标识
    std::string op_type;                // 算子类型
    std::vector<std::string> inputs;    // 输入张量名称
    std::vector<std::string> outputs;   // 输出张量名称
    OpAttributes attrs;                 // 算子属性
    OpPtr op;                           // 算子实例（延迟创建）
};

/**
 * 计算图
 */
class Graph {
public:
    Graph() = default;

    /**
     * 添加节点
     */
    void AddNode(const GraphNode& node) {
        nodes_.push_back(node);
        node_map_[node.id] = nodes_.size() - 1;
    }

    /**
     * 获取节点数量
     */
    size_t GetNumNodes() const { return nodes_.size(); }

    /**
     * 获取节点（按索引）
     */
    GraphNode& GetNode(size_t index) {
        if (index >= nodes_.size()) {
            throw std::out_of_range("Node index out of range");
        }
        return nodes_[index];
    }

    const GraphNode& GetNode(size_t index) const {
        if (index >= nodes_.size()) {
            throw std::out_of_range("Node index out of range");
        }
        return nodes_[index];
    }

    /**
     * 获取节点（按ID）
     */
    GraphNode& GetNodeById(const std::string& id) {
        auto it = node_map_.find(id);
        if (it == node_map_.end()) {
            throw std::runtime_error("Node not found: " + id);
        }
        return nodes_[it->second];
    }

    /**
     * 获取所有节点
     */
    const std::vector<GraphNode>& GetNodes() const { return nodes_; }
    std::vector<GraphNode>& GetNodes() { return nodes_; }

    /**
     * 设置架构名称
     */
    void SetArchitecture(const std::string& arch) { architecture_ = arch; }
    const std::string& GetArchitecture() const { return architecture_; }

    /**
     * 设置版本
     */
    void SetVersion(const std::string& ver) { version_ = ver; }
    const std::string& GetVersion() const { return version_; }

private:
    std::string architecture_;
    std::string version_;
    std::vector<GraphNode> nodes_;
    std::unordered_map<std::string, size_t> node_map_;  // id -> index
};

} // namespace pllm
