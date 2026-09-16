/**
 * PocketLLM 推理引擎 - 图执行器
 *
 * 负责加载模型和执行推理
 */

#pragma once

#include "graph.h"
#include "tensor.h"
#include "op_registry.h"
#include <string>
#include <unordered_map>
#include <memory>

namespace pllm {

/**
 * 图执行器
 */
class GraphExecutor {
public:
    GraphExecutor() = default;

    /**
     * 从 .pllm 目录加载模型
     * @param model_path .pllm 目录路径
     */
    void LoadModel(const std::string& model_path);

    /**
     * 执行前向推理
     * @param input_ids 输入 token IDs
     * @return 输出 logits
     */
    Tensor Forward(const Tensor& input_ids);

    /**
     * 获取计算图
     */
    const Graph& GetGraph() const { return graph_; }

    /**
     * 获取权重
     */
    const std::unordered_map<std::string, Tensor>& GetWeights() const {
        return weights_;
    }

private:
    Graph graph_;
    std::unordered_map<std::string, Tensor> weights_;  // 权重张量
    std::unordered_map<std::string, Tensor> activations_;  // 中间激活值

    /**
     * 从 JSON 加载计算图
     */
    void LoadGraph(const std::string& json_path);

    /**
     * 从二进制文件加载权重
     */
    void LoadWeights(const std::string& bin_path);

    /**
     * 构建算子实例
     */
    void BuildOps();

    /**
     * 执行单个节点
     */
    void ExecuteNode(GraphNode& node);
};

} // namespace pllm
