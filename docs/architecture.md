# PocketLLM 架构设计

## 设计哲学

PocketLLM 采用**模块化、可扩展**的设计，支持多种模型架构和推理后端。

## 模型架构抽象

### 三层架构

```
BaseModel (抽象接口)
    ↓
ConcreteModel (具体实现: LlamaLike, HybridAttention, MoE...)
    ↓
Composable Layers (可组合层: Attention, FFN, Norm...)
```

### 核心组件

#### 1. Attention 层（可插拔）

- **MultiHeadAttention**: 标准多头注意力
- **GroupedQueryAttention**: GQA（省 KV cache）
- **LinearAttention**: 线性复杂度注意力
- **SlidingWindowAttention**: 滑动窗口注意力
- **SparseAttention**: 稀疏注意力

#### 2. FFN 层（可插拔）

- **StandardFFN**: 标准前馈网络
- **SwiGLUFFN**: SwiGLU 激活
- **MoEFFN**: 专家混合

#### 3. 归一化层

- **RMSNorm**: 默认
- **LayerNorm**: 备选

#### 4. 位置编码

- **RoPE**: 旋转位置编码
- **ALiBi**: 线性偏置（未来）

## 推理引擎架构

### 模型文件格式（.pllm）

```
model_name.pllm/
├── model.json          # 计算图定义（JSON 格式）
├── weights.bin         # 权重数据（二进制）
├── tokenizer.json      # 分词器
└── config.json         # 元数据
```

### model.json 示例

```json
{
  "architecture": "llama_like",
  "version": "1.0",
  "graph": {
    "nodes": [
      {"id": "embed", "op": "Embedding", "inputs": ["input_ids"], "attrs": {...}},
      {"id": "layer_0_attn", "op": "GroupedQueryAttention", "inputs": ["embed"], "attrs": {...}},
      {"id": "layer_0_ffn", "op": "SwiGLUFFN", "inputs": ["layer_0_attn"], "attrs": {...}},
      ...
    ]
  }
}
```

### 推理引擎组件

#### OpRegistry（算子注册表）

```cpp
class OpRegistry {
  // 注册算子
  void Register(string op_name, OpCreator creator);
  
  // 创建算子实例
  Op* Create(string op_name, OpAttributes attrs);
};

// 扩展新算子
REGISTER_OP("GroupedQueryAttention", GQAOp);
REGISTER_OP("LinearAttention", LinearAttnOp);
```

#### GraphExecutor（图执行器）

```cpp
class GraphExecutor {
  // 从 model.json 加载
  void LoadModel(string model_path);
  
  // 执行推理
  Tensor Forward(Tensor input);
  
  // 动态图构建
  void BuildGraph(json graph_def);
};
```

#### Backend（多后端支持）

```cpp
// 算子可以有多个后端实现
class MatMulOp {
  virtual Tensor Forward(Tensor a, Tensor b) = 0;
};

class MatMulCPU : public MatMulOp { ... };
class MatMulNEON : public MatMulOp { ... };  // ARM 优化
class MatMulGPU : public MatMulOp { ... };   // 未来支持
```

## 扩展性设计

### 添加新模型架构

1. 继承 `BaseModel`
2. 注册新的 Layer 类型
3. 导出为标准 `.pllm` 格式
4. 推理引擎自动支持（无需修改）

### 添加新算子

1. 实现 `Op` 接口
2. 注册到 `OpRegistry`
3. （可选）提供 NEON/GPU 优化版本

### 添加新训练策略

1. 继承 `BaseTrainer`
2. 实现 `train_step()` 方法
3. 配置文件切换训练器

## 第一版实现

### v1.0: LlamaLike 模型

- **架构**: Llama-style Transformer
- **注意力**: GroupedQueryAttention (GQA)
- **FFN**: SwiGLU
- **归一化**: RMSNorm
- **位置编码**: RoPE
- **规模**: 300M / 600M 参数

### 推理引擎 v1.0

- **后端**: CPU + ARM NEON
- **量化**: INT4 / INT8
- **优化**: KV cache, SIMD 算子

### 未来路线图

- **v2.0**: Hybrid Attention 架构
- **v3.0**: MoE 支持
- **v4.0**: State Space Model
- **推理引擎**: GPU 后端、动态量化

## 目录映射

```
model/
├── base.py                    # BaseModel 抽象类
├── layers/                    # 可组合层
│   ├── attention.py           # 各种 Attention 实现
│   ├── ffn.py                 # 各种 FFN 实现
│   ├── norm.py                # 归一化层
│   └── embedding.py           # Embedding 层
├── architectures/             # 具体架构
│   ├── llama_like.py          # v1.0
│   ├── hybrid_attention.py    # v2.0 (未来)
│   └── moe.py                 # v3.0 (未来)
├── config.py                  # 配置定义
├── export.py                  # 导出为 .pllm
└── registry.py                # 模型注册表

inference/cpp/
├── include/
│   ├── op.h                   # 算子接口
│   ├── op_registry.h          # 算子注册
│   ├── graph.h                # 计算图
│   ├── executor.h             # 执行器
│   └── backend.h              # 后端接口
├── src/
│   ├── ops/                   # 算子库
│   │   ├── matmul.cpp
│   │   ├── attention.cpp      # 各种 Attention 算子
│   │   ├── ffn.cpp
│   │   └── ...
│   ├── backends/              # 多后端
│   │   ├── cpu/
│   │   ├── neon/              # ARM 优化
│   │   └── gpu/               # 未来
│   ├── graph.cpp              # 图构建
│   └── executor.cpp           # 图执行
└── format/
    └── pllm_format.md         # .pllm 格式规范
```

## 关键点

1. **训练侧**：模块化的 Layer，通过组合构建模型
2. **导出格式**：架构无关的 `.pllm` 格式（类似 ONNX 但更简单）
3. **推理引擎**：算子注册 + 动态图执行，支持任意架构
4. **多后端**：算子可以有 CPU/NEON/GPU 多个实现

这样你之后做 Hybrid Attention 或 MoE，只需要：
- 添加新的 Layer 实现
- 注册新的算子
- 导出时自动生成对应的 model.json

推理引擎完全不用改！
