# .pllm 模型文件格式规范

## 概述

`.pllm` (PocketLLM Model) 是 PocketLLM 推理引擎使用的模型文件格式。设计目标：

- **架构无关**：支持任意模型架构
- **紧凑高效**：二进制格式，支持量化
- **易于解析**：C++ 推理引擎可快速加载
- **可扩展**：支持新算子和新架构

## 文件结构

`.pllm` 是一个目录，包含以下文件：

```
model_name.pllm/
├── model.json          # 计算图定义（JSON）
├── weights.bin         # 权重数据（二进制）
├── config.json         # 模型配置
└── metadata.json       # 元数据
```

## 文件格式详解

### 1. model.json - 计算图定义

描述模型的计算图结构，架构无关。

```json
{
  "architecture": "llama_like",
  "version": "1.0",
  "graph": {
    "nodes": [
      {
        "id": "embed",
        "op": "Embedding",
        "inputs": ["input_ids"],
        "outputs": ["embedded"],
        "attrs": {
          "vocab_size": 32000,
          "hidden_size": 768
        }
      },
      {
        "id": "layer_0_attn_norm",
        "op": "RMSNorm",
        "inputs": ["embedded"],
        "outputs": ["normed_0"],
        "attrs": {
          "hidden_size": 768,
          "eps": 1e-6
        }
      },
      {
        "id": "layer_0_attn",
        "op": "GroupedQueryAttention",
        "inputs": ["normed_0"],
        "outputs": ["attn_out_0", "kv_cache_0"],
        "attrs": {
          "hidden_size": 768,
          "num_heads": 12,
          "num_kv_heads": 4,
          "head_dim": 64,
          "rope_theta": 10000.0
        }
      },
      {
        "id": "layer_0_residual_1",
        "op": "Add",
        "inputs": ["embedded", "attn_out_0"],
        "outputs": ["residual_0_1"]
      },
      {
        "id": "layer_0_ffn_norm",
        "op": "RMSNorm",
        "inputs": ["residual_0_1"],
        "outputs": ["normed_0_ffn"]
      },
      {
        "id": "layer_0_ffn",
        "op": "SwiGLUFFN",
        "inputs": ["normed_0_ffn"],
        "outputs": ["ffn_out_0"],
        "attrs": {
          "hidden_size": 768,
          "intermediate_size": 2048
        }
      },
      {
        "id": "layer_0_residual_2",
        "op": "Add",
        "inputs": ["residual_0_1", "ffn_out_0"],
        "outputs": ["layer_0_out"]
      }
    ]
  }
}
```

#### 节点（Node）定义

每个节点代表一个算子：

- **id**: 唯一标识符
- **op**: 算子类型（如 "Embedding", "RMSNorm", "GroupedQueryAttention"）
- **inputs**: 输入张量名称列表
- **outputs**: 输出张量名称列表
- **attrs**: 算子属性（字典）

#### 支持的算子类型

**基础算子**:
- `Embedding`: 词嵌入
- `Linear`: 线性变换
- `MatMul`: 矩阵乘法
- `Add`: 逐元素相加
- `Multiply`: 逐元素相乘

**归一化**:
- `RMSNorm`: RMS 归一化
- `LayerNorm`: Layer 归一化

**激活函数**:
- `ReLU`, `GELU`, `SiLU`, `Swish`

**注意力**:
- `MultiHeadAttention`: 标准多头注意力
- `GroupedQueryAttention`: GQA
- `LinearAttention`: 线性注意力（未来）

**FFN**:
- `StandardFFN`: 标准 FFN
- `SwiGLUFFN`: SwiGLU FFN
- `GeGLUFFN`: GeGLU FFN

**位置编码**:
- `RoPE`: 旋转位置编码

### 2. weights.bin - 权重数据

二进制格式存储模型权重。

#### 文件结构

```
[Header]
- Magic: "PLLM" (4 bytes)
- Version: uint32 (4 bytes)
- Quantization Type: uint32 (4 bytes)
  - 0: float32
  - 1: int4
  - 2: int8
- Num Tensors: uint32 (4 bytes)

[Tensor 1]
- Name Length: uint32 (4 bytes)
- Name: UTF-8 string
- Num Dims: uint32 (4 bytes)
- Shape: uint32[] (Num Dims * 4 bytes)
- Data Length: uint32 (4 bytes)
- Data: bytes

[Tensor 2]
...
```

#### 量化格式

**INT8 对称量化**:
```
scale = absmax / 127.0
quantized = clamp(round(value / scale), -127, 127)
```

**INT4 对称量化**:
```
scale = absmax / 7.0
quantized = clamp(round(value / scale), -7, 7)
packed = pack_two_int4_to_int8(quantized)
```

### 3. config.json - 模型配置

存储模型的超参数配置。

```json
{
  "vocab_size": 32000,
  "hidden_size": 768,
  "num_layers": 12,
  "num_attention_heads": 12,
  "num_key_value_heads": 4,
  "intermediate_size": 2048,
  "max_position_embeddings": 2048,
  "rope_theta": 10000.0,
  "rms_norm_eps": 1e-6,
  "hidden_act": "silu",
  "tie_word_embeddings": false
}
```

### 4. metadata.json - 元数据

存储模型的元信息。

```json
{
  "architecture": "llama_like",
  "num_params": 300000000,
  "quantization": "int4",
  "created_at": "2026-09-17T12:00:00Z",
  "framework": "PocketLLM",
  "framework_version": "0.1.0"
}
```

## C++ 加载示例

```cpp
#include "pllm/model_loader.h"

// 加载模型
ModelLoader loader;
Model* model = loader.Load("model.pllm");

// 推理
Tensor input_ids = CreateTensor({1, 10}, {123, 456, ...});
Tensor logits = model->Forward(input_ids);
```

## Python 导出示例

```python
from model import ModelRegistry, export_model

# 创建模型
model = ModelRegistry.create("llama_like", config)

# 导出
export_model(model, "output/model.pllm", quantize="int4")
```

## 扩展性

### 添加新算子

1. 在 `model.json` 中定义新的 `op` 类型
2. 在推理引擎中实现对应的 `Op` 类
3. 注册到 `OpRegistry`

### 添加新架构

1. 使用现有算子组合构建新架构
2. 设置 `architecture` 字段为新架构名称
3. 推理引擎自动支持（无需修改）

## 版本兼容性

- **v1.0**: 初始版本，支持基础 Transformer 架构
- **未来版本**: 向后兼容，通过 `version` 字段区分
