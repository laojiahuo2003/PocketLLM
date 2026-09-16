# PocketLLM 项目结构总览

## 目录结构

```
PocketLLM/
├── README.md                   # 项目说明
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略文件
│
├── docs/                       # 📚 文档
│   └── architecture.md         # 架构设计文档
│
├── model/                      # 🧠 模型定义（PyTorch）
│   ├── __init__.py
│   ├── base.py                 # 模型基类
│   ├── config.py               # 模型配置（含预定义配置）
│   ├── registry.py             # 模型注册表
│   ├── export.py               # 导出为 .pllm 格式
│   ├── layers/                 # 可组合层
│   │   ├── attention.py        # 注意力层（MHA, GQA...）
│   │   ├── ffn.py              # FFN 层（Standard, SwiGLU...）
│   │   └── norm.py             # 归一化层（RMSNorm）
│   └── architectures/          # 具体架构实现
│       ├── llama_like.py       # v1.0: Llama-style 架构
│       ├── hybrid_attention.py # v2.0: 混合注意力（未来）
│       └── moe.py              # v3.0: MoE（未来）
│
├── training/                   # 🎯 训练代码
│   ├── pretrain.py             # 预训练脚本
│   ├── sft.py                  # 监督微调脚本
│   ├── dpo.py                  # DPO 训练脚本
│   ├── data_loader.py          # 数据加载器
│   ├── trainer.py              # 训练器封装
│   └── configs/                # 训练配置文件
│       ├── pretrain_config.yaml
│       ├── sft_config.yaml
│       └── dpo_config.yaml
│
├── data/                       # 💾 数据目录
│   ├── raw/                    # 原始语料
│   ├── processed/              # 预处理后的数据
│   ├── tokenized/              # 分词后的数据
│   └── scripts/                # 数据处理脚本
│       ├── collect.py          # 数据收集
│       ├── clean.py            # 数据清洗
│       └── preprocess.py       # 数据预处理
│
├── checkpoints/                # 💾 模型权重（.gitignore）
│
├── inference/                  # ⚡ 推理引擎
│   ├── cpp/                    # C++ 推理引擎
│   │   ├── CMakeLists.txt      # CMake 配置
│   │   ├── include/            # 头文件
│   │   │   ├── op.h            # 算子接口
│   │   │   ├── op_registry.h   # 算子注册表
│   │   │   ├── tensor.h        # 张量定义
│   │   │   ├── graph.h         # 计算图
│   │   │   └── executor.h      # 图执行器
│   │   ├── src/                # 源文件
│   │   │   ├── ops/            # 算子实现
│   │   │   │   ├── matmul.cpp
│   │   │   │   ├── attention.cpp
│   │   │   │   └── ...
│   │   │   └── backends/       # 多后端实现
│   │   │       ├── cpu/        # CPU 实现
│   │   │       ├── neon/       # ARM NEON 优化
│   │   │       └── gpu/        # GPU（未来）
│   │   ├── format/             # 格式规范
│   │   │   └── pllm_format.md  # .pllm 格式文档
│   │   └── tests/              # 单元测试
│   │
│   ├── python/                 # Python 绑定（用于验证）
│   └── benchmarks/             # 性能测试
│
├── android/                    # 📱 Android 应用
│   ├── app/                    # Android 项目
│   │   ├── build.gradle
│   │   └── src/
│   │       └── main/
│   │           ├── java/       # Java/Kotlin 代码
│   │           ├── cpp/        # JNI 接口
│   │           ├── res/        # 资源文件
│   │           └── assets/     # 模型文件
│   ├── build.gradle            # 项目配置
│   └── settings.gradle
│
├── tools/                      # 🔧 工具脚本
│   ├── quantize.py             # 量化工具
│   ├── optimize.py             # 模型优化
│   ├── convert.py              # 格式转换
│   └── benchmark.py            # 性能测试
│
├── tests/                      # 🧪 测试
│   ├── test_model.py
│   ├── test_training.py
│   └── test_inference.py
│
└── examples/                   # 📖 示例代码
    ├── train_demo.py           # 训练示例
    └── inference_demo.py       # 推理示例
```

## 关键设计特性

### 1. 模块化架构

- **模型定义**：基于抽象基类，通过注册表管理多种架构
- **可组合层**：注意力、FFN、归一化层可独立替换
- **推理引擎**：算子注册 + 动态图执行，架构无关

### 2. 扩展性

**添加新模型架构**（3 步）：
```python
# 1. 定义新架构
@ModelRegistry.register("new_arch")
class NewArchModel(BaseModel):
    ...

# 2. 导出模型
model.export_to_pllm("model.pllm", quantize="int4")

# 3. 推理引擎自动支持（无需修改 C++ 代码）
```

**添加新算子**（3 步）：
```cpp
// 1. 实现算子
class NewOp : public Op { ... };

// 2. 注册
REGISTER_OP("NewOp", NewOp);

// 3. 在 Python 导出时使用
```

### 3. 完整的训练到部署流程

```
语料收集 → 预处理 → 预训练 → SFT → DPO
    ↓
导出 .pllm → 量化 → C++ 推理引擎 → Android APK
```

## 当前进度

### ✅ 已完成

- [x] 项目结构设计
- [x] 模型架构抽象（`model/base.py`）
- [x] 模型注册表（`model/registry.py`）
- [x] 模型配置（`model/config.py`）
- [x] 可组合层（Attention, FFN, Norm）
- [x] Llama-like 架构实现
- [x] 模型导出工具（`.pllm` 格式）
- [x] 推理引擎核心头文件
- [x] `.pllm` 格式规范文档
- [x] CMake 配置

### 🚧 待实现

#### 模型侧
- [ ] 分词器实现
- [ ] 数据加载器
- [ ] 训练脚本（预训练、SFT、DPO）
- [ ] 训练配置文件
- [ ] 数据处理脚本

#### 推理引擎
- [ ] 算子实现（MatMul, Attention, FFN...）
- [ ] 权重加载器
- [ ] 图执行器实现
- [ ] CPU 后端实现
- [ ] ARM NEON 优化
- [ ] INT4/INT8 量化支持

#### Android
- [ ] JNI 接口
- [ ] Android 应用框架
- [ ] UI 设计
- [ ] APK 打包脚本

#### 工具和测试
- [ ] 量化工具
- [ ] 性能测试
- [ ] 单元测试

## 下一步行动

建议按以下顺序推进：

1. **先跑通训练流程**
   - 实现分词器
   - 实现数据加载器
   - 实现预训练脚本
   - 在小数据集上验证

2. **再实现推理引擎**
   - 实现核心算子
   - 实现图执行器
   - 用 Python 导出的模型验证

3. **最后对接 Android**
   - 实现 JNI 接口
   - 开发 Android 应用
   - 打包测试

## 技术栈总结

| 模块 | 技术栈 |
|------|--------|
| 模型训练 | PyTorch, DeepSpeed, Transformers |
| 推理引擎 | C++17, CMake, ARM NEON |
| Android | Java/Kotlin, NDK, JNI |
| 量化 | INT4/INT8 对称量化 |
| 数据处理 | Python, NumPy, Datasets |

## 参考资源

- **Llama 2**: 架构参考
- **MQA/GQA**: 注意力优化
- **NCNN/MNN**: 移动端推理引擎参考
- **ONNX**: 模型格式参考
