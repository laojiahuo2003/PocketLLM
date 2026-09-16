# PocketLLM 项目结构总览

## 目录结构

```
PocketLLM/
├── training/                   # 🎓 训练相关（完整独立）
│   ├── model/                  # 模型定义
│   │   ├── __init__.py
│   │   ├── base.py             # 模型基类
│   │   ├── config.py           # 模型配置
│   │   ├── registry.py         # 模型注册表
│   │   ├── export.py           # 导出工具
│   │   ├── tokenizer.py        # 分词器封装
│   │   ├── layers/             # 可组合层
│   │   │   ├── attention.py
│   │   │   ├── ffn.py
│   │   │   └── norm.py
│   │   └── architectures/      # 具体架构
│   │       ├── llama_like.py   # v1.0
│   │       └── ...             # 未来架构
│   │
│   ├── tokenizer/              # 分词器文件
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   │
│   ├── data/                   # 数据处理
│   │   ├── raw/                # 原始语料
│   │   ├── processed/          # 预处理后
│   │   ├── tokenized/          # 分词后
│   │   └── scripts/            # 数据处理脚本
│   │
│   ├── scripts/                # 训练脚本
│   │   ├── pretrain.py         # 预训练
│   │   ├── sft.py              # 监督微调
│   │   └── dpo.py              # DPO
│   │
│   ├── configs/                # 训练配置
│   │   ├── pretrain_config.yaml
│   │   ├── sft_config.yaml
│   │   └── dpo_config.yaml
│   │
│   └── checkpoints/            # 模型权重
│
├── inference/                  # ⚡ 推理引擎（完整独立）
│   ├── cpp/                    # C++ 推理引擎
│   │   ├── CMakeLists.txt
│   │   ├── include/            # 头文件
│   │   │   ├── op.h
│   │   │   ├── op_registry.h
│   │   │   ├── tensor.h
│   │   │   ├── graph.h
│   │   │   └── executor.h
│   │   ├── src/                # 源文件
│   │   │   ├── ops/            # 算子实现
│   │   │   └── backends/       # 多后端
│   │   │       ├── cpu/
│   │   │       ├── neon/
│   │   │       └── gpu/
│   │   └── format/             # 格式规范
│   │       └── pllm_format.md
│   │
│   ├── python/                 # Python 绑定
│   └── benchmarks/             # 性能测试
│
├── mobile/                     # 📱 移动端部署
│   └── android/                # Android 应用
│       ├── app/
│       ├── build.gradle
│       └── settings.gradle
│
├── docs/                       # 📚 文档
│   ├── architecture.md
│   ├── project_structure.md
│   ├── tokenizer.md
│   └── lessons_from_llama_cpp.md
│
├── tools/                      # 🔧 通用工具
│   ├── quantize.py
│   ├── optimize.py
│   └── benchmark.py
│
├── tests/                      # 🧪 测试
│   └── test_basic.py
│
├── examples/                   # 📖 示例
│
├── README.md
├── requirements.txt
└── .gitignore
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
