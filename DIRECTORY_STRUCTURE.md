# PocketLLM 目录结构说明

## 顶层目录

```
PocketLLM/
├── training/     ← 训练相关（完整独立模块）
├── inference/    ← 推理引擎（完整独立模块）
├── mobile/       ← 移动端部署
├── tools/        ← 通用工具
├── docs/         ← 文档
├── tests/        ← 测试
└── examples/     ← 示例
```

## 1. training/ - 训练模块

**包含**：模型定义、分词器、数据处理、训练脚本、配置、权重

```
training/
├── model/              # 模型架构
│   ├── base.py         # 抽象基类
│   ├── config.py       # 配置（TINY/SMALL/BASE）
│   ├── registry.py     # 模型注册表
│   ├── export.py       # 导出为 .pllm
│   ├── tokenizer.py    # 分词器封装
│   ├── layers/         # 可组合层
│   │   ├── attention.py    # MHA, GQA
│   │   ├── ffn.py          # Standard, SwiGLU
│   │   └── norm.py         # RMSNorm
│   └── architectures/  # 具体架构
│       └── llama_like.py   # v1.0 实现
│
├── tokenizer/          # 分词器文件
│   ├── tokenizer.json          # 词表 6400
│   └── tokenizer_config.json
│
├── data/               # 数据目录
│   ├── raw/           # 原始语料
│   ├── processed/     # 预处理后
│   └── scripts/       # 数据处理脚本
│
├── scripts/           # 训练脚本
│   ├── pretrain.py    # 预训练
│   ├── sft.py         # 监督微调
│   └── dpo.py         # DPO
│
├── configs/           # 训练配置
│   ├── pretrain_config.yaml
│   ├── sft_config.yaml
│   └── dpo_config.yaml
│
└── checkpoints/       # 模型权重（.gitignore）
```

**使用**：
```python
# 在项目根目录
import sys
sys.path.append('training')

from model import ModelRegistry, SMALL_CONFIG
from model.tokenizer import Tokenizer

# 创建模型
model = ModelRegistry.create("llama_like", SMALL_CONFIG)

# 加载分词器
tokenizer = Tokenizer("training/tokenizer")
```

## 2. inference/ - 推理引擎

**包含**：C++ 推理引擎、Python 绑定、性能测试

```
inference/
├── cpp/                    # C++ 推理引擎
│   ├── CMakeLists.txt      # 构建配置
│   ├── include/            # 头文件
│   │   ├── op.h                # 算子接口
│   │   ├── op_registry.h       # 算子注册表
│   │   ├── tensor.h            # 张量
│   │   ├── graph.h             # 计算图
│   │   └── executor.h          # 执行器
│   ├── src/                # 源文件
│   │   ├── ops/                # 算子实现
│   │   │   ├── matmul.cpp
│   │   │   ├── attention.cpp
│   │   │   └── ...
│   │   └── backends/           # 多后端
│   │       ├── cpu/
│   │       ├── neon/           # ARM 优化
│   │       └── gpu/            # 未来
│   └── format/             # 格式规范
│       └── pllm_format.md
│
├── python/             # Python 绑定
└── benchmarks/         # 性能测试
```

**编译**：
```bash
cd inference/cpp
mkdir build && cd build
cmake ..
make -j4
```

## 3. mobile/ - 移动端部署

**包含**：Android 应用、JNI 接口

```
mobile/
└── android/
    ├── app/
    │   ├── src/
    │   │   └── main/
    │   │       ├── java/       # UI 代码
    │   │       ├── cpp/        # JNI 接口
    │   │       ├── res/        # 资源
    │   │       └── assets/     # 模型文件
    │   └── build.gradle
    ├── build.gradle
    └── settings.gradle
```

**构建 APK**：
```bash
cd mobile/android
./gradlew assembleRelease
```

## 4. tools/ - 通用工具

**包含**：量化、优化、转换、测试工具

```
tools/
├── quantize.py      # INT4/INT8 量化
├── optimize.py      # 模型优化
├── convert.py       # 格式转换
└── benchmark.py     # 性能测试
```

## 5. docs/ - 文档

```
docs/
├── architecture.md              # 架构设计
├── project_structure.md         # 项目结构详解
├── tokenizer.md                 # 分词器使用
└── lessons_from_llama_cpp.md   # llama.cpp 学习笔记
```

## 6. tests/ - 测试

```
tests/
└── test_basic.py    # 基础功能测试
```

运行测试：
```bash
python tests/test_basic.py
```

## 设计原则

### 模块独立性

- **training/** 和 **inference/** 完全独立
- 通过 `.pllm` 格式交换模型
- 可以单独开发和维护

### 清晰的职责

- **training/** 负责：模型设计、训练、导出
- **inference/** 负责：高效推理、多平台支持
- **mobile/** 负责：用户界面、系统集成

### 易于协作

- 不同模块可以由不同人开发
- 目录结构一目了然
- 减少冲突

## 工作流程

### 典型的开发流程

1. **训练模型**
   ```bash
   cd training
   python scripts/pretrain.py --config configs/pretrain_config.yaml
   ```

2. **导出模型**
   ```python
   from model import ModelRegistry
   from model.export import export_model
   
   model = ModelRegistry.create("llama_like", config)
   export_model(model, "../outputs/model.pllm", quantize="int4")
   ```

3. **推理验证**
   ```bash
   cd inference/cpp/build
   ./inference_test ../../outputs/model.pllm
   ```

4. **部署到手机**
   ```bash
   cp outputs/model.pllm mobile/android/app/src/main/assets/
   cd mobile/android
   ./gradlew assembleRelease
   ```

## 总结

这个目录结构的优点：

✅ **清晰**：训练、推理、部署各自独立  
✅ **灵活**：可以单独开发任何模块  
✅ **可维护**：职责明确，易于理解  
✅ **可扩展**：添加新功能不影响现有结构  

现在你可以专注于实现训练流程，而不用担心推理引擎和移动端部署！
