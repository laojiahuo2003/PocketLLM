# PocketLLM 使用指南

**Pocket-0.1** - 从训练到部署的完整 LLM 解决方案

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于 GPU 训练)
- CMake 3.15+ (用于 C++ 推理)
- GCC 7+ 或 Clang 5+ (用于 C++ 推理)

### 安装

```bash
# 克隆项目
git clone <repo_url>
cd PocketLLM

# 安装 Python 依赖
pip install -r requirements.txt

# 编译 C++ 推理引擎 (可选)
cd inference/cpp
mkdir build && cd build
cmake ..
make -j4
```

## 📚 完整训练流程

### 第一阶段：预训练 (Pretrain)

```bash
# 1. 准备数据
# 将文本数据放在 data/pretrain/ 目录下，格式为 .txt 或 .jsonl

# 2. 修改配置
vim training/configs/pretrain_config.yaml

# 3. 开始训练
python training/scripts/pretrain.py \
    --config training/configs/pretrain_config.yaml
```

**预训练配置示例：**
```yaml
model:
  name: pocket-0.1
  vocab_size: 32000
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  num_key_value_heads: 4  # GQA
  intermediate_size: 2048
  max_position_embeddings: 2048

training:
  batch_size: 32
  learning_rate: 3e-4
  num_epochs: 10
  warmup_steps: 1000
```

### 第二阶段：监督微调 (SFT)

```bash
# 1. 准备指令数据
# 格式 1: 简单指令对
[
  {
    "instruction": "解释什么是机器学习",
    "output": "机器学习是..."
  }
]

# 格式 2: 多轮对话
[
  {
    "conversations": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
    ]
  }
]

# 2. 开始 SFT
python training/scripts/sft.py \
    --config training/configs/sft_config.yaml
```

### 第三阶段：偏好对齐 (DPO)

```bash
# 1. 准备偏好数据
[
  {
    "prompt": "写一首关于春天的诗",
    "chosen": "春风拂面暖如棉...",
    "rejected": "春天到了很高兴..."
  }
]

# 2. 开始 DPO
python training/scripts/dpo.py \
    --config training/configs/dpo_config.yaml
```

## 🔧 模型转换

### 转换为 HuggingFace 格式

```bash
python tools/convert_to_hf.py \
    --input training/checkpoints/pretrain/final \
    --output models/pocket-0.1 \
    --model-name pocket-0.1
```

### 导出为移动端格式 (.pllm)

```bash
python tools/export_to_pllm.py \
    --input models/pocket-0.1 \
    --output pocket-0.1-fp16.pllm \
    --quant fp16

# 量化版本
python tools/export_to_pllm.py \
    --input models/pocket-0.1 \
    --output pocket-0.1-q8.pllm \
    --quant q8_0
```

## 💬 推理使用

### Python 推理

**命令行交互：**
```bash
python inference/python/chat.py \
    --model models/pocket-0.1 \
    --temperature 0.8 \
    --top-p 0.9
```

**脚本生成：**
```bash
python inference/python/generate.py \
    --model models/pocket-0.1 \
    --prompt "Once upon a time" \
    --max-tokens 100
```

**API 调用：**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("models/pocket-0.1")
tokenizer = AutoTokenizer.from_pretrained("models/pocket-0.1")

# 生成
prompt = "你好，请介绍一下自己"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### C++ 推理

**命令行交互：**
```bash
cd inference/cpp/build
./pocket-chat ../../../pocket-0.1-fp16.pllm
```

**脚本生成：**
```bash
./pocket-generate \
    ../../../pocket-0.1-fp16.pllm \
    --prompt "Hello world" \
    --max-tokens 50 \
    --temperature 0.8
```

**API 集成：**
```cpp
#include "pocket.h"

// 加载模型
pocket::Model model("model.pllm");

// 生成
pocket::Generator gen(model);
std::string prompt = "Hello";
std::string output = gen.generate(prompt, 100);
std::cout << output << std::endl;
```

## 🧪 测试

### 快速验证测试

```bash
# 测试所有组件是否正常
python tests/test_pipeline.py
```

### 端到端测试

```bash
# 测试完整流程（需要训练完成的模型）
python tests/test_e2e.py \
    --checkpoint training/checkpoints/pretrain/final \
    --quant fp16
```

### 性能基准测试

```bash
# Python 推理性能
python tests/benchmark.py \
    --model models/pocket-0.1 \
    --backend python

# C++ 推理性能
python tests/benchmark.py \
    --model pocket-0.1-fp16.pllm \
    --backend cpp
```

## 📊 监控训练

训练过程中，可以通过以下方式监控：

1. **SwanLab 仪表板**
   - 自动启动在 http://localhost:5173
   - 实时查看 loss、学习率等指标

2. **日志文件**
   - 训练日志: `training/logs/pretrain.log`
   - 检查点: `training/checkpoints/pretrain/`

3. **TensorBoard (可选)**
   ```bash
   tensorboard --logdir training/runs
   ```

## 🎯 性能优化建议

### 训练优化

1. **显存不足**
   ```yaml
   training:
     batch_size: 16        # 减小批大小
     gradient_accumulation_steps: 4  # 增加梯度累积
     mixed_precision: true  # 使用混合精度
   ```

2. **加速训练**
   ```bash
   # 使用多 GPU
   torchrun --nproc_per_node=2 training/scripts/pretrain.py
   
   # 使用更大的批大小
   # 修改 config.yaml: batch_size: 64
   ```

### 推理优化

1. **Python 推理**
   ```python
   # 使用半精度
   model = AutoModelForCausalLM.from_pretrained(
       "models/pocket-0.1",
       torch_dtype=torch.float16,
       device_map="auto"
   )
   
   # 批量推理
   outputs = model.generate(
       **inputs,
       batch_size=8,
       use_cache=True
   )
   ```

2. **C++ 推理**
   ```bash
   # 使用量化模型
   ./pocket-chat model-q8.pllm    # Q8 量化
   ./pocket-chat model-q4.pllm    # Q4 量化
   
   # 调整线程数
   export OMP_NUM_THREADS=4
   ```

## 📱 移动端部署

### Android

1. **编译 Android 版本**
   ```bash
   cd inference/cpp
   mkdir build-android && cd build-android
   
   cmake .. \
       -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
       -DANDROID_ABI=arm64-v8a \
       -DANDROID_PLATFORM=android-24
   
   make -j4
   ```

2. **集成到 Android 项目**
   - 将 `libpocket.so` 添加到 `jniLibs/arm64-v8a/`
   - 将 `.pllm` 文件放到 `assets/`
   - 使用 JNI 调用

### iOS

1. **编译 iOS 版本**
   ```bash
   cd inference/cpp
   mkdir build-ios && cd build-ios
   
   cmake .. \
       -DCMAKE_TOOLCHAIN_FILE=../cmake/ios.toolchain.cmake \
       -DPLATFORM=OS64
   
   make -j4
   ```

2. **集成到 iOS 项目**
   - 添加 `libpocket.a` 到项目
   - 将 `.pllm` 文件添加到 Bundle
   - 使用 C++ 或 Objective-C++ 调用

## 🔍 常见问题

### Q1: 训练时显存不足怎么办？

**A:** 有几种方法：
1. 减小 `batch_size` 并增加 `gradient_accumulation_steps`
2. 减小模型规格（减少层数或隐藏层大小）
3. 使用混合精度训练 (`mixed_precision: true`)
4. 使用梯度检查点（以时间换空间）

### Q2: 如何使用自定义词表？

**A:** 训练自己的分词器：
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

# 训练分词器
tokenizer = Tokenizer(BPE())
tokenizer.train(files=["data.txt"], vocab_size=32000)
tokenizer.save("tokenizer.json")
```

### Q3: 如何继续训练（Resume）？

**A:** 在配置文件中指定：
```yaml
training:
  resume_from_checkpoint: training/checkpoints/pretrain/step_1000
```

### Q4: C++ 推理速度慢怎么办？

**A:** 检查以下优化：
1. 使用量化模型（Q8 或 Q4）
2. 编译时启用优化：`cmake -DCMAKE_BUILD_TYPE=Release`
3. 使用 ARM NEON（自动启用）
4. 调整线程数：`export OMP_NUM_THREADS=<cores>`

### Q5: 如何评估模型质量？

**A:** 运行评估脚本：
```bash
# 困惑度评估
python tools/evaluate.py \
    --model models/pocket-0.1 \
    --data data/eval.jsonl \
    --metric perplexity

# 生成质量评估
python tools/evaluate.py \
    --model models/pocket-0.1 \
    --metric generation \
    --prompts data/prompts.txt
```

## 📖 更多文档

- [训练详细指南](docs/training.md)
- [Python 推理文档](docs/inference.md)
- [C++ 推理文档](inference/cpp/README.md)
- [.pllm 格式规范](inference/cpp/format/pllm_format.md)
- [测试计划](docs/testing.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache 2.0

## 🙏 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ 推理引擎设计灵感
- [MiniMind](https://github.com/jingyaogong/minimind) - 训练流程参考
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 模型接口
- [SwanLab](https://swanlab.cn) - 训练监控工具

---

## 🎯 项目状态

- ✅ 训练系统完成
- ✅ Python 推理完成
- 🟡 C++ 推理 90% 完成（缺少完整的 BPE tokenizer）
- ⏳ 移动端部署待开发

**当前版本**: 0.1.0-alpha

**下一步**:
1. 完成预训练
2. 运行端到端测试
3. 完善 BPE 分词器
4. 添加更多量化选项
5. 开发移动端示例应用
