# Pocket-0.1 推理引擎使用指南

本文档介绍如何使用 Pocket-0.1 模型进行推理。

## 📦 推理方案概览

Pocket-0.1 提供两种推理方案：

```
1. Python 推理引擎 (HuggingFace Transformers)
   - 开发阶段快速验证
   - 服务端部署
   - Web 应用

2. C++ 推理引擎 (开发中)
   - 移动端部署 (Android/iOS)
   - 边缘设备
   - 高性能场景
```

## 🚀 快速开始

### 1. 转换模型格式

训练完成后，首先将模型转换为 HuggingFace 格式：

```bash
python tools/convert_to_hf.py \
  --input training/checkpoints/pretrain/final \
  --output models/pocket-0.1-pretrain \
  --model-name pocket-0.1-pretrain
```

转换后的目录结构：

```
models/pocket-0.1-pretrain/
├── config.json              # 模型配置
├── pytorch_model.bin        # 模型权重
├── tokenizer.json           # 分词器
├── tokenizer_config.json    # 分词器配置
└── README.md                # 模型说明
```

### 2. 文本生成

简单的文本生成：

```bash
python inference/python/generate.py \
  --model models/pocket-0.1-pretrain \
  --prompt "从前有座山，" \
  --max-new-tokens 100 \
  --temperature 0.8
```

**参数说明**：
- `--model`: 模型路径
- `--prompt`: 输入文本
- `--max-new-tokens`: 最多生成多少个 token
- `--temperature`: 采样温度 (0.1-2.0，越高越随机)
- `--top-p`: Nucleus 采样 (0.0-1.0)
- `--top-k`: Top-K 采样
- `--num-return-sequences`: 生成多少个候选

### 3. 命令行对话

交互式对话界面：

```bash
python inference/python/chat.py \
  --model models/pocket-0.1-pretrain \
  --temperature 0.8 \
  --history-length 4
```

**使用方法**：
- 直接输入文本进行对话
- 输入 `clear` 清空历史
- 输入 `quit` 或 `exit` 退出

**参数说明**：
- `--history-length`: 保留多少轮对话历史 (0=无历史)
- `--max-length`: 最大序列长度
- `--device`: 运行设备 (cuda/cpu)

## 💻 Python API 使用

### 基础使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "models/pocket-0.1-pretrain",
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用半精度
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "models/pocket-0.1-pretrain",
    trust_remote_code=True
)

# 生成文本
prompt = "你好，我是"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### 批量生成

```python
prompts = [
    "从前有座山，",
    "今天天气",
    "人工智能的未来",
]

# 批量编码
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

# 批量生成
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.8,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
)

# 解码
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Prompt {i+1}: {text}\n")
```

### 流式生成

```python
from transformers import TextIteratorStreamer
from threading import Thread

prompt = "请介绍一下深度学习："
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 创建流式生成器
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

# 生成参数
generation_kwargs = {
    **inputs,
    "max_new_tokens": 200,
    "temperature": 0.8,
    "streamer": streamer,
}

# 在后台线程生成
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 实时打印生成的文本
print("AI: ", end="", flush=True)
for new_text in streamer:
    print(new_text, end="", flush=True)
print()

thread.join()
```

### 对话系统

```python
class ChatBot:
    def __init__(self, model_path, device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.device = device
        self.history = []
    
    def chat(self, user_input, max_history=4):
        # 添加用户输入
        self.history.append(f"User: {user_input}")
        
        # 构建 prompt
        history_to_use = self.history[-(max_history * 2):]
        prompt = "\n".join(history_to_use) + "\nAssistant:"
        
        # 生成
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
        
        # 解码
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated.split("Assistant:")[-1].strip()
        
        # 添加到历史
        self.history.append(f"Assistant: {response}")
        
        return response
    
    def clear_history(self):
        self.history = []


# 使用
bot = ChatBot("models/pocket-0.1-pretrain")

print(bot.chat("你好"))
print(bot.chat("你能做什么？"))
print(bot.chat("介绍一下自己"))

bot.clear_history()
```

## ⚙️ 高级功能

### 1. 混合精度推理

使用 FP16 加速推理并节省显存：

```python
model = AutoModelForCausalLM.from_pretrained(
    "models/pocket-0.1-pretrain",
    torch_dtype=torch.float16,  # 使用 FP16
).cuda()
```

### 2. 量化推理

使用 8-bit 量化节省显存：

```python
# 需要安装 bitsandbytes: pip install bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    "models/pocket-0.1-pretrain",
    load_in_8bit=True,  # 8-bit 量化
    device_map="auto",
)
```

### 3. KV Cache 优化

使用 KV Cache 加速连续生成：

```python
# 第一次生成
inputs = tokenizer("你好", return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)

# 后续生成会自动利用 cache
```

### 4. 采样策略

#### Greedy Search (贪心搜索)

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,  # 关闭采样
)
```

#### Top-K Sampling

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
)
```

#### Top-P (Nucleus) Sampling

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
)
```

#### Beam Search

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=4,
    early_stopping=True,
)
```

### 5. 生成约束

#### 长度约束

```python
outputs = model.generate(
    **inputs,
    min_new_tokens=10,   # 最少生成 10 个 token
    max_new_tokens=100,  # 最多生成 100 个 token
)
```

#### 重复惩罚

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    repetition_penalty=1.2,  # 惩罚重复（>1.0）
)
```

#### 停止词

```python
# 遇到特定词就停止
stop_words = ["结束", "完毕"]
stop_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in stop_words]

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    eos_token_id=stop_ids,
)
```

## 🔧 性能优化

### 1. 批量推理

```python
# 批量处理多个输入
prompts = ["prompt1", "prompt2", "prompt3"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).cuda()
outputs = model.generate(**inputs, max_new_tokens=50)
```

### 2. 编译优化 (PyTorch 2.0+)

```python
# 使用 torch.compile 加速
model = torch.compile(model)
```

### 3. 显存优化

```python
# 1. 使用梯度检查点 (训练时)
model.gradient_checkpointing_enable()

# 2. 清空 CUDA 缓存
import gc
gc.collect()
torch.cuda.empty_cache()

# 3. 使用更小的 batch size
```

## 📊 性能基准

在 RTX 3080 (10GB) 上的推理性能：

| 配置 | 延迟 (token/s) | 显存占用 |
|------|----------------|----------|
| FP32 | ~30 | 3.5 GB |
| FP16 | ~60 | 1.8 GB |
| INT8 | ~80 | 1.0 GB |

## 🐛 常见问题

### 1. CUDA Out of Memory

**解决方案**：
```python
# 使用 FP16
model = model.half()

# 或使用 8-bit 量化
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)

# 减小 batch size 和 max_length
```

### 2. 生成速度慢

**优化**：
```python
# 1. 使用 FP16
model = model.half()

# 2. 减小 max_new_tokens
outputs = model.generate(**inputs, max_new_tokens=50)

# 3. 使用 greedy search 代替 sampling
outputs = model.generate(**inputs, do_sample=False)
```

### 3. 生成质量差

**调优**：
```python
# 1. 调整 temperature (0.6-1.2)
outputs = model.generate(**inputs, temperature=0.8)

# 2. 使用 top-p sampling
outputs = model.generate(**inputs, top_p=0.9, top_k=50)

# 3. 增加 repetition_penalty
outputs = model.generate(**inputs, repetition_penalty=1.2)
```

## 📚 下一步

- [训练指南](training.md) - 训练自己的模型
- [C++ 推理引擎](cpp_inference.md) - 移动端部署
- [Android 集成](android.md) - Android 应用开发
- [API 服务](api_server.md) - 搭建推理服务

## 🔗 相关资源

- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers)
- [Text Generation 指南](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Model Hub](https://huggingface.co/models)
