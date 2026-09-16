# 分词器使用指南

## PocketLLM 预训练分词器

PocketLLM 自带一个针对中英文优化的预训练分词器，特点：

1. **轻量高效**：词表大小 6400，专为端侧模型设计
2. **压缩率优秀**：中文 ~2.5 chars/token，英文 ~3.5 chars/token
3. **特殊 token 完善**：包含对话、工具调用、思维链等特殊标记
4. **即开即用**：无需额外训练，开箱可用

## 快速开始

### 1. 分词器文件位置

```bash
PocketLLM/training/tokenizer/
├── tokenizer.json          # 分词器主文件
└── tokenizer_config.json   # 配置文件
```

如果目录为空，从项目资源下载：

```bash
# 下载预训练分词器
mkdir -p tokenizer
cd tokenizer
wget https://github.com/YOUR_REPO/PocketLLM/releases/download/v0.1.0/tokenizer.json
wget https://github.com/YOUR_REPO/PocketLLM/releases/download/v0.1.0/tokenizer_config.json
```

### 2. 在代码中使用

```python
from model.tokenizer import Tokenizer

# 加载分词器（相对于 training/ 目录）
tokenizer = Tokenizer("./tokenizer")
# 或使用绝对路径
tokenizer = Tokenizer("/path/to/PocketLLM/training/tokenizer")

# 编码
text = "你好，世界！"
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# 解码
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")

# 批量编码
texts = ["你好", "世界"]
batch = tokenizer.batch_encode(texts, padding=True)
```

### 3. 与模型配合使用

```python
from model import ModelRegistry, SMALL_CONFIG
from model.tokenizer import Tokenizer

# 加载分词器
tokenizer = Tokenizer("./tokenizer")

# 更新模型配置的 vocab_size
config = SMALL_CONFIG
config.vocab_size = tokenizer.get_vocab_size()  # 6400

# 创建模型
model = ModelRegistry.create("llama_like", config)
```

## 分词器特性

### 词表信息

- **词表大小**: 6400
- **类型**: Byte-Level BPE
- **特殊 token**: 36 个

### 特殊 Token

| Token | 用途 | ID |
|-------|------|-----|
| `<\|endoftext\|>` | 文本结束/PAD | 0 |
| `<\|im_start\|>` | 消息开始/BOS | 1 |
| `<\|im_end\|>` | 消息结束/EOS | 2 |
| `<\|object_ref_start\|>` | 对象引用 | 3 |
| `<tool_call>` | 工具调用 | - |
| `<think>` | 思维链 | - |

### 压缩率

- **中文**: ~2.5 字符/token
- **英文**: ~3.5 字符/token
- **混合文本**: ~3.0 字符/token

示例：
```
"人工智能是计算机科学的一个分支..." (200字) → ~80 tokens
"Large language models are..." (200词) → ~60 tokens
```

## 训练数据配置

使用 minimind 分词器训练模型时，数据处理示例：

```python
import torch
from model.tokenizer import Tokenizer

tokenizer = Tokenizer("./tokenizer")

# 预训练数据（纯文本）
def preprocess_pretrain(text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(token_ids, dtype=torch.long)

# SFT 数据（对话格式）
def preprocess_sft(conversations):
    # conversations: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    formatted = ""
    for msg in conversations:
        if msg["role"] == "user":
            formatted += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            formatted += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    return torch.tensor(token_ids, dtype=torch.long)
```

## 高级选项：训练自定义分词器（可选）

如果你需要针对特定领域训练分词器：

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 创建 BPE 分词器
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 训练
trainer = trainers.BpeTrainer(
    vocab_size=6400,
    special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
)

files = ["data.txt"]
tokenizer.train(files, trainer)

# 保存
tokenizer.save("my_tokenizer.json")
```

## 推理引擎集成

分词器也需要在 C++ 推理引擎中实现。可以：

1. **使用 tokenizers-cpp**（推荐）
   ```cpp
   #include <tokenizers_cpp.h>
   auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_blob);
   ```

2. **嵌入 Python 解释器**
   ```cpp
   // 通过 pybind11 调用 Python 分词器
   ```

3. **纯 C++ 实现**（最快但工作量大）
   ```cpp
   // 手动实现 BPE 算法
   ```

我们会在推理引擎开发时详细实现。

## 常见问题

### Q: 为什么用 6400 而不是 32000？

A: 端侧模型追求小巧，6400 的词表：
- Embedding 层参数更少（6400 vs 32000）
- 加载速度更快
- 压缩率依然不错（中文 2.5 chars/token）

### Q: 可以用其他分词器吗？

A: 可以！只需保证：
1. 分词器有 `tokenizer.json` 和 `tokenizer_config.json`
2. 更新 `ModelConfig.vocab_size` 为对应的词表大小
3. 重新训练模型

### Q: 分词器文件很大吗？

A: 很小：
- `tokenizer.json`: ~1.5 MB
- `tokenizer_config.json`: ~5 KB
- 总共不到 2MB，可以直接打包进 APK

## 下一步

分词器准备好后，可以开始：
1. 实现数据加载器（`training/data_loader.py`）
2. 实现预训练脚本（`training/pretrain.py`）
3. 准备训练数据
