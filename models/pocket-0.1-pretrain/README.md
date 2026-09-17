---
language: zh
tags:
- pytorch
- causal-lm
- pocket
license: apache-2.0
---

# pocket-0.1

这是一个基于 Llama 架构的小型语言模型，专为移动端部署设计。

## 模型规格

- **参数量**: 5.6M
- **隐藏层维度**: 256
- **层数**: 4
- **注意力头数**: 4
- **KV 头数**: 2 (GQA)
- **词表大小**: 6400

## 使用方法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/pocket-0.1-pretrain")
tokenizer = AutoTokenizer.from_pretrained("models/pocket-0.1-pretrain")

prompt = "你好，"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.8)
print(tokenizer.decode(outputs[0]))
```

## 训练信息

- **框架**: PocketLLM
- **训练阶段**: Pretrain / SFT / DPO
- **数据集**: [数据集信息]

## License

Apache 2.0
