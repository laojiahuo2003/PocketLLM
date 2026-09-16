"""
模型转换工具：PyTorch 训练格式 -> HuggingFace 格式

用法:
    python tools/convert_to_hf.py \
        --input training/checkpoints/pretrain/final \
        --output models/pocket-0.1-pretrain
"""

import argparse
import json
import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model.config import ModelConfig
from training.model.architectures.pocket_hf import PocketConfig, PocketForCausalLM
from training.model.tokenizer import Tokenizer


def convert_weights(pytorch_state_dict):
    """
    转换权重名称：训练格式 -> HuggingFace 格式

    训练格式:
        embed_tokens.weight
        layers.0.self_attn.q_proj.weight
        ...

    HuggingFace 格式:
        model.embed_tokens.weight
        model.layers.0.self_attn.q_proj.weight
        ...
    """
    hf_state_dict = {}

    for key, value in pytorch_state_dict.items():
        # 添加 "model." 前缀（除了 lm_head）
        if not key.startswith('lm_head'):
            new_key = f'model.{key}'
        else:
            new_key = key

        hf_state_dict[new_key] = value

    return hf_state_dict


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to HuggingFace format')
    parser.add_argument('--input', type=str, required=True, help='Input checkpoint directory')
    parser.add_argument('--output', type=str, required=True, help='Output HuggingFace model directory')
    parser.add_argument('--model-name', type=str, default='pocket-0.1', help='Model name')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting model from {input_dir} to {output_dir}")

    # 1. 加载训练配置
    print("Loading config...")
    with open(input_dir / 'config.json', 'r') as f:
        train_config = json.load(f)

    # 2. 创建 HuggingFace 配置
    print("Creating HuggingFace config...")
    hf_config = PocketConfig(
        vocab_size=train_config['vocab_size'],
        hidden_size=train_config['hidden_size'],
        num_hidden_layers=train_config['num_layers'],
        num_attention_heads=train_config['num_attention_heads'],
        num_key_value_heads=train_config['num_key_value_heads'],
        intermediate_size=train_config['intermediate_size'],
        hidden_act=train_config['hidden_act'],
        max_position_embeddings=train_config['max_position_embeddings'],
        initializer_range=train_config.get('initializer_range', 0.02),
        rms_norm_eps=train_config['rms_norm_eps'],
        rope_theta=train_config.get('rope_theta', 10000.0),
        tie_word_embeddings=train_config.get('tie_word_embeddings', False),
    )

    # 保存配置
    hf_config.save_pretrained(output_dir)
    print(f"Saved config to {output_dir / 'config.json'}")

    # 3. 加载权重
    print("Loading weights...")
    pytorch_weights = torch.load(input_dir / 'pytorch_model.bin', map_location='cpu')

    # 4. 转换权重
    print("Converting weights...")
    hf_weights = convert_weights(pytorch_weights)

    # 5. 创建模型并加载权重
    print("Creating HuggingFace model...")
    model = PocketForCausalLM(hf_config)

    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(hf_weights, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")

    # 6. 保存模型
    print("Saving HuggingFace model...")
    model.save_pretrained(output_dir)
    print(f"Saved model to {output_dir / 'pytorch_model.bin'}")

    # 7. 复制 tokenizer
    print("Copying tokenizer...")
    tokenizer_dir = Path('training/tokenizer')
    if tokenizer_dir.exists():
        import shutil
        for file in ['tokenizer.json', 'tokenizer_config.json']:
            src = tokenizer_dir / file
            dst = output_dir / file
            if src.exists():
                shutil.copy(src, dst)
                print(f"Copied {file}")

    # 8. 添加额外的 tokenizer 配置（用于 AutoTokenizer）
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": hf_config.max_position_embeddings,
        "tokenizer_class": "PreTrainedTokenizerFast"
    }

    with open(output_dir / 'tokenizer_config.json', 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    # 9. 添加 README
    readme = f"""---
language: zh
tags:
- pytorch
- causal-lm
- pocket
license: apache-2.0
---

# {args.model_name}

这是一个基于 Llama 架构的小型语言模型，专为移动端部署设计。

## 模型规格

- **参数量**: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M
- **隐藏层维度**: {hf_config.hidden_size}
- **层数**: {hf_config.num_hidden_layers}
- **注意力头数**: {hf_config.num_attention_heads}
- **KV 头数**: {hf_config.num_key_value_heads} (GQA)
- **词表大小**: {hf_config.vocab_size}

## 使用方法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.output}")
tokenizer = AutoTokenizer.from_pretrained("{args.output}")

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
"""

    with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme)

    print(f"\n{'='*50}")
    print("✅ Conversion completed!")
    print(f"{'='*50}")
    print(f"Model saved to: {output_dir}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"\nYou can now use it with:")
    print(f'  from transformers import AutoModelForCausalLM')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_dir}")')


if __name__ == "__main__":
    main()
