"""
快速测试训练流程

使用 smoke 数据集（小规模）快速验证训练代码是否正常工作
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from model import ModelRegistry, TINY_CONFIG
from model.tokenizer import Tokenizer
from data.data_loader import PretrainDataset, SFTDataset

print("=" * 60)
print("PocketLLM 训练流程快速测试")
print("=" * 60)

# 1. 加载分词器
print("\n[1/5] 加载分词器...")
tokenizer = Tokenizer("training/tokenizer")
print(f"✓ 分词器加载成功，词表大小: {tokenizer.get_vocab_size()}")

# 2. 创建模型
print("\n[2/5] 创建模型...")
config = TINY_CONFIG
config.vocab_size = tokenizer.get_vocab_size()
model = ModelRegistry.create("llama_like", config)
print(f"✓ 模型创建成功")
print(f"  参数量: {model.get_num_params() / 1e6:.1f}M")
print(f"  内存占用: {model.get_memory_footprint()['total_mb']:.1f} MB")

# 3. 加载数据集
print("\n[3/5] 加载数据集...")
train_dataset = PretrainDataset(
    "training/data/pretrain_t2t_smoke.jsonl",
    tokenizer,
    max_length=256
)
print(f"✓ 数据集加载成功，样本数: {len(train_dataset)}")

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

# 4. 测试前向传播
print("\n[4/5] 测试前向传播...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

batch = next(iter(train_loader))
input_ids = batch['input_ids'].to(device)
labels = batch['labels'].to(device)

print(f"  Input shape: {input_ids.shape}")
print(f"  Labels shape: {labels.shape}")

with torch.no_grad():
    logits, _ = model(input_ids, use_cache=False)
    print(f"  Output shape: {logits.shape}")

    # 计算损失
    import torch.nn.functional as F
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    print(f"  Loss: {loss.item():.4f}")

print("✓ 前向传播测试通过")

# 5. 测试反向传播
print("\n[5/5] 测试反向传播...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 一步训练
logits, _ = model(input_ids, use_cache=False)
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100
)

loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"  训练步 Loss: {loss.item():.4f}")
print("✓ 反向传播测试通过")

# 总结
print("\n" + "=" * 60)
print("✅ 所有测试通过！训练流程验证成功")
print("=" * 60)
print("\n下一步可以运行完整训练:")
print("  python training/scripts/pretrain.py --config training/configs/pretrain_config.yaml")
print("\n或者先用 smoke 数据集快速验证:")
print("  修改配置文件中的 train_file 为 pretrain_t2t_smoke.jsonl")
