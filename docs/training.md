# PocketLLM 训练指南

本文档介绍 PocketLLM 的三阶段训练流程：预训练（Pretrain）、监督微调（SFT）和直接偏好优化（DPO）。

## 训练流程概览

```
原始文本数据
    ↓
[1] 预训练 (Pretrain)
    ↓
预训练模型
    ↓
[2] 监督微调 (SFT)
    ↓
SFT 模型
    ↓
[3] 直接偏好优化 (DPO)
    ↓
最终模型
```

## 1. 预训练 (Pretrain)

预训练阶段在大规模文本语料上训练模型，学习语言的基础知识。

### 数据格式

预训练数据使用 JSONL 格式，每行一个样本：

```jsonl
{"text": "这是第一个文本样本..."}
{"text": "这是第二个文本样本..."}
```

### 配置文件

编辑 `training/configs/pretrain_config.yaml`：

```yaml
# 模型配置
model:
  architecture: "llama_like"
  vocab_size: 6400
  hidden_size: 768        # 模型维度
  num_layers: 12          # 层数
  num_attention_heads: 12
  num_key_value_heads: 4  # GQA

# 数据配置
data:
  train_file: "training/data/pretrain_t2t_mini.jsonl"
  max_length: 512

# 训练配置
training:
  num_epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-4
  bf16: true  # 混合精度训练
```

### 运行训练

```bash
# 激活环境
source .venv/bin/activate

# 开始预训练
python training/scripts/pretrain.py \
  --config training/configs/pretrain_config.yaml

# 从检查点恢复
python training/scripts/pretrain.py \
  --config training/configs/pretrain_config.yaml \
  --resume training/checkpoints/pretrain/checkpoint_epoch1_step1000.pt
```

### 输出文件

训练完成后，模型保存在：

```
training/checkpoints/pretrain/
├── checkpoint_epoch1_step1000.pt
├── checkpoint_epoch2_step2000.pt
├── best_model.pt
└── final/
    ├── pytorch_model.bin  # 模型权重
    └── config.json        # 模型配置
```

## 2. 监督微调 (SFT)

SFT 阶段在指令数据上微调预训练模型，使其能够遵循指令。

### 数据格式

SFT 支持两种数据格式：

**格式 1: 指令格式**

```jsonl
{"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是..."}
{"instruction": "翻译成英文", "input": "你好", "output": "Hello"}
```

**格式 2: 对话格式**

```jsonl
{
  "conversations": [
    {"role": "user", "content": "什么是深度学习？"},
    {"role": "assistant", "content": "深度学习是..."}
  ]
}
```

### 配置文件

编辑 `training/configs/sft_config.yaml`：

```yaml
# 模型配置
model:
  architecture: "llama_like"
  checkpoint: "training/checkpoints/pretrain/final"  # 预训练模型路径

# 数据配置
data:
  train_file: "training/data/sft_t2t_mini.jsonl"
  eval_file: "training/data/sft_t2t_smoke.jsonl"  # 可选
  max_length: 1024  # SFT 通常需要更长的序列

# 训练配置
training:
  num_epochs: 2
  batch_size: 4
  learning_rate: 2.0e-5  # 比预训练小
```

### 运行训练

```bash
# 开始 SFT 训练
python training/scripts/sft.py \
  --config training/configs/sft_config.yaml
```

### 输出文件

```
training/checkpoints/sft/
├── checkpoint_epoch1_step500.pt
├── best_model.pt
└── final/
    ├── pytorch_model.bin
    └── config.json
```

## 3. 直接偏好优化 (DPO)

DPO 阶段使用偏好数据进一步对齐模型，使其生成更符合人类偏好的回答。

### 数据格式

DPO 需要偏好对数据：每个样本包含一个 prompt、一个好的回答（chosen）和一个差的回答（rejected）。

```jsonl
{
  "prompt": "解释什么是 AI",
  "chosen": "人工智能是模拟人类智能的技术...",
  "rejected": "不知道。"
}
```

或对话格式：

```jsonl
{
  "prompt": [
    {"role": "user", "content": "解释什么是 AI"}
  ],
  "chosen": [
    {"role": "assistant", "content": "人工智能是..."}
  ],
  "rejected": [
    {"role": "assistant", "content": "不知道。"}
  ]
}
```

### 配置文件

编辑 `training/configs/dpo_config.yaml`：

```yaml
# 模型配置
model:
  checkpoint: "training/checkpoints/sft/final"  # SFT 模型

# DPO 配置
dpo:
  beta: 0.1  # 温度参数
  reference_model: "training/checkpoints/sft/final"  # 参考模型（通常是 SFT 模型）

# 数据配置
data:
  train_file: "training/data/dpo.jsonl"
  max_length: 1024

# 训练配置
training:
  num_epochs: 1
  batch_size: 2  # DPO 需要成对数据，显存占用更大
  gradient_accumulation_steps: 16
  learning_rate: 5.0e-6  # 更小的学习率
```

### 运行训练

```bash
# 开始 DPO 训练
python training/scripts/dpo.py \
  --config training/configs/dpo_config.yaml
```

### 输出文件

```
training/checkpoints/dpo/
├── checkpoint_epoch1_step500.pt
├── best_model.pt
└── final/
    ├── pytorch_model.bin
    └── config.json
```

## 训练监控

所有训练脚本都集成了 SwanLab 监控，可以实时查看训练曲线。

### 启用 SwanLab

配置文件中已默认启用：

```yaml
logging:
  use_swanlab: true
  swanlab_project: "pocketllm"
  swanlab_run_name: "pretrain"  # 或 "sft" / "dpo"
  swanlab_api_key: "your_api_key"
```

### 查看训练曲线

训练开始后，终端会输出 SwanLab 链接：

```
SwanLab initialized
View run at: https://swanlab.cn/@user/pocketllm/runs/xxx
```

点击链接即可在浏览器查看实时训练指标。

## 模型规格

PocketLLM 支持多种规格，适配不同的设备：

| 规格 | 参数量 | hidden_size | num_layers | 目标设备 |
|------|--------|-------------|------------|----------|
| TINY | ~26M | 512 | 8 | 低端手机 |
| SMALL | ~85M | 768 | 12 | 中端手机 |
| BASE | ~260M | 1024 | 24 | 高端手机 |
| LARGE | ~500M | 1280 | 32 | 旗舰手机 |

修改 `pretrain_config.yaml` 中的 `hidden_size` 和 `num_layers` 来调整模型大小。

## 显存优化

对于 5GB 显存的 RTX 3080：

### 1. 减小 batch size

```yaml
training:
  batch_size: 4  # 或更小
  gradient_accumulation_steps: 8  # 增加累积步数保持有效 batch size
```

### 2. 减小序列长度

```yaml
data:
  max_length: 512  # 预训练
  max_length: 1024  # SFT/DPO
```

### 3. 启用混合精度

```yaml
training:
  bf16: true  # RTX 3080 支持 bf16
```

### 4. 启用梯度检查点（可选）

在模型代码中启用 `gradient_checkpointing`，牺牲速度换显存。

## 训练技巧

### 1. 学习率调度

- **预训练**: 较大的学习率 (5e-4)，使用 warmup
- **SFT**: 较小的学习率 (2e-5)
- **DPO**: 更小的学习率 (5e-6)

### 2. 数据质量

- 预训练：数据量大但质量可以适当宽松
- SFT：数据质量比数量更重要
- DPO：需要高质量的偏好对数据

### 3. 训练时长

- 预训练：通常需要 3-5 个 epoch
- SFT：1-2 个 epoch 即可
- DPO：1 个 epoch 通常足够

### 4. 检查点管理

```yaml
checkpoint:
  save_strategy: "steps"
  save_steps: 1000  # 每 1000 步保存一次
  save_total_limit: 3  # 只保留最近 3 个检查点
```

## 常见问题

### 1. CUDA Out of Memory

**解决方案**：
- 减小 `batch_size`
- 减小 `max_length`
- 增加 `gradient_accumulation_steps`
- 启用 `bf16`

### 2. 训练速度慢

**优化**：
- 增大 `batch_size`（如果显存允许）
- 增大 `num_workers`（数据加载）
- 使用 `pin_memory=True`
- 考虑使用 DeepSpeed

### 3. 损失不下降

**检查**：
- 学习率是否合适（太大或太小）
- 数据是否正确加载
- 模型是否正确初始化
- 梯度是否正常（检查梯度范数）

### 4. 评估损失上升（过拟合）

**解决**：
- 增加 `weight_decay`
- 使用更多训练数据
- 减少训练 epoch
- 添加 dropout（需修改模型代码）

## 完整训练流程示例

```bash
# 1. 预训练
python training/scripts/pretrain.py \
  --config training/configs/pretrain_config.yaml

# 2. SFT（使用预训练模型）
# 先修改 sft_config.yaml 中的 checkpoint 路径
python training/scripts/sft.py \
  --config training/configs/sft_config.yaml

# 3. DPO（使用 SFT 模型）
# 先修改 dpo_config.yaml 中的 checkpoint 路径
python training/scripts/dpo.py \
  --config training/configs/dpo_config.yaml
```

## 下一步

训练完成后，可以：

1. **导出模型** - 转换为 `.pllm` 格式用于推理
2. **评估模型** - 在测试集上评估性能
3. **部署到移动端** - 集成到 Android 应用

详见：
- [推理引擎文档](inference.md)
- [移动端部署文档](mobile.md)
- [SwanLab 监控文档](swanlab.md)
