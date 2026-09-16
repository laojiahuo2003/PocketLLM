# PocketLLM 快速开始指南

## 环境配置

### 1. 使用 uv（推荐，快速）

```bash
# 创建虚拟环境
uv venv

# 激活环境
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

### 2. 使用 pip（传统方式）

```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 依赖说明

**必需依赖**：
- torch >= 2.0.0 (PyTorch)
- transformers >= 4.30.0 (HuggingFace)
- datasets >= 2.12.0 (数据加载)
- tokenizers >= 0.13.0 (分词器)
- accelerate >= 0.20.0 (训练加速)
- numpy, scipy, pyyaml, tqdm (基础库)

**可选依赖**：
- deepspeed (大规模训练，需要时手动安装)
- flash-attn (注意力优化，需要编译，可选)
- wandb (日志可视化，可选)

## 快速测试

### 测试 1: 模型创建

```bash
python tests/test_basic.py
```

**预期输出**：
```
[Tiny Config]
  Parameters: 100.0M
  Memory (FP32): 400.0 MB
  ✓ Model created successfully
...
✅ 所有测试通过！
```

### 测试 2: 训练流程

```bash
python tests/test_training.py
```

**预期输出**：
```
[1/5] 加载分词器...
✓ 分词器加载成功，词表大小: 6400

[2/5] 创建模型...
✓ 模型创建成功
  参数量: 100.0M
  
[3/5] 加载数据集...
✓ 数据集加载成功

[4/5] 测试前向传播...
✓ 前向传播测试通过

[5/5] 测试反向传播...
✓ 反向传播测试通过

✅ 所有测试通过！训练流程验证成功
```

## 开始训练

### 方案 1: Smoke 测试（快速验证，2-3分钟）

```bash
# 修改配置文件使用 smoke 数据集
# 编辑 training/configs/pretrain_config.yaml:
#   train_file: "training/data/pretrain_t2t_smoke.jsonl"
#   num_epochs: 1
#   batch_size: 8

python training/scripts/pretrain.py --config training/configs/pretrain_config.yaml
```

### 方案 2: 完整预训练（数小时）

```bash
# 使用默认配置
python training/scripts/pretrain.py --config training/configs/pretrain_config.yaml
```

**训练输出示例**：
```
Loading tokenizer...
✓ 分词器加载成功，词表大小: 6400

Creating model...
✓ Model created: 300.0M parameters
✓ Memory footprint: 1200.0 MB

Loading dataset...
✓ Loaded 50000 samples

Starting training...
==================================================
Epoch 1/3
==================================================
Epoch 1: 100%|████████| 6250/6250 [12:30<00:00, loss=3.2451, lr=5.0e-4]
Epoch 1 - Average Loss: 3.2451
✓ Saved checkpoint to training/checkpoints/pretrain/checkpoint_epoch1_step6250.pt
```

## 模型配置选择

根据你的硬件选择合适的配置：

| 配置 | 参数量 | 显存占用(bf16) | 推荐 Batch Size | 训练时间 |
|------|--------|----------------|-----------------|----------|
| TINY | 100M | ~0.5GB | 16 | ~2小时 |
| SMALL | 300M | ~1.5GB | 8 | ~6小时 |
| BASE | 600M | ~3GB | 4 | ~12小时 |

**你的 RTX 3080 (5GB 可用)**：
- TINY: ✅ 很轻松
- SMALL: ✅ 推荐（性能和速度平衡）
- BASE: ✅ 可行（使用梯度累积）

## 训练配置说明

### 关键参数

**training/configs/pretrain_config.yaml**:

```yaml
# 模型大小
model:
  hidden_size: 768        # 768=SMALL, 512=TINY, 1024=BASE
  num_layers: 12          # 层数
  num_key_value_heads: 4  # GQA 头数（省显存）

# 训练参数
training:
  batch_size: 8                      # 每个 GPU 的 batch size
  gradient_accumulation_steps: 4    # 有效 batch = 8*4=32
  learning_rate: 5.0e-4             # 预训练学习率
  bf16: true                        # 使用 bf16 混合精度
  
# 数据
data:
  train_file: "training/data/pretrain_t2t_mini.jsonl"  # 1.2GB
  max_length: 512                                       # 序列长度
```

### 显存优化技巧

如果显存不够：

1. **减小 batch_size**
   ```yaml
   batch_size: 4  # 从 8 改为 4
   gradient_accumulation_steps: 8  # 相应增加
   ```

2. **减小序列长度**
   ```yaml
   max_length: 256  # 从 512 改为 256
   ```

3. **使用更小的模型**
   ```yaml
   # 使用 TINY_CONFIG
   hidden_size: 512
   num_layers: 8
   ```

## 常见问题

### Q1: CUDA out of memory

**解决方案**：
- 减小 `batch_size`
- 减小 `max_length`
- 使用更小的模型配置
- 确保没有其他程序占用显存

### Q2: 训练速度慢

**优化方案**：
- 确保使用 `bf16: true`
- 增加 `num_workers: 4` (数据加载并行)
- 使用 `gradient_accumulation_steps` 而不是大 batch size

### Q3: 找不到模块

**确保**：
- 已激活虚拟环境：`source .venv/bin/activate`
- 在项目根目录运行脚本
- 依赖已全部安装

### Q4: tokenizer 加载失败

**检查**：
- `training/tokenizer/tokenizer.json` 文件存在
- 文件大小约 441KB
- 如果缺失，从 Git 重新克隆

## 监控训练

### 1. 查看日志

训练过程会实时输出：
- Loss 变化
- 学习率
- 训练进度

### 2. 检查点

保存位置：`training/checkpoints/pretrain/`

文件：
- `checkpoint_epoch{N}_step{M}.pt` - 定期检查点
- `best_model.pt` - 最佳模型
- `final/` - 最终模型

### 3. 恢复训练

```bash
python training/scripts/pretrain.py \
  --config training/configs/pretrain_config.yaml \
  --resume training/checkpoints/pretrain/checkpoint_epoch1_step1000.pt
```

## 下一步

训练完成后：

1. **导出模型**
   ```python
   from model.export import export_model
   model = ...  # 加载训练好的模型
   export_model(model, "output/model.pllm", quantize="int4")
   ```

2. **SFT 微调**
   ```bash
   python training/scripts/sft.py --config training/configs/sft_config.yaml
   ```

3. **DPO 对齐**
   ```bash
   python training/scripts/dpo.py --config training/configs/dpo_config.yaml
   ```

4. **推理验证**
   - 实现 C++ 推理引擎
   - 部署到 Android

## 获取帮助

如果遇到问题：

1. 检查日志输出
2. 查看 `docs/` 目录下的文档
3. 运行测试脚本验证环境
4. 检查 GPU 驱动和 CUDA 版本

祝训练顺利！🚀
