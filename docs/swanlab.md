# SwanLab 训练监控配置

## 什么是 SwanLab

SwanLab 是一个开源的机器学习实验跟踪工具，类似 Weights & Biases (wandb)，但更轻量且支持本地部署。

官网：https://swanlab.cn/

## 功能

- 📊 实时监控训练指标（loss, learning rate 等）
- 📈 可视化训练曲线
- 🔍 对比不同实验
- 💾 自动保存实验配置
- 🌐 Web 界面查看

## 已配置完成

PocketLLM 已经集成 SwanLab，配置如下：

### 1. API Key 已设置

你的 API Key：`cr0yHWFKR2vT7NJyVebZI`

已配置在：
- `training/configs/pretrain_config.yaml`
- `training/configs/sft_config.yaml`
- `training/configs/dpo_config.yaml`

### 2. 自动记录的指标

训练过程中会自动记录：

- `train/loss` - 训练损失
- `train/learning_rate` - 学习率
- `train/epoch` - 当前 epoch
- `train/step` - 训练步数
- `epoch/loss` - 每个 epoch 的平均损失
- `epoch/number` - Epoch 编号

### 3. 实验配置

- **Project**: `pocketllm`
- **Experiment Name**: 
  - 预训练: `pretrain`
  - SFT: `sft`
  - DPO: `dpo`

## 使用方法

### 方法 1: 使用配置文件（推荐）

配置文件中已经设置好，直接运行训练即可：

```bash
# 启动训练（会自动上报到 SwanLab）
python training/scripts/pretrain.py --config training/configs/pretrain_config.yaml
```

### 方法 2: 使用环境变量

```bash
# 设置 API Key
export SWANLAB_API_KEY="cr0yHWFKR2vT7NJyVebZI"

# 运行训练
python training/scripts/pretrain.py --config training/configs/pretrain_config.yaml
```

### 方法 3: 命令行参数

```bash
python training/scripts/pretrain.py \
  --config training/configs/pretrain_config.yaml \
  --swanlab-api-key cr0yHWFKR2vT7NJyVebZI
```

## 查看实验结果

训练开始后，SwanLab 会输出一个链接，类似：

```
SwanLab initialized
View run at: https://swanlab.cn/@your-username/pocketllm/runs/xxx
```

点击链接即可在浏览器中查看实时训练曲线。

或者访问：https://swanlab.cn/ 登录后查看所有实验。

## 配置说明

### 启用/禁用 SwanLab

编辑配置文件：

```yaml
logging:
  use_swanlab: true  # 改为 false 可禁用
  swanlab_project: "pocketllm"
  swanlab_run_name: "pretrain"
  swanlab_api_key: "cr0yHWFKR2vT7NJyVebZI"
```

### 自定义实验名称

```yaml
logging:
  swanlab_run_name: "pretrain-tiny-v1"  # 自定义名称
```

### 安全性

**注意**：API Key 是私密信息，不要提交到 Git！

可以：
1. 使用环境变量：`export SWANLAB_API_KEY="..."`
2. 从配置文件中删除 `swanlab_api_key`，运行时传参
3. 添加到 `.gitignore`（如果单独存储）

## 记录的内容

### 训练过程

每 10 步记录一次（可在配置中修改 `log_steps`）：
- Loss 变化
- 学习率变化
- 训练进度

### 实验配置

自动保存：
- 模型配置（hidden_size, num_layers 等）
- 训练配置（learning_rate, batch_size 等）
- 数据配置（max_length 等）

### 示例可视化

SwanLab 会自动生成：

```
📊 Loss 曲线
   ┌─────────────────────────────┐
10 │╲                            │
 8 │ ╲                           │
 6 │  ╲__                        │
 4 │     ╲___                    │
 2 │         ╲________           │
 0 └─────────────────────────────┘
   0    1000   2000   3000  steps

📈 Learning Rate 调度
   ┌─────────────────────────────┐
   │    ╱╲                       │
   │   ╱  ╲                      │
   │  ╱    ╲                     │
   │ ╱      ╲___                 │
   │╱           ╲________________│
   └─────────────────────────────┘
```

## 对比实验

SwanLab 支持对比不同实验：

1. 运行多个实验（不同配置）
2. 在 SwanLab 网页选择多个实验
3. 查看对比曲线

例如对比：
- TINY vs SMALL vs BASE
- 不同学习率
- 不同 batch size

## 本地部署（可选）

SwanLab 支持本地部署，不需要上传数据到云端：

```bash
# 安装本地版
pip install swanlab[local]

# 启动本地服务
swanlab watch

# 浏览器访问 http://localhost:5092
```

然后在配置中设置：

```yaml
logging:
  swanlab_mode: "local"  # 使用本地模式
```

## 故障排查

### 1. SwanLab 未安装

```bash
pip install swanlab
# 或
uv pip install swanlab
```

### 2. API Key 无效

检查 API Key 是否正确，或在 https://swanlab.cn/ 重新获取。

### 3. 网络问题

如果无法连接 SwanLab 服务器，可以：
- 禁用 SwanLab：`use_swanlab: false`
- 或使用本地模式

### 4. 训练中途不记录

检查：
- `log_steps` 配置是否正确
- 是否有报错信息
- SwanLab 版本是否最新

## 与 WandB 对比

| 特性 | SwanLab | WandB |
|------|---------|-------|
| 开源 | ✅ | ❌ |
| 本地部署 | ✅ | 有限 |
| 中文支持 | ✅ | ❌ |
| 免费额度 | 更多 | 有限 |
| 轻量级 | ✅ | 较重 |

## 示例输出

训练开始时：

```
2026-09-17 10:00:00 - INFO - SwanLab initialized
2026-09-17 10:00:00 - INFO - View run at: https://swanlab.cn/@user/pocketllm/runs/abc123
2026-09-17 10:00:05 - INFO - Starting training...
Epoch 1:   0%|          | 10/158780 [00:02<9:00:00, loss=8.98, lr=5.0e-4]
```

训练结束时：

```
2026-09-17 16:00:00 - INFO - Training completed!
2026-09-17 16:00:00 - INFO - Best loss: 2.3451
2026-09-17 16:00:00 - INFO - SwanLab run finished
```

## 总结

✅ SwanLab 已配置完成  
✅ API Key 已设置  
✅ 自动记录训练指标  
✅ 开箱即用  

直接运行训练即可在 SwanLab 查看实时曲线！

访问：https://swanlab.cn/ 查看你的实验。
