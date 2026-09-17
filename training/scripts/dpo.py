"""
PocketLLM DPO (Direct Preference Optimization) 训练脚本

用法:
    python training/scripts/dpo.py --config training/configs/dpo_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging

# SwanLab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("SwanLab not installed. Install with: pip install swanlab")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import ModelRegistry, ModelConfig
from model.tokenizer import Tokenizer
from data.dpo_loader import DPODataset, dpo_collate_fn


# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def load_model(checkpoint_path, device):
    """从检查点加载模型"""
    checkpoint_dir = Path(checkpoint_path)

    # 加载配置
    import json
    with open(checkpoint_dir / "config.json", 'r') as f:
        model_config_dict = json.load(f)

    # 创建模型
    architecture = model_config_dict.pop('architecture', 'llama_like')
    config = ModelConfig(**model_config_dict)
    model = ModelRegistry.create(architecture, config)

    # 加载权重
    state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model: {model.get_num_params() / 1e6:.1f}M parameters")

    return model, config


def create_optimizer(model, config):
    """创建优化器"""
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        eps=config['training']['adam_epsilon'],
        weight_decay=config['training']['weight_decay']
    )
    return optimizer


def create_scheduler(optimizer, config, total_steps):
    """创建学习率调度器"""
    warmup_steps = config['training']['warmup_steps']

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config['training']['learning_rate'] * 0.1
    )

    return scheduler


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                     reference_chosen_logps, reference_rejected_logps,
                     beta=0.1):
    """
    计算 DPO 损失

    Args:
        policy_chosen_logps: 策略模型对 chosen 的对数概率
        policy_rejected_logps: 策略模型对 rejected 的对数概率
        reference_chosen_logps: 参考模型对 chosen 的对数概率
        reference_rejected_logps: 参考模型对 rejected 的对数概率
        beta: DPO 温度参数

    Returns:
        loss, chosen_rewards, rejected_rewards
    """
    # 计算 log ratio
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO 损失
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    # 计算 rewards（用于监控）
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards


def get_batch_logps(logits, labels):
    """
    计算每个序列的平均对数概率

    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]

    Returns:
        logps: [batch_size] - 每个序列的平均对数概率
    """
    # next-token 预测：logits[t] 预测 input[t+1]，labels 需同步 shift 1 位
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()

    # 计算 log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # 收集对应 label 的 log prob
    # labels: [batch_size, seq_len] -> [batch_size, seq_len, 1]
    labels_expanded = labels.unsqueeze(-1)

    # 获取对应位置的 log prob
    per_token_logps = torch.gather(log_probs, dim=-1, index=labels_expanded).squeeze(-1)

    # 只对非 -100 的位置求平均（即只计算 response 部分）
    mask = (labels != -100).float()
    logps = (per_token_logps * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    return logps


def train_epoch(policy_model, reference_model, train_loader, optimizer, scheduler,
                config, epoch, device, swanlab_run=None):
    """训练一个 epoch"""
    policy_model.train()
    reference_model.eval()  # 参考模型始终处于评估模式

    total_loss = 0
    total_chosen_rewards = 0
    total_rejected_rewards = 0
    step = 0

    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    log_steps = config['logging']['log_steps']
    beta = config['dpo']['beta']

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # 获取 chosen 和 rejected 数据
        chosen_input_ids = batch['chosen_input_ids'].to(device)
        chosen_labels = batch['chosen_labels'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        rejected_labels = batch['rejected_labels'].to(device)

        # 策略模型前向传播
        with torch.set_grad_enabled(True):
            # Chosen
            chosen_logits, _ = policy_model(chosen_input_ids, use_cache=False)
            policy_chosen_logps = get_batch_logps(chosen_logits, chosen_labels)

            # Rejected
            rejected_logits, _ = policy_model(rejected_input_ids, use_cache=False)
            policy_rejected_logps = get_batch_logps(rejected_logits, rejected_labels)

        # 参考模型前向传播（不计算梯度）
        with torch.no_grad():
            # Chosen
            ref_chosen_logits, _ = reference_model(chosen_input_ids, use_cache=False)
            reference_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)

            # Rejected
            ref_rejected_logits, _ = reference_model(rejected_input_ids, use_cache=False)
            reference_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)

        # 计算 DPO 损失
        loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            beta=beta
        )

        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps
        total_chosen_rewards += chosen_rewards.mean().item()
        total_rejected_rewards += rejected_rewards.mean().item()

        # 更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)

            # 优化器步进
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1

            # 日志
            if step % log_steps == 0:
                avg_loss = total_loss / (step * gradient_accumulation_steps)
                avg_chosen_reward = total_chosen_rewards / (step * gradient_accumulation_steps)
                avg_rejected_reward = total_rejected_rewards / (step * gradient_accumulation_steps)
                reward_margin = avg_chosen_reward - avg_rejected_reward
                current_lr = optimizer.param_groups[0]['lr']

                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'margin': f'{reward_margin:.4f}',
                    'lr': f'{current_lr:.2e}'
                })

                # SwanLab 记录
                if swanlab_run is not None:
                    swanlab_run.log({
                        'train/loss': avg_loss,
                        'train/chosen_reward': avg_chosen_reward,
                        'train/rejected_reward': avg_rejected_reward,
                        'train/reward_margin': reward_margin,
                        'train/learning_rate': current_lr,
                        'train/epoch': epoch,
                        'train/step': step
                    })

    return {
        'loss': total_loss / (step * gradient_accumulation_steps),
        'chosen_reward': total_chosen_rewards / (step * gradient_accumulation_steps),
        'rejected_reward': total_rejected_rewards / (step * gradient_accumulation_steps)
    }


def save_checkpoint(model, optimizer, scheduler, epoch, step, config, is_best=False):
    """保存检查点"""
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config
    }

    # 保存当前检查点
    checkpoint_path = output_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # 保存最佳模型
    if is_best:
        best_path = output_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")

    # 清理旧检查点
    save_total_limit = config['checkpoint']['save_total_limit']
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    if len(checkpoints) > save_total_limit:
        for old_checkpoint in checkpoints[:-save_total_limit]:
            old_checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {old_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="PocketLLM DPO Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--swanlab-api-key', type=str, default=None, help='SwanLab API key')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # 设置随机种子
    setup_seed(config['hardware']['seed'])

    # 设置设备
    device = torch.device(config['hardware']['device'])
    logger.info(f"Using device: {device}")

    # 初始化 SwanLab
    swanlab_run = None
    use_swanlab = config['logging'].get('use_swanlab', False)

    if use_swanlab and SWANLAB_AVAILABLE:
        api_key = args.swanlab_api_key or config['logging'].get('swanlab_api_key') or os.getenv('SWANLAB_API_KEY')

        if api_key:
            os.environ['SWANLAB_API_KEY'] = api_key

        swanlab_run = swanlab.init(
            project=config['logging'].get('swanlab_project', 'pocketllm'),
            experiment_name=config['logging'].get('swanlab_run_name', 'dpo'),
            config={
                'model': config['model'],
                'training': config['training'],
                'dpo': config['dpo'],
                'data': config['data']
            }
        )
        logger.info("SwanLab initialized")
    elif use_swanlab and not SWANLAB_AVAILABLE:
        logger.warning("SwanLab requested but not installed. Install with: pip install swanlab")

    # 加载分词器
    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer("training/tokenizer")

    # 加载策略模型（待训练）
    logger.info("Loading policy model...")
    policy_model, model_config = load_model(config['model']['checkpoint'], device)
    policy_model = policy_model.to(device)

    # 加载参考模型（固定权重）
    logger.info("Loading reference model...")
    reference_model, _ = load_model(config['dpo']['reference_model'], device)
    reference_model = reference_model.to(device)
    reference_model.eval()

    # 冻结参考模型
    for param in reference_model.parameters():
        param.requires_grad = False

    # 创建数据集
    logger.info("Loading dataset...")
    train_dataset = DPODataset(
        config['data']['train_file'],
        tokenizer,
        max_length=config['data']['max_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=dpo_collate_fn,
        pin_memory=True
    )

    # 创建优化器
    optimizer = create_optimizer(policy_model, config)

    # 创建调度器
    total_steps = len(train_loader) // config['training']['gradient_accumulation_steps'] * config['training']['num_epochs']
    scheduler = create_scheduler(optimizer, config, total_steps)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        policy_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    # 训练循环
    logger.info("Starting DPO training...")
    best_margin = -float('inf')

    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*50}")

        # 训练
        metrics = train_epoch(
            policy_model, reference_model, train_loader, optimizer, scheduler,
            config, epoch + 1, device, swanlab_run
        )

        reward_margin = metrics['chosen_reward'] - metrics['rejected_reward']

        logger.info(f"Epoch {epoch + 1} - Loss: {metrics['loss']:.4f}")
        logger.info(f"Epoch {epoch + 1} - Chosen Reward: {metrics['chosen_reward']:.4f}")
        logger.info(f"Epoch {epoch + 1} - Rejected Reward: {metrics['rejected_reward']:.4f}")
        logger.info(f"Epoch {epoch + 1} - Reward Margin: {reward_margin:.4f}")

        # SwanLab 记录 epoch 指标
        if swanlab_run is not None:
            swanlab_run.log({
                'epoch/loss': metrics['loss'],
                'epoch/chosen_reward': metrics['chosen_reward'],
                'epoch/rejected_reward': metrics['rejected_reward'],
                'epoch/reward_margin': reward_margin,
                'epoch/number': epoch + 1
            })

        # 保存检查点（使用 reward margin 判断最佳模型）
        is_best = reward_margin > best_margin
        if is_best:
            best_margin = reward_margin

        save_checkpoint(
            policy_model, optimizer, scheduler,
            epoch + 1, (epoch + 1) * len(train_loader),
            config, is_best=is_best
        )

    # 保存最终模型
    final_path = Path(config['training']['output_dir']) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(policy_model.state_dict(), final_path / "pytorch_model.bin")

    import json
    with open(final_path / "config.json", 'w') as f:
        json.dump(model_config.to_dict(), f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info("DPO training completed!")
    logger.info(f"Best reward margin: {best_margin:.4f}")
    logger.info(f"Final model saved to {final_path}")
    logger.info(f"{'='*50}")

    # 关闭 SwanLab
    if swanlab_run is not None:
        swanlab_run.finish()
        logger.info("SwanLab run finished")


if __name__ == "__main__":
    main()
