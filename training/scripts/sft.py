"""
PocketLLM SFT (Supervised Fine-Tuning) 训练脚本

用法:
    python training/scripts/sft.py --config training/configs/sft_config.yaml
"""

import os
import sys
import argparse
import contextlib
import yaml
import torch
import torch.nn as nn
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
from data.sft_loader import SFTDataset


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


def load_pretrained_model(checkpoint_path, device):
    """从预训练检查点加载模型"""
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

    logger.info(f"Loaded pretrained model from {checkpoint_path}")
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


def train_epoch(model, train_loader, optimizer, scheduler, config, epoch, device, swanlab_run=None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    step = 0

    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    log_steps = config['logging']['log_steps']

    # 混合精度上下文（函数内构造，避免依赖 main 作用域）
    use_amp = config['training']['bf16'] or config['training']['fp16']
    dtype = torch.bfloat16 if config['training']['bf16'] else torch.float16
    autocast = torch.autocast(device_type='cuda', dtype=dtype) if use_amp else contextlib.nullcontext()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播（bf16/fp16 混合精度）
        with autocast:
            logits, _ = model(input_ids, use_cache=False)

        # 计算损失
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps

        # 更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # 优化器步进
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1

            # 日志
            if step % log_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)  # 除以累计 batch 数（原实现误除以 optimizer 步数，放大 4 倍）
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })

                # SwanLab 记录
                if swanlab_run is not None:
                    swanlab_run.log({
                        'train/loss': avg_loss,
                        'train/learning_rate': current_lr,
                        'train/epoch': epoch,
                        'train/step': step
                    })

    return total_loss / len(train_loader)  # 平均到所有 batch


def evaluate(model, eval_loader, device, config):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_steps = 0

    use_amp = config['training']['bf16'] or config['training']['fp16']
    dtype = torch.bfloat16 if config['training']['bf16'] else torch.float16
    autocast = torch.autocast(device_type='cuda', dtype=dtype) if use_amp else contextlib.nullcontext()

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast:
                logits, _ = model(input_ids, use_cache=False)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            total_loss += loss.item()
            total_steps += 1

    return total_loss / total_steps


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
    parser = argparse.ArgumentParser(description="PocketLLM SFT Training")
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
            experiment_name=config['logging'].get('swanlab_run_name', 'sft'),
            config={
                'model': config['model'],
                'training': config['training'],
                'data': config['data']
            }
        )
        logger.info("SwanLab initialized")
    elif use_swanlab and not SWANLAB_AVAILABLE:
        logger.warning("SwanLab requested but not installed. Install with: pip install swanlab")

    # 加载分词器
    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer("training/tokenizer")

    # 加载预训练模型
    logger.info("Loading pretrained model...")
    model, model_config = load_pretrained_model(config['model']['checkpoint'], device)
    model = model.to(device)

    # 启用混合精度
    use_amp = config['training']['bf16'] or config['training']['fp16']
    dtype = torch.bfloat16 if config['training']['bf16'] else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and config['training']['fp16'])
    autocast = torch.autocast(device_type='cuda', dtype=dtype) if use_amp else contextlib.nullcontext()

    # 创建数据集
    logger.info("Loading dataset...")
    train_dataset = SFTDataset(
        config['data']['train_file'],
        tokenizer,
        max_length=config['data']['max_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # 评估数据集（可选）
    eval_loader = None
    if config['data'].get('eval_file'):
        eval_dataset = SFTDataset(
            config['data']['eval_file'],
            tokenizer,
            max_length=config['data']['max_length']
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )

    # 创建优化器
    optimizer = create_optimizer(model, config)

    # 创建调度器
    total_steps = len(train_loader) // config['training']['gradient_accumulation_steps'] * config['training']['num_epochs']
    scheduler = create_scheduler(optimizer, config, total_steps)

    # 恢复训练
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    # 训练循环
    logger.info("Starting SFT training...")
    best_loss = float('inf')

    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*50}")

        # 训练
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch + 1, device, swanlab_run
        )

        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}")

        # 评估
        if eval_loader is not None:
            eval_loss = evaluate(model, eval_loader, device, config)
            logger.info(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")

            # SwanLab 记录评估指标
            if swanlab_run is not None:
                swanlab_run.log({
                    'eval/loss': eval_loss,
                    'epoch/number': epoch + 1
                })

            # 使用评估损失判断最佳模型
            is_best = eval_loss < best_loss
            if is_best:
                best_loss = eval_loss
        else:
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

        # SwanLab 记录 epoch 指标
        if swanlab_run is not None:
            swanlab_run.log({
                'epoch/train_loss': avg_loss,
                'epoch/number': epoch + 1
            })

        # 保存检查点
        save_checkpoint(
            model, optimizer, scheduler,
            epoch + 1, (epoch + 1) * len(train_loader),
            config, is_best=is_best
        )

    # 保存最终模型
    final_path = Path(config['training']['output_dir']) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path / "pytorch_model.bin")

    import json
    with open(final_path / "config.json", 'w') as f:
        json.dump(model_config.to_dict(), f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info("SFT training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Final model saved to {final_path}")
    logger.info(f"{'='*50}")

    # 关闭 SwanLab
    if swanlab_run is not None:
        swanlab_run.finish()
        logger.info("SwanLab run finished")


if __name__ == "__main__":
    main()
