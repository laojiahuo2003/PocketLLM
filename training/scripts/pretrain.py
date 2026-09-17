"""
PocketLLM 预训练脚本

用法:
    python training/scripts/pretrain.py --config training/configs/pretrain_config.yaml
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
from data.data_loader import PretrainDataset


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


def create_model(model_config_dict):
    """创建模型"""
    # 提取 architecture（如果有），其余参数传给 ModelConfig
    architecture = model_config_dict.pop('architecture', 'llama_like')

    # 从字典创建 ModelConfig
    config = ModelConfig(**model_config_dict)

    # 创建模型
    model = ModelRegistry.create(architecture, config)

    logger.info(f"Model created: {model.get_num_params() / 1e6:.1f}M parameters")
    logger.info(f"Memory footprint: {model.get_memory_footprint()['total_mb']:.1f} MB")

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

    # 简单的 Cosine 调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=config['training']['learning_rate'] * 0.1
    )

    return scheduler


def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch, device, swanlab_run=None):
    """训练一个 epoch"""
    model.train()
    step = 0

    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    log_steps = config['logging']['log_steps']
    save_steps = config['checkpoint'].get('save_steps') or 0

    # 混合精度上下文（在函数内构造，避免依赖 main 作用域）
    use_amp = config['training']['bf16'] or config['training']['fp16']
    dtype = torch.bfloat16 if config['training']['bf16'] else torch.float16
    autocast = torch.autocast(device_type='cuda', dtype=dtype) if use_amp else contextlib.nullcontext()
    use_scaler = scaler is not None and scaler.is_enabled()

    # loss 累积放在 GPU 上，只在打日志时同步一次
    # （原来的 loss.item() 每个 micro-batch 都强制 device->host 同步，会打断异步流水）
    total_loss = torch.zeros((), device=device)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # 前向传播（bf16/fp16 混合精度）
        with autocast:
            logits, _ = model(input_ids, use_cache=False)

        # 混合精度自检：autocast 失效会让整条链路退化到 fp32（GEMM 慢约 8 倍）
        if batch_idx == 0:
            logger.info(
                f"[AMP 自检] autocast={'on' if use_amp else 'off'} 请求 dtype={dtype} "
                f"-> logits.dtype={logits.dtype}"
            )

        # 计算损失（next-token 预测：logits[i] 预测 input[i+1]，必须 shift labels）
        loss = nn.functional.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100
        )

        # 真实 loss 先记账（detach 后累积，不同步），再缩放后反传
        total_loss += loss.detach()

        if use_scaler:
            scaler.scale(loss / gradient_accumulation_steps).backward()
        else:
            (loss / gradient_accumulation_steps).backward()

        # 更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # 优化器步进
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1

            # 日志
            if step % log_steps == 0:
                avg_loss = (total_loss / (step * gradient_accumulation_steps)).item()
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

            # 周期存盘（原来只在 epoch 结束存一次，长 epoch 中途崩了全丢）
            if save_steps and step % save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, step, config)
                logger.info(f"周期 checkpoint 已保存: epoch={epoch} step={step}")

    return (total_loss / max(step * gradient_accumulation_steps, 1)).item()


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

    # 清理旧检查点（按修改时间排序；按文件名排会把 epoch10 排到 epoch2 前面）
    save_total_limit = config['checkpoint']['save_total_limit']
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
    if len(checkpoints) > save_total_limit:
        for old_checkpoint in checkpoints[:-save_total_limit]:
            old_checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {old_checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="PocketLLM Pretraining")
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
        # 从环境变量或参数获取 API key
        api_key = args.swanlab_api_key or config['logging'].get('swanlab_api_key') or os.getenv('SWANLAB_API_KEY')

        if api_key:
            os.environ['SWANLAB_API_KEY'] = api_key

        swanlab_run = swanlab.init(
            project=config['logging'].get('swanlab_project', 'pocketllm'),
            experiment_name=config['logging'].get('swanlab_run_name', 'pretrain'),
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

    # 更新 vocab_size
    config['model']['vocab_size'] = tokenizer.get_vocab_size()

    # 创建模型
    logger.info("Creating model...")
    model, model_config = create_model(config['model'])
    model = model.to(device)

    # 启用混合精度（模型自动转为 bf16/fp16 计算）
    use_amp = config['training']['bf16'] or config['training']['fp16']
    dtype = torch.bfloat16 if config['training']['bf16'] else torch.float16
    # 旧写法 torch.cuda.amp.GradScaler 已废弃；且 fp16 才需要 scaler，bf16 不需要。
    # 原来建了 scaler 却从不使用，等于 fp16 模式下梯度缩放完全没生效。
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and config['training']['fp16'])

    # 创建数据集
    logger.info("Loading dataset...")
    train_dataset = PretrainDataset(
        config['data']['train_file'],
        tokenizer,
        max_length=config['data']['max_length'],
        pack=config['data'].get('pack', False),
        cache_dir=config['data'].get('cache_dir')
    )

    num_workers = config['data']['num_workers']
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['data'].get('pin_memory', True),
        persistent_workers=num_workers > 0,   # 避免每个 epoch 重新 fork 全部 worker
        prefetch_factor=config['data'].get('prefetch_factor', 4) if num_workers > 0 else None,
        drop_last=True,                       # 丢掉尾部不完整 batch，避免梯度累积组不齐
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
    logger.info("Starting training...")
    best_loss = float('inf')

    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*50}")

        # 训练
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, epoch + 1, device, swanlab_run
        )

        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # SwanLab 记录 epoch 指标
        if swanlab_run is not None:
            swanlab_run.log({
                'epoch/loss': avg_loss,
                'epoch/number': epoch + 1
            })

        # 保存检查点
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        save_checkpoint(
            model, optimizer, scheduler,
            epoch + 1, (epoch + 1) * len(train_loader),
            config, is_best=is_best
        )

    # 保存最终模型
    final_path = Path(config['training']['output_dir']) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_path / "pytorch_model.bin")
    model_config.to_dict()
    import json
    with open(final_path / "config.json", 'w') as f:
        json.dump(model_config.to_dict(), f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Final model saved to {final_path}")
    logger.info(f"{'='*50}")

    # 关闭 SwanLab
    if swanlab_run is not None:
        swanlab_run.finish()
        logger.info("SwanLab run finished")


if __name__ == "__main__":
    main()
