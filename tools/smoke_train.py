#!/usr/bin/env python3
"""
冒烟测试 - 用最小数据快速跑通 Pretrain -> SFT -> DPO 全流程

用法:
    python tools/smoke_train.py            # 跑全部三个阶段
    python tools/smoke_train.py --stage pretrain
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from model import ModelRegistry, ModelConfig
from model.tokenizer import Tokenizer
from data.data_loader import PretrainDataset, SFTDataset, DPODataset


class C:
    G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; B = '\033[94m'; X = '\033[0m'


def log(msg, color=C.B):
    print(f"{color}{msg}{C.X}", flush=True)


def ok(msg):
    print(f"{C.G}  ✓ {msg}{C.X}", flush=True)


def err(msg):
    print(f"{C.R}  ✗ {msg}{C.X}", flush=True)


# ---------------------------------------------------------------- 通用工具

def build_model(ckpt=None, device="cuda"):
    """构建小模型（冒烟测试用缩小版，避免 OOM）"""
    cfg = ModelConfig(
        vocab_size=6400,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    model = ModelRegistry.create("llama_like", cfg)

    if ckpt is not None:
        ckpt_path = Path(ckpt) / "pytorch_model.bin"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            # 兼容裸 state_dict 与 checkpoint 包装
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                log(f"    加载时缺失 {len(missing)} 个权重", C.Y)
            ok(f"从 {ckpt_path} 加载权重")
        else:
            log(f"    未找到 {ckpt_path}，使用随机初始化", C.Y)

    return model.to(device), cfg


def save_final(model, cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    ok(f"保存到 {out_dir}")


def make_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ---------------------------------------------------------------- Stage 1

def smoke_pretrain(device="cuda", steps=30):
    log("\n" + "=" * 60)
    log("阶段 1/3: 预训练 (Pretrain)")
    log("=" * 60)

    tok = Tokenizer(str(PROJECT_ROOT / "training/tokenizer"))
    ok(f"分词器词表大小: {tok.get_vocab_size()}")

    dataset = PretrainDataset(
        str(PROJECT_ROOT / "training/data/pretrain_t2t_smoke.jsonl"),
        tok, max_length=256,
    )
    ok(f"数据集: {len(dataset)} 条样本")

    model, cfg = build_model(device=device)
    ok(f"模型参数: {model.get_num_params() / 1e6:.2f}M")

    loader = make_loader(dataset, batch_size=4)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    model.train()
    losses = []
    t0 = time.time()
    step = 0
    for epoch in range(100):  # 由 steps 控制提前终止
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(ids, use_cache=False)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            step += 1
            if step % 5 == 0:
                print(f"    step {step:3d}  loss={loss.item():.4f}", flush=True)
            if step >= steps:
                break
        if step >= steps:
            break

    dt = time.time() - t0
    ok(f"{steps} 步耗时 {dt:.1f}s ({steps/dt:.2f} step/s)")
    ok(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # 检查是否有梯度
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    if grad_norm > 0:
        ok(f"梯度非零 (sum|grad|={grad_norm:.2f})")
    else:
        err("梯度全为零！反向传播可能有问题")
        raise RuntimeError("梯度为零")

    out = PROJECT_ROOT / "training/checkpoints/smoke/pretrain"
    save_final(model, cfg, out)
    if hasattr(model, "config"):
        pass
    return out


# ---------------------------------------------------------------- Stage 2

def smoke_sft(device="cuda", steps=20):
    log("\n" + "=" * 60)
    log("阶段 2/3: 监督微调 (SFT)")
    log("=" * 60)

    tok = Tokenizer(str(PROJECT_ROOT / "training/tokenizer"))
    dataset = SFTDataset(
        str(PROJECT_ROOT / "training/data/sft_t2t_smoke.jsonl"),
        tok, max_length=256,
    )
    ok(f"数据集: {len(dataset)} 条样本")

    model, cfg = build_model(
        ckpt=PROJECT_ROOT / "training/checkpoints/smoke/pretrain",
        device=device,
    )

    loader = make_loader(dataset, batch_size=2)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    model.train()
    losses = []
    t0 = time.time()
    step = 0
    for epoch in range(100):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(ids, use_cache=False)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            step += 1
            if step % 5 == 0:
                print(f"    step {step:3d}  loss={loss.item():.4f}", flush=True)
            if step >= steps:
                break
        if step >= steps:
            break

    dt = time.time() - t0
    ok(f"{steps} 步耗时 {dt:.1f}s ({steps/dt:.2f} step/s)")
    ok(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    out = PROJECT_ROOT / "training/checkpoints/smoke/sft"
    save_final(model, cfg, out)
    return out


# ---------------------------------------------------------------- Stage 3

def smoke_dpo(device="cuda", steps=15):
    log("\n" + "=" * 60)
    log("阶段 3/3: 偏好对齐 (DPO)")
    log("=" * 60)

    import torch.nn.functional as F

    tok = Tokenizer(str(PROJECT_ROOT / "training/tokenizer"))
    dataset = DPODataset(
        str(PROJECT_ROOT / "training/data/dpo.jsonl"),
        tok, max_length=256,
    )
    ok(f"数据集: {len(dataset)} 条样本")

    sft_ckpt = PROJECT_ROOT / "training/checkpoints/smoke/sft"
    policy, cfg = build_model(ckpt=sft_ckpt, device=device)
    ref, _ = build_model(ckpt=sft_ckpt, device=device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    ok("策略模型 + 参考模型就绪")

    def seq_logprob(model, ids, labels):
        logits, _ = model(ids, use_cache=False)
        logp = F.log_softmax(logits, dim=-1)
        mask = (labels != -100)
        safe_labels = labels.clone()
        safe_labels[~mask] = 0
        gathered = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        return (gathered * mask).sum(dim=-1)

    loader = make_loader(dataset, batch_size=1)
    opt = torch.optim.AdamW(policy.parameters(), lr=5e-6, weight_decay=0.01)
    beta = 0.1

    policy.train()
    margins = []
    t0 = time.time()
    step = 0
    for epoch in range(100):
        for batch in loader:
            c_ids = batch["chosen_input_ids"].to(device)
            c_lab = batch["chosen_labels"].to(device)
            r_ids = batch["rejected_input_ids"].to(device)
            r_lab = batch["rejected_labels"].to(device)

            pi_c = seq_logprob(policy, c_ids, c_lab)
            pi_r = seq_logprob(policy, r_ids, r_lab)
            with torch.no_grad():
                rf_c = seq_logprob(ref, c_ids, c_lab)
                rf_r = seq_logprob(ref, r_ids, r_lab)

            margin = beta * ((pi_c - rf_c) - (pi_r - rf_r))
            loss = -F.logsigmoid(margin).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            margins.append(margin.mean().item())
            step += 1
            if step % 5 == 0:
                print(f"    step {step:3d}  loss={loss.item():.4f}  "
                      f"margin={margin.mean().item():+.4f}", flush=True)
            if step >= steps:
                break
        if step >= steps:
            break

    dt = time.time() - t0
    ok(f"{steps} 步耗时 {dt:.1f}s ({steps/dt:.2f} step/s)")
    ok(f"reward margin: {margins[0]:+.4f} -> {margins[-1]:+.4f}")

    out = PROJECT_ROOT / "training/checkpoints/smoke/dpo"
    save_final(policy, cfg, out)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "pretrain", "sft", "dpo"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    log(f"设备: {args.device}", C.B)
    if args.device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}", C.B)

    results = {}
    stages = ["pretrain", "sft", "dpo"] if args.stage == "all" else [args.stage]

    for stage in stages:
        try:
            if stage == "pretrain":
                results[stage] = smoke_pretrain(args.device)
            elif stage == "sft":
                results[stage] = smoke_sft(args.device)
            elif stage == "dpo":
                results[stage] = smoke_dpo(args.device)
            results[stage] = True
        except Exception as e:
            err(f"{stage} 失败: {e}")
            traceback.print_exc()
            results[stage] = False

    log("\n" + "=" * 60)
    log("冒烟测试总结")
    log("=" * 60)
    for s in stages:
        if results.get(s):
            ok(f"{s}: PASS")
        else:
            err(f"{s}: FAIL")

    return 0 if all(results.get(s) for s in stages) else 1


if __name__ == "__main__":
    sys.exit(main())