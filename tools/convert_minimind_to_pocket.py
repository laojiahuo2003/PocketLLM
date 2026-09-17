#!/usr/bin/env python3
"""
把 MiniMind 的 checkpoint（.pth）转换为 Pocket 的 HuggingFace 模型目录。

MiniMind 与 Pocket 的张量命名完全一致（model.layers.N.self_attn.q_proj.weight 等），
所以转换的关键不在改名，而在把架构差异如实写进 config：

  - MiniMind 带 QK-norm（q_norm / k_norm，per-head RMSNorm）→ PocketConfig.qk_norm=True
  - rope_theta = 1e6（Pocket 默认 1e4）
  - 结构由 state_dict 自动推断，不手写，避免写错

用法:
    python tools/convert_minimind_to_pocket.py \
        --checkpoint /home/uos/code/minimind/checkpoints/pretrain_768.pth \
        --tokenizer  /home/uos/code/minimind/model \
        --output     models/minimind-768-pocket
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.model.architectures.pocket_hf import PocketConfig, PocketForCausalLM


def infer_config(sd, rope_theta: float, max_pos: int) -> PocketConfig:
    """从 state_dict 的真实形状推断结构。"""
    layers = [int(m.group(1)) for k in sd if (m := re.search(r"layers\.(\d+)\.", k))]
    n_layers = max(layers) + 1

    embed = sd["model.embed_tokens.weight"]
    vocab_size, hidden_size = embed.shape

    head_dim = sd["model.layers.0.self_attn.q_norm.weight"].shape[0]
    num_attention_heads = hidden_size // head_dim
    num_key_value_heads = sd["model.layers.0.self_attn.k_proj.weight"].shape[0] // head_dim
    intermediate_size = sd["model.layers.0.mlp.gate_proj.weight"].shape[0]

    print("从 checkpoint 推断的结构：")
    print(f"  vocab={vocab_size} hidden={hidden_size} layers={n_layers}")
    print(f"  heads={num_attention_heads} kv_heads={num_key_value_heads} head_dim={head_dim}")
    print(f"  intermediate={intermediate_size}")

    # 依据是否有 q_norm/k_norm 决定是否启用 QK-norm
    qk_norm = any(k.endswith("self_attn.q_norm.weight") for k in sd)
    print(f"  qk_norm={qk_norm}  rope_theta={rope_theta}")

    return PocketConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        max_position_embeddings=max_pos,
        rms_norm_eps=1e-6,
        rope_theta=rope_theta,
        tie_word_embeddings=False,
        qk_norm=qk_norm,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def main():
    ap = argparse.ArgumentParser(description="MiniMind .pth -> Pocket HuggingFace 目录")
    ap.add_argument("--checkpoint", required=True, help="MiniMind 的 .pth 文件")
    ap.add_argument("--tokenizer", required=True, help="MiniMind 的 tokenizer 目录")
    ap.add_argument("--output", required=True, help="输出的 HF 模型目录")
    ap.add_argument("--rope-theta", type=float, default=1e6)
    ap.add_argument("--max-position-embeddings", type=int, default=2048)
    args = ap.parse_args()

    print(f"加载 checkpoint: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]

    config = infer_config(sd, args.rope_theta, args.max_position_embeddings)

    print("\n构建 Pocket 模型并严格加载权重（strict=True 会校验结构完全一致）...")
    model = PocketForCausalLM(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing or unexpected:
        print(f"  ✗ 结构不匹配：missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
        raise SystemExit("权重与 Pocket 结构不一致，中止")
    print(f"  ✓ 全部 {len(sd)} 个张量精确匹配")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(out)

    print(f"\n✅ 转换完成: {out}")
    for f in sorted(out.iterdir()):
        print(f"   {f.name:24s} {f.stat().st_size / 1024 / 1024:8.2f} MB")


if __name__ == "__main__":
    main()