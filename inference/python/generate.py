"""
Pocket-0.1 文本生成脚本

用法:
    python inference/python/generate.py \
        --model models/pocket-0.1-pretrain \
        --prompt "从前有座山，"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Pocket-0.1 Text Generation')
    parser.add_argument('--model', type=str, required=True, help='Path to HuggingFace model')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--max-new-tokens', type=int, default=100, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--num-return-sequences', type=int, default=1, help='Number of sequences to generate')
    parser.add_argument('--do-sample', action='store_true', default=True, help='Use sampling (vs greedy)')
    args = parser.parse_args()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
    ).to(args.device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 设置特殊 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"{'='*60}\n")

    # 编码输入
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    print(f"Prompt: {args.prompt}")
    print(f"\nGenerating {args.num_return_sequences} sequence(s)...\n")

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码并打印
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"{'='*60}")
        print(f"Sequence {i+1}:")
        print(f"{'='*60}")
        print(generated_text)
        print()


if __name__ == "__main__":
    main()
