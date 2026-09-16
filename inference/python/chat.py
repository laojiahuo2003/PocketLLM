"""
Pocket-0.1 命令行对话界面

用法:
    python inference/python/chat.py --model models/pocket-0.1-pretrain
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Pocket-0.1 Chat Interface')
    parser.add_argument('--model', type=str, required=True, help='Path to HuggingFace model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--history-length', type=int, default=0, help='Number of history turns to keep (0=no history)')
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
    print(f"🤖 Pocket-0.1 Chat")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"{'='*60}\n")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print()

    # 对话历史
    conversation_history = []

    while True:
        try:
            # 获取用户输入
            user_input = input("👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if user_input.lower() == 'clear':
                conversation_history = []
                print("✅ 对话历史已清空")
                continue

            # 添加到历史
            conversation_history.append(f"User: {user_input}")

            # 保留最近的历史
            if args.history_length > 0:
                history_to_use = conversation_history[-(args.history_length * 2):]
            else:
                history_to_use = [conversation_history[-1]]  # 只用当前输入

            # 构建 prompt
            prompt = "\n".join(history_to_use) + "\nAssistant:"

            # 编码
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}

            # 生成
            print("🤖 Assistant: ", end='', flush=True)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 解码
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取 Assistant 的回复
            if "Assistant:" in generated_text:
                assistant_response = generated_text.split("Assistant:")[-1].strip()
            else:
                assistant_response = generated_text[len(prompt):].strip()

            # 去除可能的续写
            if "User:" in assistant_response:
                assistant_response = assistant_response.split("User:")[0].strip()

            print(assistant_response)
            print()

            # 添加回复到历史
            conversation_history.append(f"Assistant: {assistant_response}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print()


if __name__ == "__main__":
    main()
