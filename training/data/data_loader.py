"""
PocketLLM 数据加载器

支持：
1. 预训练数据集（纯文本）
2. SFT 数据集（对话格式）
3. DPO 数据集（偏好对）
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Features, Sequence, Value

# 禁用 tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preprocess_conversations(conversations, add_system_ratio=0.2):
    """
    预处理对话数据

    Args:
        conversations: 对话列表
        add_system_ratio: 添加 system prompt 的概率

    Returns:
        处理后的对话列表
    """
    # 如果有工具调用，完整保留
    if any(conv.get('tools') for conv in conversations):
        return conversations

    # System prompts 池
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI助手，尽力为用户提供准确的信息。",
        "你是 PocketLLM，一个小巧但强大的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是 PocketLLM，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are PocketLLM, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer carefully.",
        "You are a knowledgeable AI assistant.",
        "You are PocketLLM, a compact but capable language model."
    ]

    # 概率性添加 system prompt
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations

    return conversations


def postprocess_prompt(prompt_content, empty_think_ratio=0.2):
    """
    后处理生成的 prompt

    Args:
        prompt_content: prompt 内容
        empty_think_ratio: 保留空思考标签的概率

    Returns:
        处理后的 prompt
    """
    # 以一定概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    """
    预训练数据集

    数据格式：
    {"text": "这是一段文本..."}
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Args:
            data_path: JSONL 文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据集
        print(f"Loading pretrain dataset from {data_path}...")
        self.samples = load_dataset('json', data_files=data_path, split='train')
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 分词
        tokens = self.tokenizer.encode(
            str(sample['text']),
            add_special_tokens=False
        )

        # 截断到 max_length - 2（留给 BOS/EOS）
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]

        # 添加 BOS/EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Padding
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Labels（padding 位置设为 -100）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'labels': labels
        }


class SFTDataset(Dataset):
    """
    监督微调数据集

    数据格式：
    {"conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你的？"}
    ]}
    """

    def __init__(self, data_path, tokenizer, max_length=1024):
        """
        Args:
            data_path: JSONL 文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 定义数据格式
        features = Features({
            'conversations': [{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),
                'tools': Value('string'),
                'tool_calls': Value('string')
            }]
        })

        # 加载数据集
        print(f"Loading SFT dataset from {data_path}...")
        self.samples = load_dataset('json', data_files=data_path, split='train', features=features)
        print(f"Loaded {len(self.samples)} samples")

        # 获取 BOS/EOS token IDs（用于生成 labels）
        self.bos_id = tokenizer.encode(
            f'{tokenizer.tokenizer.bos_token}assistant\n',
            add_special_tokens=False
        )
        self.eos_id = tokenizer.encode(
            f'{tokenizer.tokenizer.eos_token}\n',
            add_special_tokens=False
        )

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """使用 tokenizer 的 chat template 生成 prompt"""
        messages = []
        tools = None

        for message in conversations:
            message = dict(message)

            # 处理工具定义
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]

            # 处理工具调用
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)

        # 使用 HuggingFace tokenizer 的 apply_chat_template
        return self.tokenizer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        生成 labels：只对 assistant 的回复计算 loss

        Args:
            input_ids: token IDs

        Returns:
            labels: 只有 assistant 部分是有效 label，其余为 -100
        """
        labels = [-100] * len(input_ids)
        i = 0

        # 查找所有 assistant 回复的位置
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 找到 assistant 开始
                start = i + len(self.bos_id)
                end = start

                # 找到 assistant 结束（EOS）
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # 设置 labels（包括 EOS）
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]

                i = end + len(self.eos_id)
            else:
                i += 1

        return labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # 预处理对话
        conversations = preprocess_conversations(sample['conversations'])

        # 生成 prompt
        prompt = self.create_chat_prompt(conversations)

        # 后处理
        prompt = postprocess_prompt(prompt)

        # 分词
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        # 生成 labels
        labels = self.generate_labels(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        labels = labels + [-100] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class DPODataset(Dataset):
    """
    DPO 数据集

    数据格式：
    {"prompt": "问题", "chosen": "好回答", "rejected": "差回答"}
    """

    def __init__(self, data_path, tokenizer, max_length=1024):
        """
        Args:
            data_path: JSONL 文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据集
        print(f"Loading DPO dataset from {data_path}...")
        self.samples = load_dataset('json', data_files=data_path, split='train')
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 编码 prompt
        prompt_ids = self.tokenizer.encode(sample['prompt'], add_special_tokens=False)

        # 编码 chosen 和 rejected
        chosen_ids = self.tokenizer.encode(sample['chosen'], add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(sample['rejected'], add_special_tokens=False)

        # 拼接并截断
        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids

        if len(chosen_input_ids) > self.max_length:
            chosen_input_ids = chosen_input_ids[:self.max_length]
        if len(rejected_input_ids) > self.max_length:
            rejected_input_ids = rejected_input_ids[:self.max_length]

        # Padding
        chosen_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(chosen_input_ids))
        rejected_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(rejected_input_ids))

        return {
            'chosen_input_ids': torch.tensor(chosen_input_ids, dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_input_ids, dtype=torch.long)
        }


if __name__ == "__main__":
    # 测试数据加载
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from model.tokenizer import Tokenizer

    # 加载分词器
    tokenizer = Tokenizer("../tokenizer")

    # 测试预训练数据集
    print("\n" + "=" * 60)
    print("测试预训练数据集")
    print("=" * 60)
    pretrain_ds = PretrainDataset("../data/pretrain_t2t_smoke.jsonl", tokenizer, max_length=256)
    sample = pretrain_ds[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Sample text: {tokenizer.decode(sample['input_ids'].tolist()[:50])}")

    # 测试 SFT 数据集
    print("\n" + "=" * 60)
    print("测试 SFT 数据集")
    print("=" * 60)
    sft_ds = SFTDataset("../data/sft_t2t_smoke.jsonl", tokenizer, max_length=512)
    sample = sft_ds[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Sample text: {tokenizer.decode(sample['input_ids'].tolist()[:100])}")

    print("\n✅ 数据加载器测试完成！")
