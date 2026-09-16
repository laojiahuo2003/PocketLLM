"""
SFT (Supervised Fine-Tuning) 数据加载器

支持对话格式的微调数据
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class SFTDataset(Dataset):
    """
    SFT 数据集

    数据格式:
    {
        "instruction": "用户指令",
        "input": "输入内容（可选）",
        "output": "期望输出"
    }

    或者对话格式:
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }
    """

    def __init__(self, data_path, tokenizer, max_length=1024):
        """
        初始化 SFT 数据集

        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载数据
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 格式化输入文本
        if 'conversations' in sample:
            # 对话格式
            text = self._format_conversations(sample['conversations'])
        else:
            # 指令格式
            text = self._format_instruction(sample)

        # 分词
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # 截断
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]

        # 添加特殊 token
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Padding
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        # 创建 labels（预测下一个 token）
        labels = input_ids.copy()

        # 对 padding 位置设置为 -100（不计算损失）
        labels = [-100 if token_id == self.tokenizer.pad_token_id else token_id
                  for token_id in labels]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def _format_instruction(self, sample):
        """格式化指令数据"""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')

        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        return text

    def _format_conversations(self, conversations):
        """格式化对话数据"""
        formatted_text = ""

        for turn in conversations:
            role = turn.get('role', 'user')
            content = turn.get('content', '')

            if role == 'user':
                formatted_text += f"### User:\n{content}\n\n"
            elif role == 'assistant':
                formatted_text += f"### Assistant:\n{content}\n\n"
            else:
                formatted_text += f"### {role.capitalize()}:\n{content}\n\n"

        return formatted_text.strip()
