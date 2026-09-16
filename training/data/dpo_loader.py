"""
DPO (Direct Preference Optimization) 数据加载器

DPO 需要偏好对数据：每个样本包含一个 prompt，一个 chosen 回答和一个 rejected 回答
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class DPODataset(Dataset):
    """
    DPO 数据集

    数据格式:
    {
        "prompt": "用户输入",
        "chosen": "更好的回答",
        "rejected": "较差的回答"
    }

    或者对话格式:
    {
        "prompt": [
            {"role": "user", "content": "..."},
            ...
        ],
        "chosen": [
            {"role": "assistant", "content": "..."}
        ],
        "rejected": [
            {"role": "assistant", "content": "..."}
        ]
    }
    """

    def __init__(self, data_path, tokenizer, max_length=1024):
        """
        初始化 DPO 数据集

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

        # 获取 prompt、chosen 和 rejected
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']

        # 格式化文本
        if isinstance(prompt, list):
            # 对话格式
            prompt_text = self._format_conversations(prompt)
            chosen_text = self._format_conversations(chosen)
            rejected_text = self._format_conversations(rejected)
        else:
            # 字符串格式
            prompt_text = prompt
            chosen_text = chosen
            rejected_text = rejected

        # 编码 chosen 序列
        chosen_tokens = self._encode_pair(prompt_text, chosen_text)
        chosen_input_ids, chosen_labels = chosen_tokens

        # 编码 rejected 序列
        rejected_tokens = self._encode_pair(prompt_text, rejected_text)
        rejected_input_ids, rejected_labels = rejected_tokens

        return {
            'chosen_input_ids': torch.tensor(chosen_input_ids, dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_input_ids, dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long)
        }

    def _encode_pair(self, prompt, response):
        """编码 prompt-response 对"""
        # 编码 prompt（不计算损失）
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        # 编码 response（计算损失）
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        # 合并
        tokens = [self.tokenizer.bos_token_id] + prompt_tokens + response_tokens + [self.tokenizer.eos_token_id]

        # 截断
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            # 确保有 EOS
            tokens[-1] = self.tokenizer.eos_token_id

        # Padding
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        # 创建 labels
        # prompt 部分不计算损失（设为 -100）
        # response 部分计算损失
        prompt_length = len(prompt_tokens) + 1  # +1 for BOS

        labels = [-100] * prompt_length + tokens[prompt_length:]
        labels = labels + [-100] * (self.max_length - len(labels))

        # 对 padding 位置也设置为 -100
        labels = [-100 if input_ids[i] == self.tokenizer.pad_token_id else labels[i]
                  for i in range(len(labels))]

        return input_ids, labels

    def _format_conversations(self, conversations):
        """格式化对话数据"""
        if isinstance(conversations, str):
            return conversations

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


def dpo_collate_fn(batch):
    """
    DPO 数据的自定义 collate 函数

    将 chosen 和 rejected 样本堆叠在一起
    """
    chosen_input_ids = torch.stack([item['chosen_input_ids'] for item in batch])
    chosen_labels = torch.stack([item['chosen_labels'] for item in batch])
    rejected_input_ids = torch.stack([item['rejected_input_ids'] for item in batch])
    rejected_labels = torch.stack([item['rejected_labels'] for item in batch])

    return {
        'chosen_input_ids': chosen_input_ids,
        'chosen_labels': chosen_labels,
        'rejected_input_ids': rejected_input_ids,
        'rejected_labels': rejected_labels
    }
