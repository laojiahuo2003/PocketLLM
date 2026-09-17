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
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
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
    预训练数据集（预 tokenize + 磁盘缓存 + 可选 packing）

    数据格式：
        {"text": "这是一段文本..."}

    切分模式（config: data.pack）：
        pack=False  每个样本独立，不足 max_length 用 pad 补齐（与 MiniMind 行为一致）
        pack=True   所有文档用 EOS 拼接后按 max_length 切块，无 pad 浪费（推荐）

    关键改动：首次运行把全量 tokenize 结果缓存成 .npy，之后 mmap 读取。
    原实现在每个 epoch 对每条样本重新 tokenize（8M 样本 × 2 epoch = 1600+ 万次），
    且每次都走 HF datasets 的随机访问，是纯重复劳动。

    注意：pack=True 时文档间会互相 attend（GPT/Llama 的标准做法），
    与 MiniMind 的「每样本独立 padding」不是同一数据分布，做对照实验时要留意。
    """

    def __init__(self, data_path, tokenizer, max_length=512, pack=False, cache_dir=None):
        """
        Args:
            data_path: JSONL 文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            pack: 是否把所有文档拼接切块（消除 padding 浪费）
            cache_dir: tokenize 缓存目录，默认 data_path 同级 .token_cache
                       —— 大数据建议指到本地盘，别放网络挂载点
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack = pack

        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        data_path = Path(data_path)
        cache_dir = Path(cache_dir) if cache_dir else data_path.parent / ".token_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        tag = f"{data_path.stem}.v{tokenizer.get_vocab_size()}.L{max_length}.{'pack' if pack else 'pad'}"
        flat_path = cache_dir / f"{tag}.tokens.npy"
        off_path = cache_dir / f"{tag}.offsets.npy"

        if flat_path.exists() and off_path.exists():
            print(f"✓ Token 缓存命中: {flat_path}")
        else:
            print(f"预 tokenize {data_path} ...（只做一次，之后走缓存）")
            self._build_cache(data_path, flat_path, off_path)

        self.flat = np.load(flat_path, mmap_mode="r")
        self.offsets = np.load(off_path, mmap_mode="r")

        if self.pack:
            self.n_items = len(self.flat) // self.max_length
            print(f"✓ 已加载: {len(self.flat):,} tokens -> {self.n_items:,} 个定长块 (len={self.max_length})")
        else:
            self.n_items = len(self.offsets) - 1
            print(f"✓ 已加载: {self.n_items:,} 条样本（缓存命中，未重新 tokenize）")

    def _build_cache(self, data_path, flat_path, off_path):
        """一次性 tokenize 全量数据，流式落盘（分块 flush 控制内存）"""
        flat_tmp = str(flat_path) + ".tmp"
        off_tmp = str(off_path) + ".tmp"
        FLUSH = 1 << 21  # 每 ~200 万 token 落盘一次

        buf, offsets, total = [], [0], 0
        n_ok = 0

        with open(data_path, "r", encoding="utf-8") as fin, open(flat_tmp, "wb") as fout:
            for line in tqdm(fin, desc="tokenizing", unit="line"):
                line = line.strip()
                if not line:
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if not text:
                    continue

                ids = self.tokenizer.encode(str(text), add_special_tokens=False)

                if self.pack:
                    ids = ids + [self.eos_id]
                else:
                    if len(ids) > self.max_length - 2:
                        ids = ids[: self.max_length - 2]
                    ids = [self.bos_id] + ids + [self.eos_id]

                buf.extend(ids)
                n_ok += 1
                if not self.pack:
                    offsets.append(total + len(buf))

                if len(buf) >= FLUSH:
                    np.asarray(buf, dtype=np.int32).tofile(fout)
                    total += len(buf)
                    buf = []

            if buf:
                np.asarray(buf, dtype=np.int32).tofile(fout)
                total += len(buf)

        os.replace(flat_tmp, flat_path)
        np.asarray(offsets, dtype=np.int64).tofile(off_tmp)
        os.replace(off_tmp, off_path)
        print(f"✓ 缓存完成: {n_ok:,} 条样本 -> {total:,} tokens -> {flat_path}")

    def __len__(self):
        return self.n_items

    def __getitem__(self, index):
        if self.pack:
            start = index * self.max_length
            ids = np.asarray(self.flat[start:start + self.max_length], dtype=np.int64)
            input_ids = torch.from_numpy(ids)
            # 拼接块内没有 pad，全部位置都是有效 label
            return {"input_ids": input_ids, "labels": input_ids.clone()}

        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        ids = np.asarray(self.flat[start:end], dtype=np.int64)

        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = np.concatenate([ids, np.full(pad_len, self.pad_id, dtype=np.int64)])
        else:
            ids = ids[: self.max_length]

        input_ids = torch.from_numpy(ids)
        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100

        return {"input_ids": input_ids, "labels": labels}


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
