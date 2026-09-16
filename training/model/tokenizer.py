"""
PocketLLM 分词器封装

支持：
1. 使用 PocketLLM 的预训练分词器（词表 6400）
2. 使用 HuggingFace 的任意分词器
3. 训练自定义分词器（可选）
"""

from typing import List, Union, Optional
from pathlib import Path
import json


class Tokenizer:
    """
    分词器封装类

    优先使用 HuggingFace transformers 的 AutoTokenizer
    如果不可用，回退到 tokenizers 库
    """

    def __init__(self, tokenizer_path: str):
        """
        初始化分词器

        Args:
            tokenizer_path: 分词器目录路径（包含 tokenizer.json 和 tokenizer_config.json）
        """
        self.tokenizer_path = Path(tokenizer_path)
        self.tokenizer = None
        self.vocab_size = 0
        self.special_tokens = {}

        self._load_tokenizer()

    def _load_tokenizer(self):
        """加载分词器"""
        try:
            # 优先使用 transformers
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path),
                trust_remote_code=True
            )
            self.vocab_size = len(self.tokenizer)
            self._load_special_tokens()
            print(f"✓ Loaded tokenizer from {self.tokenizer_path}")
            print(f"  Vocab size: {self.vocab_size}")

        except ImportError:
            # 回退到 tokenizers 库
            try:
                from tokenizers import Tokenizer as HFTokenizer
                self.tokenizer = HFTokenizer.from_file(
                    str(self.tokenizer_path / "tokenizer.json")
                )
                self.vocab_size = self.tokenizer.get_vocab_size()
                print(f"✓ Loaded tokenizer from {self.tokenizer_path} (using tokenizers)")
                print(f"  Vocab size: {self.vocab_size}")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {e}")

    def _load_special_tokens(self):
        """加载特殊 token"""
        if hasattr(self.tokenizer, 'bos_token_id'):
            self.special_tokens['bos_token_id'] = self.tokenizer.bos_token_id
        if hasattr(self.tokenizer, 'eos_token_id'):
            self.special_tokens['eos_token_id'] = self.tokenizer.eos_token_id
        if hasattr(self.tokenizer, 'pad_token_id'):
            self.special_tokens['pad_token_id'] = self.tokenizer.pad_token_id
        if hasattr(self.tokenizer, 'unk_token_id'):
            self.special_tokens['unk_token_id'] = self.tokenizer.unk_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False
    ) -> List[int]:
        """
        编码文本为 token IDs

        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊 token（如 BOS/EOS）

        Returns:
            token IDs 列表
        """
        if hasattr(self.tokenizer, 'encode'):
            # transformers AutoTokenizer
            return self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens
            )
        else:
            # tokenizers Tokenizer
            encoding = self.tokenizer.encode(text)
            return encoding.ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        解码 token IDs 为文本

        Args:
            token_ids: token IDs 列表
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            解码后的文本
        """
        if hasattr(self.tokenizer, 'decode'):
            # transformers AutoTokenizer
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
        else:
            # tokenizers Tokenizer
            return self.tokenizer.decode(token_ids)

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = False,
        padding: bool = False,
        max_length: Optional[int] = None
    ) -> dict:
        """
        批量编码

        Args:
            texts: 文本列表
            add_special_tokens: 是否添加特殊 token
            padding: 是否填充到相同长度
            max_length: 最大长度

        Returns:
            包含 input_ids 和 attention_mask 的字典
        """
        if hasattr(self.tokenizer, '__call__'):
            # transformers AutoTokenizer
            return self.tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding,
                max_length=max_length,
                truncation=max_length is not None,
                return_tensors='pt'
            )
        else:
            # tokenizers Tokenizer - 手动实现
            encodings = [self.tokenizer.encode(text) for text in texts]
            input_ids = [enc.ids for enc in encodings]

            if padding:
                max_len = max_length or max(len(ids) for ids in input_ids)
                pad_id = self.special_tokens.get('pad_token_id', 0)
                input_ids = [
                    ids + [pad_id] * (max_len - len(ids))
                    for ids in input_ids
                ]

            return {'input_ids': input_ids}

    def get_vocab_size(self) -> int:
        """获取词表大小"""
        return self.vocab_size

    def get_special_tokens(self) -> dict:
        """获取特殊 token IDs"""
        return self.special_tokens

    @property
    def bos_token_id(self) -> Optional[int]:
        """BOS token ID"""
        return self.special_tokens.get('bos_token_id')

    @property
    def eos_token_id(self) -> Optional[int]:
        """EOS token ID"""
        return self.special_tokens.get('eos_token_id')

    @property
    def pad_token_id(self) -> Optional[int]:
        """PAD token ID"""
        return self.special_tokens.get('pad_token_id')


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("测试分词器")
    print("=" * 60)

    # 加载分词器
    import os
    tokenizer_path = os.path.join(os.path.dirname(__file__), "../tokenizer")
    tokenizer = Tokenizer(tokenizer_path)

    # 测试编码
    text = "你好，世界！Hello, World!"
    token_ids = tokenizer.encode(text)
    print(f"\n原文: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Token 数量: {len(token_ids)}")

    # 测试解码
    decoded = tokenizer.decode(token_ids)
    print(f"解码后: {decoded}")
    print(f"一致性: {decoded == text}")

    # 测试批量编码
    texts = ["你好", "世界", "Hello"]
    batch = tokenizer.batch_encode(texts, padding=True)
    print(f"\n批量编码: {batch['input_ids']}")

    # 特殊 token
    print(f"\n特殊 Token:")
    print(f"  BOS: {tokenizer.bos_token_id}")
    print(f"  EOS: {tokenizer.eos_token_id}")
    print(f"  PAD: {tokenizer.pad_token_id}")

    print("\n✅ 分词器测试完成！")
