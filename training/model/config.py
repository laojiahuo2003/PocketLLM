"""
PocketLLM 模型配置
针对端侧推理优化的超参数配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型架构配置"""

    # 模型规模
    vocab_size: int = 32000  # 词表大小
    hidden_size: int = 768   # 隐藏层维度
    num_layers: int = 12     # Transformer 层数
    num_attention_heads: int = 12  # 注意力头数
    num_key_value_heads: Optional[int] = None  # GQA 的 KV 头数（None 则等于 num_attention_heads）

    # FFN 配置
    intermediate_size: int = 2048  # FFN 中间层维度

    # 位置编码
    max_position_embeddings: int = 2048  # 最大序列长度
    rope_theta: float = 10000.0  # RoPE 的 theta 参数

    # 归一化
    rms_norm_eps: float = 1e-6

    # Dropout
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # 激活函数
    hidden_act: str = "silu"  # silu (SwiGLU), gelu, relu

    # 初始化
    initializer_range: float = 0.02

    # 其他
    tie_word_embeddings: bool = False  # 是否共享输入输出 embedding
    use_cache: bool = True  # 推理时是否使用 KV cache

    def __post_init__(self):
        """自动设置派生参数"""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # 验证配置合法性
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_attention_heads

    def to_dict(self):
        """转换为字典"""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'rope_theta': self.rope_theta,
            'rms_norm_eps': self.rms_norm_eps,
            'attention_dropout': self.attention_dropout,
            'hidden_dropout': self.hidden_dropout,
            'hidden_act': self.hidden_act,
            'initializer_range': self.initializer_range,
            'tie_word_embeddings': self.tie_word_embeddings,
            'use_cache': self.use_cache,
        }


# 预定义配置
TINY_CONFIG = ModelConfig(
    vocab_size=32000,
    hidden_size=512,
    num_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,  # GQA
    intermediate_size=1536,
    max_position_embeddings=2048,
)  # ~100M 参数

SMALL_CONFIG = ModelConfig(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,  # GQA
    intermediate_size=2048,
    max_position_embeddings=2048,
)  # ~300M 参数

BASE_CONFIG = ModelConfig(
    vocab_size=32000,
    hidden_size=1024,
    num_layers=16,
    num_attention_heads=16,
    num_key_value_heads=4,  # GQA
    intermediate_size=2816,
    max_position_embeddings=2048,
)  # ~600M 参数
