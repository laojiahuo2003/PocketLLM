"""
可组合的注意力层实现
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAttentionLayer
from ..config import ModelConfig
from ..registry import LayerRegistry


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码（RoPE）"""
    # q, k: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """旋转输入张量的一半维度"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """旋转位置编码（RoPE）"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings

        # 预计算 cos/sin
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x, seq_len: int):
        """
        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
            seq_len: 序列长度

        Returns:
            cos, sin: [seq_len, head_dim]
        """
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
        )


@LayerRegistry.register_attention("multi_head")
class MultiHeadAttention(BaseAttentionLayer):
    """标准多头注意力"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        # Q, K, V 投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()

        # QKV 投影
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        cos, sin = self.rotary_emb(value_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 注意力计算（SDPA：避免物化分数矩阵）
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :]
            else:
                attn_mask = attention_mask
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                dropout_p=0.0,
                is_causal=True,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


@LayerRegistry.register_attention("grouped_query")
class GroupedQueryAttention(BaseAttentionLayer):
    """分组查询注意力（GQA）- 省 KV cache"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        # Q 投影（所有头）
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # K, V 投影（只有部分头）
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()

        # QKV 投影
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        cos, sin = self.rotary_emb(value_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 将 KV 头复制到 Q 头数量（GQA 的关键）
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # 注意力计算（SDPA：Ampere 上自动走 flash / memory-efficient 内核，
        # 避免物化 512×512 分数矩阵，显著降低显存与算力开销）
        if attention_mask is not None:
            # 兼容 [B, S] 的 0/1 mask 或 [B, 1, S, S] 的加性 mask
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :]  # [B,1,1,S]
            else:
                attn_mask = attention_mask
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                dropout_p=0.0,
                is_causal=True,
            )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value
