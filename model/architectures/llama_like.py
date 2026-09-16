"""
Llama-like 模型架构（第一版实现）

采用 Llama 风格的架构：
- RMSNorm
- RoPE 位置编码
- GQA（分组查询注意力）
- SwiGLU FFN
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from ..base import BaseModel
from ..config import ModelConfig
from ..registry import ModelRegistry, LayerRegistry
from ..layers.norm import RMSNorm


class LlamaDecoderLayer(nn.Module):
    """Llama 解码器层"""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # 注意力
        attention_type = "grouped_query" if config.num_key_value_heads != config.num_attention_heads else "multi_head"
        self.self_attn = LayerRegistry.get_attention(attention_type)(config)

        # FFN
        ffn_type = "swiglu" if config.hidden_act == "silu" else "standard"
        self.mlp = LayerRegistry.get_ffn(ffn_type)(config)

        # 归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states

        # 注意力
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


@ModelRegistry.register("llama_like")
class LlamaLikeModel(BaseModel):
    """
    Llama-like 模型

    特性：
    - RMSNorm（快速归一化）
    - RoPE 位置编码
    - GQA（省 KV cache）
    - SwiGLU FFN（强表达能力）
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder 层
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head
        if config.tie_word_embeddings:
            self.lm_head = None  # 与 embed_tokens 共享权重
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[Tuple[torch.FloatTensor]]]]:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            past_key_values: KV cache
            use_cache: 是否返回 KV cache

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            past_key_values: 新的 KV cache
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Decoder 层
        next_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                next_cache.append(present_key_value)

        # 最后的归一化
        hidden_states = self.norm(hidden_states)

        # LM Head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # 共享权重
            logits = torch.matmul(hidden_states, self.embed_tokens.weight.t())

        return logits, next_cache if use_cache else None

    def get_architecture_name(self) -> str:
        return "llama_like"

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        """
        简单的贪心生成（用于测试）

        实际使用时应该用更完善的生成策略
        """
        self.eval()

        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids, use_cache=False)
            logits = logits[:, -1, :] / temperature

            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            # Top-p 采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
