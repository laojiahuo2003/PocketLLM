"""
可组合的 FFN 层实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseFFNLayer
from ..config import ModelConfig
from ..registry import LayerRegistry


@LayerRegistry.register_ffn("standard")
class StandardFFN(BaseFFNLayer):
    """标准前馈网络"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # 激活函数
        self.act_fn = self._get_activation(config.hidden_act)

    def _get_activation(self, name: str):
        """获取激活函数"""
        activations = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch_size, seq_len, hidden_size]
        up = self.gate_proj(hidden_states)
        up = self.act_fn(up)
        down = self.down_proj(up)
        return down


@LayerRegistry.register_ffn("swiglu")
class SwiGLUFFN(BaseFFNLayer):
    """
    SwiGLU FFN（Swish-Gated Linear Unit）

    Paper: GLU Variants Improve Transformer
    更强的表达能力，但计算量约为标准 FFN 的 1.5 倍
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # SwiGLU 需要两个 gate
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(W1 @ x) ⊙ (W2 @ x)) @ W3
        gate = self.gate_proj(hidden_states)
        gate = F.silu(gate)  # Swish activation
        up = self.up_proj(hidden_states)
        intermediate = gate * up  # 门控机制
        down = self.down_proj(intermediate)
        return down


@LayerRegistry.register_ffn("geglu")
class GeGLUFFN(BaseFFNLayer):
    """
    GeGLU FFN（GELU-Gated Linear Unit）

    类似 SwiGLU，但使用 GELU 激活
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        gate = F.gelu(gate)
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        down = self.down_proj(intermediate)
        return down
