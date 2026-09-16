"""
PocketLLM 基础模型抽象
所有模型架构的统一接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from .config import ModelConfig


class BaseModel(nn.Module, ABC):
    """
    所有 PocketLLM 模型的基类

    新架构只需继承此类并实现 forward() 方法
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
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
            past_key_values: 新的 KV cache（如果 use_cache=True）
        """
        pass

    @abstractmethod
    def get_architecture_name(self) -> str:
        """返回架构名称（用于序列化）"""
        pass

    def get_num_params(self, trainable_only: bool = False) -> int:
        """计算参数量"""
        params = self.parameters() if not trainable_only else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)

    def export_to_pllm(self, output_path: str, quantize: Optional[str] = None):
        """
        导出为 .pllm 格式

        Args:
            output_path: 输出路径
            quantize: 量化方式 ('int4', 'int8', None)
        """
        from .export import export_model
        export_model(self, output_path, quantize)

    def get_memory_footprint(self) -> Dict[str, Any]:
        """计算内存占用"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())

        return {
            'params_mb': param_size / 1024 / 1024,
            'buffers_mb': buffer_size / 1024 / 1024,
            'total_mb': (param_size + buffer_size) / 1024 / 1024,
        }


class BaseAttentionLayer(nn.Module, ABC):
    """注意力层基类"""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
            past_key_value: (key, value) cache
        """
        pass


class BaseFFNLayer(nn.Module, ABC):
    """前馈网络层基类"""

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        pass
