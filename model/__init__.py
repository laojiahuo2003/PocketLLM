"""
PocketLLM 模型包

用法:
    from model import ModelConfig, ModelRegistry

    # 创建配置
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        ...
    )

    # 创建模型
    model = ModelRegistry.create("llama_like", config)

    # 或者使用预定义配置
    from model.config import SMALL_CONFIG
    model = ModelRegistry.create("llama_like", SMALL_CONFIG)
"""

from .config import ModelConfig, TINY_CONFIG, SMALL_CONFIG, BASE_CONFIG
from .base import BaseModel, BaseAttentionLayer, BaseFFNLayer
from .registry import ModelRegistry, LayerRegistry
from .export import export_model

# 导入具体实现以注册到 Registry
from .architectures import llama_like
from .layers import attention, ffn, norm

__all__ = [
    # 配置
    "ModelConfig",
    "TINY_CONFIG",
    "SMALL_CONFIG",
    "BASE_CONFIG",
    # 基类
    "BaseModel",
    "BaseAttentionLayer",
    "BaseFFNLayer",
    # 注册表
    "ModelRegistry",
    "LayerRegistry",
    # 工具
    "export_model",
]
