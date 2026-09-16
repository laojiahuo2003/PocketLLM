"""
模型注册表
用于管理和动态加载不同的模型架构
"""

from typing import Dict, Type, Callable
from .base import BaseModel
from .config import ModelConfig


class ModelRegistry:
    """全局模型注册表"""

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        装饰器：注册模型架构

        用法:
            @ModelRegistry.register("llama_like")
            class LlamaLikeModel(BaseModel):
                ...
        """
        def decorator(model_cls: Type[BaseModel]):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """获取模型类"""
        if name not in cls._registry:
            raise ValueError(f"Unknown model architecture: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, config: ModelConfig) -> BaseModel:
        """创建模型实例"""
        model_cls = cls.get(name)
        return model_cls(config)

    @classmethod
    def list_available(cls) -> list:
        """列出所有可用架构"""
        return list(cls._registry.keys())


class LayerRegistry:
    """层注册表（用于可组合层）"""

    _attention_registry: Dict[str, Type] = {}
    _ffn_registry: Dict[str, Type] = {}

    @classmethod
    def register_attention(cls, name: str):
        """注册注意力层"""
        def decorator(layer_cls):
            cls._attention_registry[name] = layer_cls
            return layer_cls
        return decorator

    @classmethod
    def register_ffn(cls, name: str):
        """注册 FFN 层"""
        def decorator(layer_cls):
            cls._ffn_registry[name] = layer_cls
            return layer_cls
        return decorator

    @classmethod
    def get_attention(cls, name: str):
        """获取注意力层"""
        if name not in cls._attention_registry:
            raise ValueError(f"Unknown attention type: {name}")
        return cls._attention_registry[name]

    @classmethod
    def get_ffn(cls, name: str):
        """获取 FFN 层"""
        if name not in cls._ffn_registry:
            raise ValueError(f"Unknown FFN type: {name}")
        return cls._ffn_registry[name]
