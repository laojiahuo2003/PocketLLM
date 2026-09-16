"""
模型导出工具
将 PyTorch 模型导出为 .pllm 格式（推理引擎可读）
"""

import json
import struct
import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np

from .base import BaseModel


class PLLMExporter:
    """
    .pllm 格式导出器

    .pllm 目录结构:
        model_name.pllm/
        ├── model.json       # 计算图定义
        ├── weights.bin      # 权重数据
        ├── config.json      # 配置信息
        └── metadata.json    # 元数据
    """

    def __init__(self, model: BaseModel):
        self.model = model
        self.config = model.config

    def export(self, output_path: str, quantize: Optional[str] = None):
        """
        导出模型

        Args:
            output_path: 输出路径
            quantize: 量化方式 ('int4', 'int8', None)
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting model to {output_dir}")

        # 1. 导出计算图
        graph_def = self._build_graph()
        with open(output_dir / "model.json", "w") as f:
            json.dump(graph_def, f, indent=2)
        print("✓ Exported model.json")

        # 2. 导出权重
        self._export_weights(output_dir / "weights.bin", quantize)
        print(f"✓ Exported weights.bin (quantize={quantize})")

        # 3. 导出配置
        config_dict = self.config.to_dict()
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        print("✓ Exported config.json")

        # 4. 导出元数据
        metadata = self._build_metadata(quantize)
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("✓ Exported metadata.json")

        print(f"\n✅ Model exported successfully to {output_dir}")
        self._print_summary(output_dir, metadata)

    def _build_graph(self) -> Dict[str, Any]:
        """
        构建计算图定义

        这是一个简化版本，实际需要遍历模型结构
        """
        architecture = self.model.get_architecture_name()

        graph = {
            "architecture": architecture,
            "version": "1.0",
            "nodes": []
        }

        # TODO: 根据实际模型结构生成节点
        # 这里需要遍历 self.model 的所有 layer，生成对应的节点定义
        # 暂时返回占位符
        graph["nodes"].append({
            "id": "placeholder",
            "op": "Placeholder",
            "comment": "Graph generation will be implemented per architecture"
        })

        return graph

    def _export_weights(self, output_path: Path, quantize: Optional[str]):
        """导出权重到二进制文件"""
        with open(output_path, "wb") as f:
            # 写入魔数和版本
            f.write(b"PLLM")
            f.write(struct.pack("I", 1))  # version 1

            # 写入量化信息
            quant_type = {"int4": 1, "int8": 2, None: 0}.get(quantize, 0)
            f.write(struct.pack("I", quant_type))

            # 写入权重数量
            state_dict = self.model.state_dict()
            f.write(struct.pack("I", len(state_dict)))

            # 写入每个权重
            for name, param in state_dict.items():
                tensor = param.cpu().detach()

                # 量化
                if quantize == "int4":
                    tensor = self._quantize_int4(tensor)
                elif quantize == "int8":
                    tensor = self._quantize_int8(tensor)

                # 写入名称
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("I", len(name_bytes)))
                f.write(name_bytes)

                # 写入形状
                f.write(struct.pack("I", len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack("I", dim))

                # 写入数据
                data = tensor.numpy().tobytes()
                f.write(struct.pack("I", len(data)))
                f.write(data)

    def _quantize_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """INT4 量化（对称量化）"""
        # 简化版本：逐 tensor 量化
        # 实际应该逐 channel 或 group 量化
        absmax = tensor.abs().max()
        scale = absmax / 7.0  # INT4 范围 [-7, 7]
        quantized = torch.clamp(torch.round(tensor / scale), -7, 7)
        # 返回 int8 存储（两个 int4 打包成一个 int8）
        return quantized.to(torch.int8)

    def _quantize_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """INT8 量化（对称量化）"""
        absmax = tensor.abs().max()
        scale = absmax / 127.0
        quantized = torch.clamp(torch.round(tensor / scale), -127, 127)
        return quantized.to(torch.int8)

    def _build_metadata(self, quantize: Optional[str]) -> Dict[str, Any]:
        """构建元数据"""
        return {
            "architecture": self.model.get_architecture_name(),
            "num_params": self.model.get_num_params(),
            "quantization": quantize or "none",
            "config": self.config.to_dict(),
        }

    def _print_summary(self, output_dir: Path, metadata: Dict[str, Any]):
        """打印导出摘要"""
        total_size = sum(f.stat().st_size for f in output_dir.iterdir())
        print(f"\n📊 Summary:")
        print(f"  Architecture: {metadata['architecture']}")
        print(f"  Parameters: {metadata['num_params'] / 1e6:.1f}M")
        print(f"  Quantization: {metadata['quantization']}")
        print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")


def export_model(model: BaseModel, output_path: str, quantize: Optional[str] = None):
    """
    便捷函数：导出模型

    Args:
        model: 模型实例
        output_path: 输出路径
        quantize: 量化方式 ('int4', 'int8', None)
    """
    exporter = PLLMExporter(model)
    exporter.export(output_path, quantize)
