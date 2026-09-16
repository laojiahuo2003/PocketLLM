#!/usr/bin/env python3
"""
Export HuggingFace model to .pllm format for C++ inference.

Usage:
    python tools/export_to_pllm.py \
        --input models/pocket-0.1 \
        --output pocket-0.1.pllm \
        --quant fp16
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QuantType:
    """Quantization types matching C++ enum."""
    F32 = 0
    F16 = 1
    Q8_0 = 2
    Q4_0 = 3


def quantize_f16(tensor: np.ndarray) -> bytes:
    """Convert float32 to float16."""
    return tensor.astype(np.float16).tobytes()


def quantize_q8_0(tensor: np.ndarray) -> bytes:
    """
    Quantize to Q8_0 format: block-wise 8-bit quantization.
    Block size = 32 elements.
    Format per block: scale (fp16) + 32 int8 values
    """
    block_size = 32
    flat = tensor.flatten()

    # Pad to multiple of block_size
    pad_size = (block_size - len(flat) % block_size) % block_size
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size), mode='constant')

    n_blocks = len(flat) // block_size
    data = bytearray()

    for i in range(n_blocks):
        block = flat[i * block_size : (i + 1) * block_size]

        # Find scale (max absolute value)
        amax = np.abs(block).max()
        scale = amax / 127.0 if amax > 0 else 1.0

        # Quantize to int8
        quant = np.round(block / scale).astype(np.int8)

        # Write scale (fp16) + int8 values
        data.extend(struct.pack('e', scale))  # fp16
        data.extend(quant.tobytes())

    return bytes(data)


def quantize_q4_0(tensor: np.ndarray) -> bytes:
    """
    Quantize to Q4_0 format: block-wise 4-bit quantization.
    Block size = 32 elements.
    Format per block: scale (fp16) + 16 bytes (32 4-bit values packed)
    """
    block_size = 32
    flat = tensor.flatten()

    # Pad to multiple of block_size
    pad_size = (block_size - len(flat) % block_size) % block_size
    if pad_size > 0:
        flat = np.pad(flat, (0, pad_size), mode='constant')

    n_blocks = len(flat) // block_size
    data = bytearray()

    for i in range(n_blocks):
        block = flat[i * block_size : (i + 1) * block_size]

        # Find scale
        amax = np.abs(block).max()
        scale = amax / 7.0 if amax > 0 else 1.0

        # Quantize to 4-bit [-7, 7]
        quant = np.round(block / scale).astype(np.int8)
        quant = np.clip(quant, -7, 7)

        # Pack two 4-bit values into one byte
        packed = bytearray(16)
        for j in range(16):
            low = quant[j * 2] & 0x0F
            high = quant[j * 2 + 1] & 0x0F
            packed[j] = (high << 4) | low

        # Write scale (fp16) + packed values
        data.extend(struct.pack('e', scale))
        data.extend(packed)

    return bytes(data)


class PLLMExporter:
    """Export HuggingFace model to .pllm format."""

    def __init__(self, model_path: str, quant_type: str = "fp16"):
        self.model_path = Path(model_path)
        self.quant_type = quant_type.lower()

        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = self.model.config

        print(f"Model loaded: {self.config.model_type}")
        print(f"Quantization: {self.quant_type}")

    def _get_quant_type_id(self) -> int:
        """Get quantization type ID."""
        mapping = {
            "fp32": QuantType.F32,
            "fp16": QuantType.F16,
            "f16": QuantType.F16,
            "q8_0": QuantType.Q8_0,
            "q8": QuantType.Q8_0,
            "q4_0": QuantType.Q4_0,
            "q4": QuantType.Q4_0,
        }
        return mapping.get(self.quant_type, QuantType.F16)

    def _quantize_tensor(self, tensor: torch.Tensor) -> Tuple[bytes, int, List[int]]:
        """
        Quantize a tensor.
        Returns: (data_bytes, quant_type, original_shape)
        """
        np_tensor = tensor.detach().cpu().numpy()
        original_shape = list(np_tensor.shape)

        if self.quant_type in ["fp32", "f32"]:
            data = np_tensor.astype(np.float32).tobytes()
            quant_type = QuantType.F32
        elif self.quant_type in ["fp16", "f16"]:
            data = quantize_f16(np_tensor)
            quant_type = QuantType.F16
        elif self.quant_type in ["q8_0", "q8"]:
            data = quantize_q8_0(np_tensor)
            quant_type = QuantType.Q8_0
        elif self.quant_type in ["q4_0", "q4"]:
            data = quantize_q4_0(np_tensor)
            quant_type = QuantType.Q4_0
        else:
            raise ValueError(f"Unknown quantization type: {self.quant_type}")

        return data, quant_type, original_shape

    def _build_header(self) -> Dict:
        """Build JSON header with model config and vocabulary."""
        # Model configuration
        config_dict = {
            "model_type": self.config.model_type,
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": getattr(self.config, "num_key_value_heads",
                                           self.config.num_attention_heads),
            "intermediate_size": self.config.intermediate_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "rms_norm_eps": getattr(self.config, "rms_norm_eps", 1e-5),
            "rope_theta": getattr(self.config, "rope_theta", 10000.0),
            "bos_token_id": self.config.bos_token_id,
            "eos_token_id": self.config.eos_token_id,
            "pad_token_id": getattr(self.config, "pad_token_id", -1),
        }

        # Vocabulary
        vocab = self.tokenizer.get_vocab()
        vocab_list = sorted(vocab.items(), key=lambda x: x[1])
        vocab_tokens = [token for token, _ in vocab_list]

        # Special tokens
        special_tokens = {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "unk_token": self.tokenizer.unk_token,
            "pad_token": getattr(self.tokenizer, "pad_token", None),
        }

        header = {
            "config": config_dict,
            "vocab": vocab_tokens,
            "special_tokens": special_tokens,
            "tokenizer_type": "bpe",  # Assume BPE for now
        }

        return header

    def _collect_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """Collect all model tensors with standardized names."""
        tensors = []
        state_dict = self.model.state_dict()

        # Map HuggingFace names to .pllm names
        name_mapping = {
            "model.embed_tokens.weight": "embedding.weight",
            "model.norm.weight": "output_norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        for name, tensor in state_dict.items():
            # Handle transformer layers
            if "model.layers." in name:
                # Extract layer number
                parts = name.split(".")
                layer_idx = parts[2]
                rest = ".".join(parts[3:])

                # Map component names
                component_map = {
                    "input_layernorm.weight": "attn_norm.weight",
                    "self_attn.q_proj.weight": "attn.q_proj.weight",
                    "self_attn.k_proj.weight": "attn.k_proj.weight",
                    "self_attn.v_proj.weight": "attn.v_proj.weight",
                    "self_attn.o_proj.weight": "attn.o_proj.weight",
                    "post_attention_layernorm.weight": "ffn_norm.weight",
                    "mlp.gate_proj.weight": "ffn.gate_proj.weight",
                    "mlp.up_proj.weight": "ffn.up_proj.weight",
                    "mlp.down_proj.weight": "ffn.down_proj.weight",
                }

                if rest in component_map:
                    new_name = f"layers.{layer_idx}.{component_map[rest]}"
                    tensors.append((new_name, tensor))

            # Handle other weights
            elif name in name_mapping:
                tensors.append((name_mapping[name], tensor))

        # Sort by name for consistency
        tensors.sort(key=lambda x: x[0])

        return tensors

    def export(self, output_path: str):
        """Export model to .pllm format."""
        output_path = Path(output_path)
        print(f"\nExporting to {output_path}...")

        # Build header
        print("Building header...")
        header = self._build_header()
        header_json = json.dumps(header, ensure_ascii=False)
        header_bytes = header_json.encode('utf-8')

        # Collect and quantize tensors
        print("Collecting tensors...")
        tensors = self._collect_tensors()
        print(f"Found {len(tensors)} tensors")

        print(f"Quantizing to {self.quant_type}...")
        tensor_data = []
        tensor_metadata = []

        for i, (name, tensor) in enumerate(tensors):
            if (i + 1) % 10 == 0:
                print(f"  Processing {i + 1}/{len(tensors)}: {name}")

            data, quant_type, shape = self._quantize_tensor(tensor)
            tensor_data.append(data)
            tensor_metadata.append({
                "name": name,
                "shape": shape,
                "dtype": quant_type,
                "size": len(data),
            })

        # Write .pllm file
        print("Writing file...")
        with open(output_path, 'wb') as f:
            # 1. Magic number "PLLM" (4 bytes)
            f.write(b'PLLM')

            # 2. Version (uint32, little-endian)
            f.write(struct.pack('<I', 1))

            # 3. Header size (uint32)
            f.write(struct.pack('<I', len(header_bytes)))

            # 4. Header (JSON)
            f.write(header_bytes)

            # 5. Number of tensors (uint32)
            f.write(struct.pack('<I', len(tensors)))

            # 6. Tensor metadata
            for meta in tensor_metadata:
                # Name length + name
                name_bytes = meta["name"].encode('utf-8')
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)

                # Shape (rank + dimensions)
                f.write(struct.pack('<I', len(meta["shape"])))
                for dim in meta["shape"]:
                    f.write(struct.pack('<I', dim))

                # Data type (uint32)
                f.write(struct.pack('<I', meta["dtype"]))

                # Data size (uint64)
                f.write(struct.pack('<Q', meta["size"]))

            # 7. Tensor data
            for data in tensor_data:
                f.write(data)

        # Print summary
        file_size = output_path.stat().st_size
        print(f"\n✅ Export complete!")
        print(f"   Output: {output_path}")
        print(f"   Size: {file_size / 1024 / 1024:.2f} MB")
        print(f"   Tensors: {len(tensors)}")
        print(f"   Quantization: {self.quant_type}")

        # Calculate compression ratio
        original_size = sum(t.numel() * 4 for _, t in tensors)  # FP32 size
        ratio = original_size / file_size
        print(f"   Compression: {ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Export HuggingFace model to .pllm format")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to HuggingFace model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .pllm file path"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "f16", "q8_0", "q8", "q4_0", "q4"],
        help="Quantization type (default: fp16)"
    )

    args = parser.parse_args()

    # Export
    exporter = PLLMExporter(args.input, args.quant)
    exporter.export(args.output)

    print("\n🎉 Done! You can now use this model with C++ inference:")
    print(f"   ./pocket-chat {args.output}")


if __name__ == "__main__":
    main()
