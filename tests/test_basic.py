"""
快速验证脚本：测试模型创建和基本功能
"""

import sys
from pathlib import Path

# 添加 training 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import ModelRegistry, TINY_CONFIG, SMALL_CONFIG, BASE_CONFIG

def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("PocketLLM 模型创建测试")
    print("=" * 60)

    # 测试配置
    configs = [
        ("Tiny", TINY_CONFIG),
        ("Small", SMALL_CONFIG),
        ("Base", BASE_CONFIG),
    ]

    for name, config in configs:
        print(f"\n[{name} Config]")
        print(f"  Hidden Size: {config.hidden_size}")
        print(f"  Num Layers: {config.num_layers}")
        print(f"  Attention Heads: {config.num_attention_heads}")
        print(f"  KV Heads: {config.num_key_value_heads}")
        print(f"  Intermediate Size: {config.intermediate_size}")

        # 创建模型
        model = ModelRegistry.create("llama_like", config)
        num_params = model.get_num_params()
        memory = model.get_memory_footprint()

        print(f"  Parameters: {num_params / 1e6:.1f}M")
        print(f"  Memory (FP32): {memory['total_mb']:.1f} MB")
        print(f"  Memory (INT4): ~{memory['total_mb'] / 8:.1f} MB")
        print(f"  ✓ Model created successfully")

def test_registry():
    """测试注册表"""
    print("\n" + "=" * 60)
    print("注册表测试")
    print("=" * 60)

    from model.registry import ModelRegistry, LayerRegistry

    print(f"\nAvailable Models: {ModelRegistry.list_available()}")
    print(f"Available Attention Layers: {list(LayerRegistry._attention_registry.keys())}")
    print(f"Available FFN Layers: {list(LayerRegistry._ffn_registry.keys())}")
    print("✓ Registry working")

def test_forward_shape():
    """测试前向传播形状"""
    import torch

    print("\n" + "=" * 60)
    print("前向传播形状测试")
    print("=" * 60)

    config = TINY_CONFIG
    model = ModelRegistry.create("llama_like", config)
    model.eval()

    # 输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # 前向传播
    with torch.no_grad():
        logits, _ = model(input_ids, use_cache=False)

    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Shape mismatch!"
    print("✓ Forward pass shape correct")

if __name__ == "__main__":
    try:
        test_model_creation()
        test_registry()
        test_forward_shape()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n下一步：")
        print("1. 实现分词器")
        print("2. 实现数据加载器")
        print("3. 实现训练脚本")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
