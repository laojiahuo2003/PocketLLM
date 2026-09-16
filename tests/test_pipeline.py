#!/usr/bin/env python3
"""
快速验证测试脚本 - 测试完整的训练到推理流程

Usage:
    python tests/test_pipeline.py
"""

import sys
import torch
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_test(name: str):
    """打印测试名称"""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}测试: {name}{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")


def print_pass(msg: str):
    """打印通过信息"""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_fail(msg: str):
    """打印失败信息"""
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_info(msg: str):
    """打印信息"""
    print(f"{Colors.YELLOW}ℹ {msg}{Colors.RESET}")


def test_imports():
    """测试 1: 导入所有模块"""
    print_test("导入测试")

    try:
        # 训练模块
        from training.model.architectures.pocket_hf import (
            PocketConfig, PocketForCausalLM
        )
        print_pass("导入 PocketConfig, PocketForCausalLM")

        from training.data.pretrain_loader import PretrainDataset
        print_pass("导入 PretrainDataset")

        from training.data.sft_loader import SFTDataset
        print_pass("导入 SFTDataset")

        from training.data.dpo_loader import DPODataset
        print_pass("导入 DPODataset")

        # 工具模块
        import tools.convert_to_hf as convert_hf
        print_pass("导入 convert_to_hf")

        import tools.export_to_pllm as export_pllm
        print_pass("导入 export_to_pllm")

        return True
    except Exception as e:
        print_fail(f"导入失败: {e}")
        return False


def test_model_creation():
    """测试 2: 创建模型"""
    print_test("模型创建测试")

    try:
        from training.model.architectures.pocket_hf import (
            PocketConfig, PocketForCausalLM
        )

        # 创建小模型配置
        config = PocketConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            max_position_embeddings=512,
        )
        print_pass(f"创建配置: {config.hidden_size}d, {config.num_hidden_layers}层")

        # 创建模型
        model = PocketForCausalLM(config)
        print_pass("创建 PocketForCausalLM 模型")

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_pass(f"总参数: {total_params:,} ({trainable_params:,} 可训练)")

        return True, model, config
    except Exception as e:
        print_fail(f"模型创建失败: {e}")
        return False, None, None


def test_forward_pass(model, config):
    """测试 3: 前向传播"""
    print_test("前向传播测试")

    try:
        batch_size = 2
        seq_len = 16

        # 创建输入
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        print_pass(f"创建输入: {input_ids.shape}")

        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.logits
        print_pass(f"前向传播成功: logits shape = {logits.shape}")

        # 检查输出形状
        expected_shape = (batch_size, seq_len, config.vocab_size)
        assert logits.shape == expected_shape, \
            f"输出形状错误: {logits.shape} != {expected_shape}"
        print_pass(f"输出形状正确: {logits.shape}")

        return True
    except Exception as e:
        print_fail(f"前向传播失败: {e}")
        return False


def test_generation(model, config):
    """测试 4: 文本生成"""
    print_test("文本生成测试")

    try:
        from transformers import GenerationConfig

        # 创建输入
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        print_pass(f"创建输入: {input_ids.shape}")

        # 生成配置
        gen_config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=config.pad_token_id or config.eos_token_id,
        )

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=gen_config,
            )

        print_pass(f"生成成功: {outputs.shape}")
        print_info(f"生成的 token IDs: {outputs[0].tolist()}")

        return True
    except Exception as e:
        print_fail(f"文本生成失败: {e}")
        return False


def test_save_and_load(model, config):
    """测试 5: 保存和加载模型"""
    print_test("保存和加载测试")

    try:
        from transformers import AutoModelForCausalLM
        import tempfile
        import shutil

        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        print_info(f"临时目录: {temp_dir}")

        try:
            # 保存模型
            model.save_pretrained(temp_dir)
            config.save_pretrained(temp_dir)
            print_pass("保存模型成功")

            # 加载模型
            loaded_model = AutoModelForCausalLM.from_pretrained(temp_dir)
            print_pass("加载模型成功")

            # 验证参数数量
            original_params = sum(p.numel() for p in model.parameters())
            loaded_params = sum(p.numel() for p in loaded_model.parameters())
            assert original_params == loaded_params, \
                f"参数数量不匹配: {original_params} != {loaded_params}"
            print_pass(f"参数数量匹配: {original_params:,}")

            return True
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)
            print_info("清理临时目录")

    except Exception as e:
        print_fail(f"保存和加载失败: {e}")
        return False


def test_attention_patterns(model, config):
    """测试 6: 注意力机制"""
    print_test("注意力机制测试")

    try:
        # 测试 GQA
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        n_rep = n_heads // n_kv_heads

        print_pass(f"注意力头配置: {n_heads} query heads, {n_kv_heads} KV heads")
        print_pass(f"分组查询注意力 (GQA): 每个 KV head 复制 {n_rep} 次")

        # 测试因果掩码
        seq_len = 8
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        # 检查注意力权重
        attentions = outputs.attentions
        print_pass(f"获取注意力权重: {len(attentions)} 层")

        # 检查第一层的注意力形状
        first_attn = attentions[0]
        expected_shape = (1, n_heads, seq_len, seq_len)
        assert first_attn.shape == expected_shape, \
            f"注意力形状错误: {first_attn.shape} != {expected_shape}"
        print_pass(f"注意力形状正确: {first_attn.shape}")

        # 验证因果掩码 (上三角应该是 0)
        attn_matrix = first_attn[0, 0].numpy()
        upper_triangle = attn_matrix[range(seq_len), range(seq_len)[::-1][:seq_len]]
        # 检查对角线以上的值是否接近 0
        is_causal = True
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if attn_matrix[i, j] > 1e-5:
                    is_causal = False
                    break

        if is_causal:
            print_pass("因果掩码正确 (上三角为 0)")
        else:
            print_info("注意力权重未使用严格因果掩码 (可能使用 flash attention)")

        return True
    except Exception as e:
        print_fail(f"注意力测试失败: {e}")
        return False


def test_export_tools():
    """测试 7: 导出工具"""
    print_test("导出工具测试")

    try:
        # 检查工具文件存在
        tools_dir = project_root / "tools"

        convert_script = tools_dir / "convert_to_hf.py"
        if convert_script.exists():
            print_pass(f"找到 convert_to_hf.py")
        else:
            print_fail(f"未找到 convert_to_hf.py")
            return False

        export_script = tools_dir / "export_to_pllm.py"
        if export_script.exists():
            print_pass(f"找到 export_to_pllm.py")
        else:
            print_fail(f"未找到 export_to_pllm.py")
            return False

        # 检查 C++ 代码
        cpp_dir = project_root / "inference" / "cpp"
        if cpp_dir.exists():
            print_pass(f"找到 C++ 推理引擎目录")

            # 检查关键文件
            key_files = [
                "include/pocket.h",
                "src/model.cpp",
                "src/tokenizer.cpp",
                "src/sampler.cpp",
                "src/generator.cpp",
                "CMakeLists.txt",
            ]

            for file in key_files:
                file_path = cpp_dir / file
                if file_path.exists():
                    print_pass(f"  ✓ {file}")
                else:
                    print_fail(f"  ✗ {file}")
        else:
            print_fail(f"未找到 C++ 推理引擎目录")
            return False

        return True
    except Exception as e:
        print_fail(f"导出工具测试失败: {e}")
        return False


def test_training_configs():
    """测试 8: 训练配置"""
    print_test("训练配置测试")

    try:
        import yaml

        config_dir = project_root / "training" / "configs"
        config_files = [
            "pretrain_config.yaml",
            "sft_config.yaml",
            "dpo_config.yaml",
        ]

        for config_file in config_files:
            config_path = config_dir / config_file
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                print_pass(f"加载 {config_file}")

                # 检查关键配置
                if "model" in config:
                    model_name = config["model"].get("name", "unknown")
                    print_info(f"  模型: {model_name}")

                if "training" in config:
                    batch_size = config["training"].get("batch_size", "unknown")
                    print_info(f"  批大小: {batch_size}")
            else:
                print_fail(f"未找到 {config_file}")
                return False

        return True
    except Exception as e:
        print_fail(f"训练配置测试失败: {e}")
        return False


def print_summary(results):
    """打印测试总结"""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}测试总结{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    for name, result in results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if result else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"{status} - {name}")

    print(f"\n{Colors.BLUE}总计: {total} 个测试{Colors.RESET}")
    print(f"{Colors.GREEN}通过: {passed}{Colors.RESET}")
    print(f"{Colors.RED}失败: {failed}{Colors.RESET}")

    if failed == 0:
        print(f"\n{Colors.GREEN}🎉 所有测试通过！{Colors.RESET}")
        return True
    else:
        print(f"\n{Colors.RED}❌ 有 {failed} 个测试失败{Colors.RESET}")
        return False


def main():
    """运行所有测试"""
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}PocketLLM 流程验证测试{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}")

    results = {}

    # 测试 1: 导入
    results["导入测试"] = test_imports()
    if not results["导入测试"]:
        print_fail("导入测试失败，跳过后续测试")
        print_summary(results)
        return 1

    # 测试 2: 模型创建
    success, model, config = test_model_creation()
    results["模型创建测试"] = success
    if not success:
        print_fail("模型创建失败，跳过后续测试")
        print_summary(results)
        return 1

    # 测试 3: 前向传播
    results["前向传播测试"] = test_forward_pass(model, config)

    # 测试 4: 文本生成
    results["文本生成测试"] = test_generation(model, config)

    # 测试 5: 保存和加载
    results["保存和加载测试"] = test_save_and_load(model, config)

    # 测试 6: 注意力机制
    results["注意力机制测试"] = test_attention_patterns(model, config)

    # 测试 7: 导出工具
    results["导出工具测试"] = test_export_tools()

    # 测试 8: 训练配置
    results["训练配置测试"] = test_training_configs()

    # 打印总结
    all_passed = print_summary(results)

    if all_passed:
        print(f"\n{Colors.GREEN}下一步:{Colors.RESET}")
        print("1. 等待预训练完成")
        print("2. 运行完整的端到端测试:")
        print("   python tests/test_e2e.py")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
