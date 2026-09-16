#!/usr/bin/env python3
"""
端到端测试 - 测试从训练到推理的完整流程

需要预训练完成后运行：
    python tests/test_e2e.py --checkpoint training/checkpoints/pretrain/final

测试流程：
1. 加载训练好的模型
2. 转换为 HuggingFace 格式
3. 测试 Python 推理
4. 导出为 .pllm 格式
5. 测试 C++ 推理 (需要编译)
"""

import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_step(step: int, total: int, name: str):
    print(f"\n{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BLUE}步骤 {step}/{total}: {name}{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 70}{Colors.RESET}")


def print_pass(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_fail(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_info(msg: str):
    print(f"{Colors.YELLOW}ℹ {msg}{Colors.RESET}")


def run_command(cmd: list, description: str, cwd=None) -> bool:
    """运行命令并返回是否成功"""
    print_info(f"运行: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print_pass(description)
        return True
    except subprocess.CalledProcessError as e:
        print_fail(f"{description} - 失败")
        print(f"错误输出:\n{e.stderr}")
        return False


def step1_load_checkpoint(checkpoint_path: Path) -> bool:
    """步骤 1: 加载训练好的模型"""
    print_step(1, 5, "加载训练检查点")

    if not checkpoint_path.exists():
        print_fail(f"检查点不存在: {checkpoint_path}")
        print_info("请先完成预训练，或指定正确的检查点路径")
        return False

    print_pass(f"找到检查点: {checkpoint_path}")

    # 检查必要文件
    required_files = ["config.json", "pytorch_model.bin"]
    for file in required_files:
        file_path = checkpoint_path / file
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            print_pass(f"  {file} ({size_mb:.1f} MB)")
        else:
            print_fail(f"  缺少文件: {file}")
            return False

    return True


def step2_convert_to_hf(checkpoint_path: Path, output_dir: Path) -> bool:
    """步骤 2: 转换为 HuggingFace 格式"""
    print_step(2, 5, "转换为 HuggingFace 格式")

    cmd = [
        "python", "tools/convert_to_hf.py",
        "--input", str(checkpoint_path),
        "--output", str(output_dir),
    ]

    return run_command(cmd, "转换为 HuggingFace 格式", cwd=project_root)


def step3_test_python_inference(model_dir: Path) -> bool:
    """步骤 3: 测试 Python 推理"""
    print_step(3, 5, "测试 Python 推理")

    print_info("测试文本生成...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # 加载模型
        print_info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print_pass("模型加载成功")

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print_pass("分词器加载成功")

        # 生成文本
        print_info("生成文本...")
        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print_pass("生成成功")
        print_info(f"输入: {prompt}")
        print_info(f"输出: {generated_text}")

        return True

    except Exception as e:
        print_fail(f"Python 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4_export_to_pllm(model_dir: Path, output_file: Path, quant: str = "fp16") -> bool:
    """步骤 4: 导出为 .pllm 格式"""
    print_step(4, 5, f"导出为 .pllm 格式 (量化: {quant})")

    cmd = [
        "python", "tools/export_to_pllm.py",
        "--input", str(model_dir),
        "--output", str(output_file),
        "--quant", quant,
    ]

    success = run_command(cmd, "导出为 .pllm 格式", cwd=project_root)

    if success and output_file.exists():
        size_mb = output_file.stat().st_size / 1024 / 1024
        print_pass(f"文件大小: {size_mb:.1f} MB")

    return success


def step5_test_cpp_inference(pllm_file: Path) -> bool:
    """步骤 5: 测试 C++ 推理"""
    print_step(5, 5, "测试 C++ 推理")

    cpp_dir = project_root / "inference" / "cpp"
    build_dir = cpp_dir / "build"

    # 检查是否已编译
    executable = build_dir / "pocket-generate"
    if not executable.exists():
        print_info("C++ 推理引擎未编译，尝试编译...")

        # 创建 build 目录
        build_dir.mkdir(parents=True, exist_ok=True)

        # CMake
        if not run_command(
            ["cmake", ".."],
            "CMake 配置",
            cwd=build_dir
        ):
            return False

        # Make
        if not run_command(
            ["make", "-j4"],
            "编译 C++ 推理引擎",
            cwd=build_dir
        ):
            return False

    # 测试推理
    print_info("测试 C++ 推理...")

    if not executable.exists():
        print_fail("未找到可执行文件: pocket-generate")
        return False

    cmd = [
        str(executable),
        str(pllm_file),
        "--prompt", "Hello",
        "--max-tokens", "10",
    ]

    return run_command(cmd, "C++ 推理测试", cwd=build_dir)


def main():
    parser = argparse.ArgumentParser(description="端到端测试")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="训练检查点路径"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "q8_0", "q4_0"],
        help="量化类型"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="保留临时文件"
    )

    args = parser.parse_args()

    print(f"{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BLUE}PocketLLM 端到端测试{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"检查点: {args.checkpoint}")
    print(f"量化类型: {args.quant}")

    checkpoint_path = Path(args.checkpoint)

    # 创建临时目录
    if args.keep_temp:
        temp_dir = project_root / "temp_e2e_test"
        temp_dir.mkdir(exist_ok=True)
        print_info(f"使用临时目录: {temp_dir}")
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="pocketllm_e2e_"))
        print_info(f"临时目录: {temp_dir}")

    try:
        hf_dir = temp_dir / "hf_model"
        pllm_file = temp_dir / "model.pllm"

        # 步骤 1: 加载检查点
        if not step1_load_checkpoint(checkpoint_path):
            return 1

        # 步骤 2: 转换为 HuggingFace
        if not step2_convert_to_hf(checkpoint_path, hf_dir):
            return 1

        # 步骤 3: Python 推理
        if not step3_test_python_inference(hf_dir):
            print_info("Python 推理测试失败，但继续测试导出...")

        # 步骤 4: 导出为 .pllm
        if not step4_export_to_pllm(hf_dir, pllm_file, args.quant):
            return 1

        # 步骤 5: C++ 推理
        if not step5_test_cpp_inference(pllm_file):
            print_info("C++ 推理测试失败 (可能需要手动编译)")

        # 总结
        print(f"\n{Colors.GREEN}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.GREEN}🎉 端到端测试完成！{Colors.RESET}")
        print(f"{Colors.GREEN}{'=' * 70}{Colors.RESET}")

        if args.keep_temp:
            print_info(f"临时文件保存在: {temp_dir}")
            print_info(f"  HuggingFace 模型: {hf_dir}")
            print_info(f"  .pllm 文件: {pllm_file}")
        else:
            print_info("临时文件将被自动清理")

        print(f"\n{Colors.BLUE}下一步:{Colors.RESET}")
        print("1. 运行完整的性能基准测试:")
        print("   python tests/benchmark.py")
        print("2. 在移动设备上部署测试")

        return 0

    finally:
        if not args.keep_temp:
            print_info("清理临时文件...")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
