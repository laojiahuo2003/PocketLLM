#!/usr/bin/env python3
"""
预训练进度监控脚本

Usage:
    python tools/check_training.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta


def format_time(seconds):
    """格式化时间"""
    return str(timedelta(seconds=int(seconds)))


def check_process():
    """检查训练进程"""
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )

        lines = [l for l in result.stdout.split('\n') if 'pretrain.py' in l and 'grep' not in l]

        if lines:
            print("✓ 预训练进程运行中")
            for line in lines[:2]:  # 只显示前2个进程
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                print(f"  PID: {pid}, CPU: {cpu}%, MEM: {mem}%")
            return True
        else:
            print("✗ 未找到预训练进程")
            return False
    except Exception as e:
        print(f"✗ 检查进程失败: {e}")
        return False


def check_checkpoints():
    """检查训练检查点"""
    checkpoint_dir = Path("training/checkpoints/pretrain")

    if not checkpoint_dir.exists():
        print("✗ 检查点目录不存在")
        return None

    # 查找所有步骤检查点
    step_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])

    if not step_dirs:
        print("✗ 未找到检查点")
        return None

    print(f"✓ 找到 {len(step_dirs)} 个检查点")

    # 显示最新的几个检查点
    for step_dir in step_dirs[-3:]:
        step_num = step_dir.name.split("_")[1]

        # 检查文件
        config_file = step_dir / "config.json"
        model_file = step_dir / "pytorch_model.bin"

        if config_file.exists() and model_file.exists():
            size_mb = model_file.stat().st_size / 1024 / 1024
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            time_ago = datetime.now() - mtime

            print(f"  Step {step_num}: {size_mb:.1f} MB, {time_ago.total_seconds() / 60:.0f} 分钟前")

    return step_dirs[-1]


def check_logs():
    """检查训练日志"""
    log_file = Path("training/logs/pretrain.log")

    if not log_file.exists():
        print("✗ 日志文件不存在")
        return

    print(f"✓ 日志文件: {log_file}")

    # 读取最后几行
    try:
        with open(log_file) as f:
            lines = f.readlines()

        if lines:
            print(f"  总行数: {len(lines)}")
            print("\n  最后 5 行:")
            for line in lines[-5:]:
                print(f"    {line.rstrip()}")
    except Exception as e:
        print(f"  读取日志失败: {e}")


def check_swanlab():
    """检查 SwanLab 运行"""
    try:
        import subprocess
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )

        if 'swanlab' in result.stdout:
            print("✓ SwanLab 运行中")
            print("  访问: http://localhost:5173")
        else:
            print("ℹ SwanLab 未运行")
    except Exception as e:
        print(f"✗ 检查 SwanLab 失败: {e}")


def estimate_completion(config_path, latest_checkpoint):
    """估算完成时间"""
    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        total_epochs = config.get('training', {}).get('num_epochs', 0)

        if latest_checkpoint:
            step_num = int(latest_checkpoint.name.split("_")[1])

            # 尝试从日志推断
            log_file = Path("training/logs/pretrain.log")
            if log_file.exists():
                with open(log_file) as f:
                    lines = f.readlines()

                # 查找包含 epoch 的行
                for line in reversed(lines[-100:]):
                    if 'Epoch' in line or 'epoch' in line:
                        print(f"  最新日志: {line.rstrip()}")
                        break

        print(f"ℹ 配置: {total_epochs} epochs")

    except Exception as e:
        print(f"✗ 估算完成时间失败: {e}")


def main():
    print("=" * 60)
    print("PocketLLM 预训练进度检查")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 检查进程
    print("1. 检查训练进程")
    process_running = check_process()
    print()

    # 2. 检查检查点
    print("2. 检查训练检查点")
    latest_checkpoint = check_checkpoints()
    print()

    # 3. 检查日志
    print("3. 检查训练日志")
    check_logs()
    print()

    # 4. 检查 SwanLab
    print("4. 检查监控")
    check_swanlab()
    print()

    # 5. 估算完成时间
    print("5. 训练配置")
    config_path = Path("training/configs/pretrain_config.yaml")
    if config_path.exists():
        estimate_completion(config_path, latest_checkpoint)
    print()

    # 总结
    print("=" * 60)
    if process_running:
        print("✓ 训练正在进行中，请耐心等待")
        print("\n下一步:")
        print("  1. 等待训练完成")
        print("  2. 运行端到端测试:")
        print("     python tests/test_e2e.py --checkpoint training/checkpoints/pretrain/final")
    else:
        print("ℹ 训练已完成或未运行")
        if latest_checkpoint:
            print("\n可以运行测试:")
            print(f"  python tests/test_e2e.py --checkpoint {latest_checkpoint}")
    print("=" * 60)


if __name__ == "__main__":
    main()
