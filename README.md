# PocketLLM

端侧大语言模型完整解决方案 - 从训练到移动端部署的全链路实现

## 项目简介

PocketLLM 是一个完全自研的端侧 LLM 项目，覆盖：

- **模型架构**：专为手机/端侧硬件优化的轻量化架构
- **训练流程**：预训练 → SFT → DPO 完整训练链路
- **推理引擎**：自研 C++ 推理引擎，高效、轻量
- **移动端部署**：打包为 Android APK，可独立运行

## 目录结构

```
PocketLLM/
├── model/            # 模型架构定义
├── training/         # 训练代码（预训练/SFT/DPO）
├── data/             # 语料数据
├── inference/        # C++ 推理引擎
├── android/          # Android 应用
├── tools/            # 工具脚本（量化/转换/测试）
└── docs/             # 文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 编译推理引擎
cd inference/cpp
mkdir build && cd build
cmake ..
make -j4
```

### 2. 模型训练

```bash
# 预训练
python training/pretrain.py --config training/configs/pretrain_config.yaml

# 监督微调
python training/sft.py --config training/configs/sft_config.yaml

# DPO 训练
python training/dpo.py --config training/configs/dpo_config.yaml
```

### 3. 模型导出与量化

```bash
# 导出模型
python model/export.py --checkpoint checkpoints/latest.pt --output model.bin

# 量化
python tools/quantize.py --input model.bin --output model_int4.bin --bits 4
```

### 4. Android 部署

```bash
cd android
./gradlew assembleRelease
```

## 硬件需求

### 训练环境
- GPU: NVIDIA RTX 3080 10GB 或更高
- 内存: 16GB+
- 存储: 100GB+

### 推理环境（移动端）
- Android 8.0+
- RAM: 6GB+
- 存储: 2GB+

## 技术栈

- **训练**: PyTorch + DeepSpeed
- **推理**: 自研 C++ 引擎（支持 ARM NEON/INT4 量化）
- **移动端**: Android NDK + JNI

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
