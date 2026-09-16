# PocketLLM 测试计划

完整的端到端测试流程，确保所有组件正常工作。

## 📋 测试检查清单

### Phase 1: 训练流程测试

- [ ] **预训练**
  ```bash
  python training/scripts/pretrain.py --config training/configs/pretrain_config.yaml
  ```
  - [ ] 模型能正常初始化
  - [ ] 数据加载正常
  - [ ] 训练损失下降
  - [ ] 检查点正常保存
  - [ ] SwanLab 监控正常

- [ ] **SFT 训练**
  ```bash
  python training/scripts/sft.py --config training/configs/sft_config.yaml
  ```
  - [ ] 从预训练模型加载成功
  - [ ] SFT 数据格式正确
  - [ ] 训练损失正常
  - [ ] 评估指标正常

- [ ] **DPO 训练**
  ```bash
  python training/scripts/dpo.py --config training/configs/dpo_config.yaml
  ```
  - [ ] 策略模型和参考模型加载
  - [ ] Reward margin 正常计算
  - [ ] 训练收敛

### Phase 2: Python 推理测试

- [ ] **模型转换**
  ```bash
  python tools/convert_to_hf.py \
    --input training/checkpoints/pretrain/final \
    --output models/pocket-0.1-pretrain
  ```
  - [ ] 转换无错误
  - [ ] 配置文件正确
  - [ ] 权重正确转换
  - [ ] README 生成

- [ ] **文本生成**
  ```bash
  python inference/python/generate.py \
    --model models/pocket-0.1-pretrain \
    --prompt "你好，我是"
  ```
  - [ ] 模型加载成功
  - [ ] 能够生成文本
  - [ ] 生成质量合理
  - [ ] 参数调整有效

- [ ] **命令行对话**
  ```bash
  python inference/python/chat.py \
    --model models/pocket-0.1-pretrain
  ```
  - [ ] 交互式对话正常
  - [ ] 历史管理正常
  - [ ] clear 命令有效
  - [ ] quit/exit 正常退出

### Phase 3: C++ 推理测试

- [ ] **模型导出**
  ```bash
  python tools/export_to_pllm.py \
    --input models/pocket-0.1-pretrain \
    --output pocket-0.1.pllm \
    --quant f32
  ```
  - [ ] .pllm 文件生成
  - [ ] 文件格式正确
  - [ ] 权重完整

- [ ] **编译 C++ 引擎**
  ```bash
  cd inference/cpp
  mkdir build && cd build
  cmake ..
  make -j4
  ```
  - [ ] 编译无错误
  - [ ] 所有库链接成功
  - [ ] 示例程序生成

- [ ] **C++ 文本生成**
  ```bash
  ./build/pocket-generate ../../pocket-0.1.pllm \
    --prompt "你好" --max-tokens 50
  ```
  - [ ] 模型加载成功
  - [ ] 生成文本正常
  - [ ] 性能合理

- [ ] **C++ 命令行对话**
  ```bash
  ./build/pocket-chat ../../pocket-0.1.pllm
  ```
  - [ ] 交互正常
  - [ ] 流式输出正常
  - [ ] 性能良好

### Phase 4: 性能测试

- [ ] **Python 推理性能**
  ```bash
  time python inference/python/generate.py \
    --model models/pocket-0.1-pretrain \
    --prompt "测试" --max-new-tokens 100
  ```
  - [ ] 记录 tokens/s
  - [ ] 记录内存占用
  - [ ] 与预期性能对比

- [ ] **C++ 推理性能**
  ```bash
  time ./build/pocket-generate pocket-0.1.pllm \
    --prompt "测试" --max-tokens 100
  ```
  - [ ] 记录 tokens/s
  - [ ] 记录内存占用
  - [ ] 对比 Python 版本

- [ ] **量化模型测试**
  ```bash
  # FP16
  python tools/export_to_pllm.py --quant f16
  ./build/pocket-generate pocket-0.1-f16.pllm --prompt "测试"
  
  # Q8_0
  python tools/export_to_pllm.py --quant q8_0
  ./build/pocket-generate pocket-0.1-q8.pllm --prompt "测试"
  ```
  - [ ] FP16 速度和精度
  - [ ] Q8_0 速度和精度
  - [ ] 量化效果对比

### Phase 5: 质量测试

- [ ] **生成质量评估**
  - [ ] 流畅性
  - [ ] 连贯性
  - [ ] 相关性
  - [ ] 多样性

- [ ] **不同采样策略**
  ```bash
  # Greedy
  --temperature 0.0
  
  # 低温度
  --temperature 0.3
  
  # 高温度
  --temperature 1.2
  
  # Top-K
  --top-k 10
  
  # Top-P
  --top-p 0.95
  ```
  - [ ] 各策略效果对比
  - [ ] 参数影响分析

### Phase 6: 边界情况测试

- [ ] **空输入**
  ```bash
  python inference/python/generate.py --model ... --prompt ""
  ```
  - [ ] 处理正常

- [ ] **长输入**
  ```bash
  # 超过 max_seq_len
  --prompt "很长很长的文本..."
  ```
  - [ ] 截断处理
  - [ ] 不崩溃

- [ ] **特殊字符**
  ```bash
  --prompt "测试😀emoji和\n换行符"
  ```
  - [ ] 正确处理

- [ ] **OOM 测试**
  ```bash
  --max-new-tokens 10000
  ```
  - [ ] 优雅处理
  - [ ] 错误提示

## 🔍 详细测试脚本

### 测试 1: 端到端流程

```bash
#!/bin/bash

echo "=== Phase 1: 训练 ==="
# 使用 smoke test 数据快速验证
python training/scripts/pretrain.py \
  --config training/configs/pretrain_config.yaml \
  --max-steps 100

echo "=== Phase 2: 转换 ==="
python tools/convert_to_hf.py \
  --input training/checkpoints/pretrain/final \
  --output models/test-model

echo "=== Phase 3: Python 推理 ==="
python inference/python/generate.py \
  --model models/test-model \
  --prompt "你好" \
  --max-new-tokens 20

echo "=== Phase 4: 导出 .pllm ==="
python tools/export_to_pllm.py \
  --input models/test-model \
  --output test-model.pllm

echo "=== Phase 5: C++ 推理 ==="
./inference/cpp/build/pocket-generate test-model.pllm \
  --prompt "你好" \
  --max-tokens 20

echo "=== 测试完成 ==="
```

### 测试 2: 性能基准

```bash
#!/bin/bash

MODEL="models/pocket-0.1-pretrain"
PROMPT="从前有座山，山上有座庙"
TOKENS=100

echo "=== Python 基准 ==="
/usr/bin/time -v python inference/python/generate.py \
  --model $MODEL \
  --prompt "$PROMPT" \
  --max-new-tokens $TOKENS \
  2>&1 | grep -E "(Maximum resident|User time)"

echo "=== C++ 基准 (FP32) ==="
/usr/bin/time -v ./inference/cpp/build/pocket-generate \
  pocket-0.1.pllm \
  --prompt "$PROMPT" \
  --max-tokens $TOKENS \
  2>&1 | grep -E "(Maximum resident|User time)"

echo "=== C++ 基准 (FP16) ==="
/usr/bin/time -v ./inference/cpp/build/pocket-generate \
  pocket-0.1-f16.pllm \
  --prompt "$PROMPT" \
  --max-tokens $TOKENS \
  2>&1 | grep -E "(Maximum resident|User time)"
```

### 测试 3: 质量评估

```python
#!/usr/bin/env python3
"""
生成质量评估脚本
"""

import sys
sys.path.insert(0, 'inference/python')

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/pocket-0.1-pretrain"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 测试用例
test_prompts = [
    "从前有座山，",
    "今天天气",
    "人工智能的未来是",
    "请介绍一下自己",
    "1+1=",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.8,
        do_sample=True
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
```

## 📊 性能目标

基于 RTX 3080 (10GB)，85M 参数模型：

| 配置 | 目标速度 | 目标内存 | 实际速度 | 实际内存 | 状态 |
|------|----------|----------|----------|----------|------|
| Python FP32 | 30+ tokens/s | < 2GB | ? | ? | ⏳ |
| Python FP16 | 60+ tokens/s | < 1GB | ? | ? | ⏳ |
| C++ FP32 | 100+ tokens/s | < 1GB | ? | ? | ⏳ |
| C++ FP16 | 200+ tokens/s | < 500MB | ? | ? | ⏳ |
| C++ Q8_0 | 250+ tokens/s | < 300MB | ? | ? | ⏳ |

## 🐛 已知问题跟踪

| 问题 | 状态 | 优先级 | 说明 |
|------|------|--------|------|
| JSON 解析简化 | Open | Medium | 需要完整的 JSON 库 |
| BPE tokenizer 未完整实现 | Open | High | 当前是占位符 |
| 量化算子未实现 | Open | Medium | Q8_0/Q4_0 |
| export_to_pllm.py 未创建 | Open | High | 必需工具 |

## ✅ 测试通过标准

- [ ] 所有训练脚本无错误运行
- [ ] Python 推理生成合理文本
- [ ] C++ 推理生成与 Python 一致
- [ ] 性能达到目标的 80%
- [ ] 内存占用在预期范围内
- [ ] 无内存泄漏
- [ ] 无段错误或崩溃

## 📝 测试报告模板

```markdown
# 测试报告 - [日期]

## 环境
- OS: 
- GPU: 
- 模型: 
- 配置: 

## 测试结果

### 训练
- 预训练: ✅/❌
- SFT: ✅/❌
- DPO: ✅/❌

### Python 推理
- 加载: ✅/❌
- 生成: ✅/❌
- 速度: X tokens/s
- 内存: X MB

### C++ 推理
- 编译: ✅/❌
- 加载: ✅/❌
- 生成: ✅/❌
- 速度: X tokens/s
- 内存: X MB

### 质量评估
- 流畅性: ⭐⭐⭐⭐⭐
- 连贯性: ⭐⭐⭐⭐⭐
- 相关性: ⭐⭐⭐⭐⭐

## 问题
1. 

## 建议
1. 
```

## 🚀 快速验证（训练完成后）

```bash
# 1分钟快速验证整个流程
cd /home/uos/code/PocketLLM

# 1. 转换模型
python tools/convert_to_hf.py \
  --input training/checkpoints/pretrain/final \
  --output models/pocket-0.1

# 2. Python 测试
python inference/python/generate.py \
  --model models/pocket-0.1 \
  --prompt "你好" \
  --max-new-tokens 10

# 如果上面都成功，说明训练和 Python 推理都正常！
```
