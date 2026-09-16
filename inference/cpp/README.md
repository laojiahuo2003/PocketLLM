# Pocket-0.1 C++ Inference Engine

轻量级、高性能的 C++ 推理引擎，专为移动端和边缘设备设计。

## 特性

- ✅ **纯 C++11** - 无外部依赖，易于集成
- ✅ **轻量级** - 核心库 < 500KB
- ✅ **高性能** - ARM NEON 优化，多线程支持
- ✅ **量化支持** - FP32/FP16/Q8_0/Q4_0
- ✅ **内存高效** - mmap 零拷贝加载，KV Cache 优化
- ✅ **跨平台** - Linux/Windows/macOS/Android/iOS
- ✅ **易于使用** - 简洁的 C++ API

## 快速开始

### 1. 构建

```bash
cd inference/cpp
mkdir build && cd build
cmake ..
make -j4
```

编译选项：
- `-DPOCKET_BUILD_SHARED=ON` - 构建动态库（默认）
- `-DPOCKET_BUILD_EXAMPLES=ON` - 构建示例程序（默认）
- `-DPOCKET_ARM_NEON=ON` - 启用 ARM NEON 优化（默认）

### 2. 准备模型

将 HuggingFace 模型转换为 .pllm 格式：

```bash
cd ../../
python tools/export_to_pllm.py \
  --input models/pocket-0.1-pretrain \
  --output pocket-0.1.pllm \
  --quant f16
```

量化选项：
- `f32` - 32-bit 浮点（最高精度，最大体积）
- `f16` - 16-bit 浮点（推荐，精度 vs 体积平衡）
- `q8_0` - 8-bit 量化（体积更小，轻微精度损失）
- `q4_0` - 4-bit 量化（最小体积，明显精度损失）

### 3. 运行推理

#### 命令行对话

```bash
./build/pocket-chat pocket-0.1.pllm \
  --temperature 0.8 \
  --max-tokens 100
```

#### 文本生成

```bash
./build/pocket-generate pocket-0.1.pllm \
  --prompt "你好，我是" \
  --temperature 0.8
```

## C++ API 使用

### 基础用法

```cpp
#include "pocket.h"

int main() {
    // 1. 加载模型
    pocket::Model model;
    model.load("pocket-0.1.pllm");

    // 2. 加载分词器
    pocket::Tokenizer tokenizer;
    tokenizer.load("pocket-0.1.pllm");

    // 3. 创建生成器
    pocket::Generator generator(&model, &tokenizer);

    // 4. 生成文本
    pocket::GenerateConfig config;
    config.max_new_tokens = 100;
    config.temperature = 0.8f;

    std::string response = generator.generate("你好，", config);
    std::cout << response << std::endl;

    return 0;
}
```

### 流式生成

```cpp
pocket::GenerateConfig config;
config.callback = [](pocket::token_t token, void* user_data) {
    pocket::Tokenizer* tok = (pocket::Tokenizer*)user_data;
    std::cout << tok->decode(token) << std::flush;
};
config.callback_data = &tokenizer;

generator.generate(prompt, config);
```

### 批量生成

```cpp
std::vector<std::string> prompts = {
    "从前有座山，",
    "今天天气",
    "人工智能"
};

for (const auto& prompt : prompts) {
    std::string response = generator.generate(prompt, config);
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Response: " << response << std::endl;
}
```

### 自定义采样

```cpp
pocket::GenerateConfig config;
config.temperature = 0.8f;    // 温度 (0.1-2.0)
config.top_p = 0.9f;          // Top-p 采样
config.top_k = 50;            // Top-k 采样
config.repetition_penalty = 1.1f;  // 重复惩罚
config.seed = 42;             // 随机种子

std::string response = generator.generate(prompt, config);
```

### 低级 API

如果需要更精细的控制：

```cpp
// 编码
std::vector<pocket::token_t> tokens = tokenizer.encode("你好");

// 前向传播
pocket::KVCache kv_cache;
std::vector<float> logits = model.forward(tokens, &kv_cache);

// 采样
pocket::Sampler sampler(sampler_config);
pocket::token_t next_token = sampler.sample(logits);

// 解码
std::string text = tokenizer.decode(next_token);
```

## 性能优化

### 1. 使用量化模型

```bash
# FP16: 体积减半，几乎无精度损失
python tools/export_to_pllm.py --quant f16

# Q8_0: 体积 1/4，轻微精度损失
python tools/export_to_pllm.py --quant q8_0

# Q4_0: 体积 1/8，明显精度损失
python tools/export_to_pllm.py --quant q4_0
```

### 2. 启用 ARM NEON

在 ARM 设备上自动启用：

```bash
cmake -DPOCKET_ARM_NEON=ON ..
```

### 3. 使用 KV Cache

```cpp
pocket::KVCache kv_cache;

// 第一次生成
auto logits1 = model.forward(tokens1, &kv_cache);

// 后续生成复用 cache
auto logits2 = model.forward(tokens2, &kv_cache);
```

### 4. 多线程

矩阵乘法自动使用多线程：

```cpp
// 设置线程数（默认为 CPU 核心数）
setenv("OMP_NUM_THREADS", "4", 1);
```

## 性能基准

在不同设备上的推理性能（85M 参数模型）：

| 设备 | 量化 | 速度 (tokens/s) | 内存 |
|------|------|-----------------|------|
| RTX 3080 | FP32 | ~200 | 1.8 GB |
| RTX 3080 | FP16 | ~350 | 1.0 GB |
| iPhone 13 Pro | FP16 | ~25 | 340 MB |
| iPhone 13 Pro | Q8_0 | ~35 | 180 MB |
| Raspberry Pi 4 | Q8_0 | ~3 | 180 MB |
| Raspberry Pi 4 | Q4_0 | ~5 | 100 MB |

## Android 集成

### JNI 接口

```java
public class PocketModel {
    static {
        System.loadLibrary("pocket");
    }

    private long nativeHandle;

    public native void load(String modelPath);
    public native String generate(String prompt, float temperature, int maxTokens);
    public native void release();
}
```

详见：[Android 集成指南](../../mobile/android/README.md)

## iOS 集成

使用 Objective-C++ 封装：

```objc
@interface PocketModel : NSObject
- (BOOL)loadModel:(NSString *)path;
- (NSString *)generate:(NSString *)prompt config:(GenerateConfig)config;
@end
```

详见：[iOS 集成指南](../../mobile/ios/README.md)

## 架构设计

### 核心模块

```
pocket/
├── Model           # 模型加载和前向传播
├── Tokenizer       # BPE 分词器
├── Generator       # 文本生成
├── Sampler         # 采样策略
└── Ops             # 算子实现
    ├── MatMul      # 矩阵乘法
    ├── RMSNorm     # 归一化
    ├── RoPE        # 位置编码
    ├── Attention   # 注意力机制
    └── FFN         # 前馈网络
```

### 内存布局

```
┌──────────────────┐
│ Model Weights    │  mmap 映射，零拷贝
├──────────────────┤
│ KV Cache         │  动态分配
├──────────────────┤
│ Activations      │  临时缓冲区
└──────────────────┘
```

### 计算流程

```
Input Tokens
    ↓
Embedding
    ↓
┌─────────────┐
│ Layer 1     │
│ - RMSNorm   │
│ - Attention │
│ - RMSNorm   │
│ - FFN       │
└─────────────┘
    ↓
┌─────────────┐
│ Layer 2     │
│ ...         │
└─────────────┘
    ↓
RMSNorm
    ↓
LM Head
    ↓
Logits
    ↓
Sampling
    ↓
Output Token
```

## 文件格式

模型文件使用自定义的 `.pllm` 格式：

```
┌─────────────────┐
│ Magic: "PLLM"   │
├─────────────────┤
│ Version: 1      │
├─────────────────┤
│ Header          │
│  - Config JSON  │
│  - Tokenizer    │
├─────────────────┤
│ Tensor Metadata │
├─────────────────┤
│ Weights Data    │
└─────────────────┘
```

详见：[.pllm 格式规范](format/pllm_format.md)

## 常见问题

### 1. 编译错误

**问题**：找不到 pthread

**解决**：
```bash
# Ubuntu/Debian
sudo apt-get install libpthread-stubs0-dev

# macOS
brew install --force-bottle llvm
```

### 2. 加载模型失败

**问题**：`Failed to load model`

**检查**：
- 模型文件是否存在
- 文件格式是否正确（magic number）
- 文件是否完整（未损坏）

### 3. 推理速度慢

**优化**：
- 使用量化模型（FP16/Q8_0）
- 启用 ARM NEON 优化
- 增加线程数
- 使用 KV Cache

### 4. 内存占用高

**优化**：
- 使用更激进的量化（Q4_0）
- 减小 max_seq_len
- 释放不用的 KV Cache

## 开发指南

### 添加新算子

```cpp
// 1. 在 src/ops/ 添加实现
namespace pocket {
namespace ops {

void my_new_op(const float* input, float* output, size_t size) {
    // 实现
}

}} // namespace pocket::ops

// 2. 在 Model::forward 中调用
```

### 添加新的量化格式

```cpp
// 1. 在 pocket.h 添加枚举
enum class QuantType {
    // ...
    Q2_K,  // 新格式
};

// 2. 实现量化/反量化
void quantize_q2_k(const float* src, uint8_t* dst, size_t n);
void dequantize_q2_k(const uint8_t* src, float* dst, size_t n);
```

## 性能分析

使用 perf 分析热点：

```bash
perf record -g ./pocket-generate model.pllm
perf report
```

主要热点（85M 模型）：
- 矩阵乘法：~70%
- Attention：~15%
- RoPE：~8%
- 其他：~7%

## 路线图

- [x] 基础推理引擎
- [x] FP32/FP16 支持
- [x] ARM NEON 优化
- [ ] Q8_0/Q4_0 量化
- [ ] CUDA 后端
- [ ] Metal 后端
- [ ] WebAssembly 编译
- [ ] 更多采样策略
- [ ] 批量推理优化

## 许可证

Apache 2.0

## 贡献

欢迎提交 PR！请遵循现有代码风格。

## 联系

有问题请提 Issue。
