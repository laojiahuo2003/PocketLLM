# 从 llama.cpp 学到的设计经验

在研究了 llama.cpp 的代码后，我们可以借鉴以下优秀设计：

## 1. **Backend 抽象层**（最重要）

llama.cpp 使用 `ggml-backend.h` 实现了统一的后端接口，这是它能支持 CPU/CUDA/Metal/OpenCL/Vulkan 等多后端的关键。

### 我们的改进方案

```cpp
// backend.h
class Backend {
public:
    virtual void* Allocate(size_t size) = 0;
    virtual void Free(void* ptr) = 0;
    virtual void MatMul(Tensor& a, Tensor& b, Tensor& c) = 0;
    virtual void Attention(...) = 0;
    ...
};

class CPUBackend : public Backend { ... };
class NEONBackend : public Backend { ... };
class GPUBackend : public Backend { ... };
```

**好处**：
- 同一套算子代码，多个后端实现
- 运行时动态选择后端
- 方便添加新硬件支持

## 2. **GGUF 格式的设计思路**

llama.cpp 的 GGUF (GPT-Generated Unified Format) 格式非常优秀：

- **自描述**：文件包含所有元数据（架构、超参数、张量信息）
- **KV 结构**：用键值对存储配置，易于扩展
- **对齐优化**：张量数据按 32 字节对齐，加速加载
- **量化嵌入**：量化信息存在文件中，不需要额外配置

### 我们的 .pllm 格式可以借鉴

```
.pllm 文件（单文件，而非目录）:
[Header]
- Magic: "PLLM" (4 bytes)
- Version: uint32
- Metadata KV pairs
  - architecture: string
  - num_layers: int
  - hidden_size: int
  - ...

[Tensors]
- Tensor 1 (name, shape, type, data)
- Tensor 2
- ...
```

**优势**：
- 单文件，方便分发
- 自描述，不需要额外 JSON
- 易于扩展

## 3. **量化策略**

llama.cpp 的量化非常细致：

- **逐 block 量化**：不是整个 tensor 一个 scale，而是每 32/64 个元素一组
- **混合精度**：敏感层（如输入 embedding、输出层）保持 FP16
- **动态量化**：运行时根据硬件选择量化策略

### 我们应该实现的量化方案

```cpp
// 逐 block INT4 量化
struct QuantizedTensor {
    int8_t* data;           // 打包的 INT4 数据
    float* scales;          // 每个 block 的 scale
    int block_size;         // 如 32 或 64
};

// 量化时
for (int i = 0; i < num_blocks; i++) {
    float absmax = max(abs(tensor[i*block_size : (i+1)*block_size]));
    scales[i] = absmax / 7.0;
    // 量化这个 block
}
```

**好处**：
- 精度损失更小（1-2% vs 5-10%）
- 对异常值更鲁棒

## 4. **KV Cache 管理**

llama.cpp 有专门的 `llama-kv-cache.cpp`，管理注意力的 KV cache：

- **预分配池**：提前分配固定大小的 cache buffer
- **Sequence 管理**：支持多个并发序列（batch 推理）
- **循环覆盖**：长对话时自动丢弃旧 token 的 cache

### 我们的 KV Cache 设计

```cpp
class KVCache {
    Tensor key_cache;    // [num_layers, max_seq, num_heads, head_dim]
    Tensor value_cache;
    int current_length;

    void Append(int layer, const Tensor& k, const Tensor& v);
    Tensor Get(int layer);
    void Clear();
};
```

## 5. **算子注册机制**

llama.cpp 虽然用的是 C，但通过函数指针表实现了类似注册表的机制：

```c
struct ggml_op_impl {
    ggml_compute_fn fn;
    const char* name;
};

static struct ggml_op_impl ops[] = {
    { ggml_compute_forward_add, "add" },
    { ggml_compute_forward_mul, "mul" },
    ...
};
```

**我们已经实现**（C++ 的注册表更优雅）。

## 6. **SIMD 优化**

llama.cpp 对 ARM NEON 的优化非常激进：

```c
// 示例：向量点积
float vec_dot_f32(const float* a, const float* b, int n) {
#ifdef __ARM_NEON
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }
    return vaddvq_f32(sum);  // 水平求和
#else
    // 标量实现
#endif
}
```

### 我们应该优化的算子

1. **MatMul**（最耗时）
2. **RMSNorm**（频繁调用）
3. **RoPE**（位置编码）
4. **SwiGLU**（激活函数）

## 7. **内存管理**

llama.cpp 使用自己的 allocator (`ggml-alloc.h`)：

- **预分配**：模型加载时一次性分配所有内存
- **零拷贝**：权重直接 mmap，不需要额外拷贝
- **对齐**：保证内存对齐，加速 SIMD

### 我们可以实现

```cpp
class MemoryPool {
    void* buffer;
    size_t size;
    size_t offset;

public:
    void* Allocate(size_t bytes, size_t alignment = 32);
    void Reset();  // 推理结束后重置，复用内存
};
```

## 8. **错误处理**

llama.cpp 的错误处理非常详细：

```c
if (!model) {
    fprintf(stderr, "error: failed to load model from %s\n", path);
    return NULL;
}
```

**我们应该**：
- 每个关键步骤都有错误检查
- 提供清晰的错误信息（不是简单的 "failed"）
- 使用日志系统（可选的 verbose 模式）

## 9. **多线程支持**

llama.cpp 在 CPU 后端使用 OpenMP 并行：

```c
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    // 并行计算
}
```

**我们可以用**：
- OpenMP（简单）
- `std::thread`（更灵活）

## 10. **模型加载优化**

llama.cpp 使用 `mmap`：

```c
void* data = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
```

**好处**：
- 不占用内存（直到真正访问）
- 操作系统自动管理 page cache
- 加载速度极快

---

## 我们项目的改进清单

基于 llama.cpp 的经验，我们应该：

### 立即改进

1. ✅ **Backend 抽象层**（已在 `include/backend.h` 规划）
2. ✅ **单文件 .pllm 格式**（改进现有设计）
3. ✅ **逐 block 量化**（改进 `export.py`）
4. ✅ **KV Cache 管理**（新增 `kv_cache.h`）

### 中期改进

5. **SIMD 优化**（MatMul, RMSNorm, RoPE）
6. **内存池**（减少碎片化）
7. **mmap 加载**（加速启动）
8. **多线程推理**（CPU 并行）

### 长期优化

9. **动态量化**（运行时量化）
10. **模型编译优化**（算子融合）

---

## 项目结构调整

参考 llama.cpp，我们可以调整：

```
inference/cpp/
├── include/
│   ├── pllm.h              # 主 API（类似 llama.h）
│   ├── backend.h           # Backend 抽象（新增）
│   ├── kv_cache.h          # KV Cache 管理（新增）
│   ├── memory_pool.h       # 内存池（新增）
│   └── ... (现有文件)
├── src/
│   ├── backends/
│   │   ├── cpu/            # CPU 实现
│   │   │   ├── matmul.cpp
│   │   │   ├── attention.cpp
│   │   │   └── ...
│   │   ├── neon/           # ARM NEON 优化
│   │   │   ├── matmul_neon.cpp
│   │   │   └── ...
│   │   └── gpu/            # GPU（未来）
│   └── ...
```

---

## 总结

llama.cpp 的核心设计哲学：

1. **抽象 + 实现分离**：Backend 抽象使得多平台支持变得简单
2. **极致优化**：SIMD、量化、内存管理都做到极致
3. **实用主义**：功能完善（mmap、多线程、KV cache）
4. **单一职责**：每个模块只做一件事（kv-cache、sampler、vocab 都独立）

**我们的项目已经有了良好的基础（模块化、注册表、扩展性），现在需要补充这些工程细节。**
