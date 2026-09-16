# .pllm 文件格式规范

Pocket-0.1 模型的自定义二进制格式，专为移动端设计。

## 设计目标

- **简单易解析** - 无需复杂的库
- **高效加载** - 支持 mmap
- **紧凑存储** - 支持量化
- **可扩展** - 向后兼容

## 文件结构

```
┌─────────────────────────────────────┐
│ Magic Number (4 bytes)              │  "PLLM"
├─────────────────────────────────────┤
│ Version (4 bytes)                   │  uint32_t: 1
├─────────────────────────────────────┤
│ Header (variable)                   │
│  - ModelConfig (JSON)               │
│  - Tokenizer vocab                  │
├─────────────────────────────────────┤
│ Tensor Metadata (variable)          │
│  - Tensor 1: name, shape, qtype     │
│  - Tensor 2: name, shape, qtype     │
│  - ...                              │
├─────────────────────────────────────┤
│ Weights Data (variable)             │
│  - Tensor 1 data                    │
│  - Tensor 2 data                    │
│  - ...                              │
└─────────────────────────────────────┘
```

## 详细格式

### 1. Magic Number (4 bytes)

```
0x50 0x4C 0x4C 0x4D  ("PLLM")
```

用于快速验证文件类型。

### 2. Version (4 bytes)

```
uint32_t version = 1;
```

版本号，用于向后兼容。

### 3. Header

#### 3.1 Header Size (4 bytes)

```
uint32_t header_size;
```

整个 header 部分的字节数（不包括 magic 和 version）。

#### 3.2 ModelConfig (JSON)

使用 JSON 存储模型配置，便于调试和扩展。

```json
{
  "architecture": "pocket",
  "version": 1,
  "vocab_size": 6400,
  "hidden_size": 768,
  "num_layers": 12,
  "num_heads": 12,
  "num_kv_heads": 4,
  "intermediate_size": 2048,
  "max_seq_len": 2048,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "quant_type": "F32"
}
```

格式：
```
uint32_t config_size;    // JSON 字符串长度
char config_json[...];   // JSON 字符串（UTF-8）
```

#### 3.3 Tokenizer

存储词表和特殊 token。

```
uint32_t vocab_size;
for (int i = 0; i < vocab_size; i++) {
    uint32_t token_len;
    char token[token_len];
}
```

### 4. Tensor Metadata

存储每个张量的元数据，但不包含实际数据。

```
uint32_t num_tensors;

for (int i = 0; i < num_tensors; i++) {
    // 张量名称
    uint32_t name_len;
    char name[name_len];
    
    // 形状
    uint32_t ndim;
    uint64_t shape[ndim];  // [dim0, dim1, ...]
    
    // 量化类型
    uint8_t qtype;  // 0=F32, 1=F16, 2=Q8_0, 3=Q4_0
    
    // 数据偏移和大小
    uint64_t data_offset;  // 在文件中的偏移量
    uint64_t data_size;    // 字节数
}
```

### 5. Weights Data

实际的权重数据，按照 tensor metadata 中的 offset 和 size 存储。

数据紧密排列，支持 mmap 零拷贝加载。

## 张量命名规范

遵循 HuggingFace 的命名约定：

```
model.embed_tokens.weight                    [vocab_size, hidden_size]

model.layers.{i}.input_layernorm.weight      [hidden_size]
model.layers.{i}.self_attn.q_proj.weight     [hidden_size, num_heads * head_dim]
model.layers.{i}.self_attn.k_proj.weight     [hidden_size, num_kv_heads * head_dim]
model.layers.{i}.self_attn.v_proj.weight     [hidden_size, num_kv_heads * head_dim]
model.layers.{i}.self_attn.o_proj.weight     [num_heads * head_dim, hidden_size]

model.layers.{i}.post_attention_layernorm.weight  [hidden_size]
model.layers.{i}.mlp.gate_proj.weight        [hidden_size, intermediate_size]
model.layers.{i}.mlp.up_proj.weight          [hidden_size, intermediate_size]
model.layers.{i}.mlp.down_proj.weight        [intermediate_size, hidden_size]

model.norm.weight                            [hidden_size]
lm_head.weight                               [vocab_size, hidden_size]
```

## 量化格式

### FP32 (0)

标准 32-bit 浮点数。

```
float data[numel];
```

### FP16 (1)

16-bit 浮点数。

```
float16 data[numel];
```

### Q8_0 (2)

8-bit 块量化。每 32 个元素一个块。

```
struct Block {
    float scale;      // 缩放因子
    int8_t data[32];  // 量化后的数据
};

Block blocks[numel / 32];
```

反量化：`value = scale * data[i]`

### Q4_0 (3)

4-bit 块量化。每 32 个元素一个块。

```
struct Block {
    float scale;       // 缩放因子
    uint8_t data[16];  // 每个字节存储 2 个 4-bit 值
};

Block blocks[numel / 32];
```

反量化：
```
int4_t q = (data[i/2] >> ((i%2) * 4)) & 0xF;  // 提取 4-bit 值
q -= 8;  // 转换为有符号 [-8, 7]
value = scale * q;
```

## 字节对齐

所有数据按 64 字节对齐，优化缓存性能：

```
if (offset % 64 != 0) {
    offset = (offset / 64 + 1) * 64;
}
```

## 加载流程

### 方式 1：完整加载 (小模型)

```cpp
// 1. 读取整个文件到内存
std::vector<uint8_t> data = read_file(path);

// 2. 解析 header
parse_header(data);

// 3. 解析 tensor metadata
parse_tensors(data);

// 4. 所有数据已在内存
```

### 方式 2：mmap 映射 (大模型)

```cpp
// 1. mmap 文件
void* mapped = mmap(path);

// 2. 解析 header (拷贝到内存)
parse_header(mapped);

// 3. 解析 tensor metadata (拷贝到内存)
parse_tensors(mapped);

// 4. weights 数据零拷贝访问
tensor.data = (float*)(mapped + tensor.offset);
```

## 文件大小估算

以 85M 参数模型为例：

| 量化类型 | 参数大小 | Overhead | 总大小 |
|----------|----------|----------|--------|
| FP32     | 340 MB   | ~1 MB    | 341 MB |
| FP16     | 170 MB   | ~1 MB    | 171 MB |
| Q8_0     | 85 MB    | ~3 MB    | 88 MB  |
| Q4_0     | 43 MB    | ~3 MB    | 46 MB  |

Overhead 包括：
- Header: ~100 KB (config + vocab)
- Metadata: ~10 KB (100 tensors)
- Alignment padding: ~1 MB

## 转换工具

从 HuggingFace 格式转换为 .pllm：

```bash
python tools/export_to_pllm.py \
  --input models/pocket-0.1-pretrain \
  --output pocket-0.1.pllm \
  --quant q8_0
```

## 校验

文件末尾添加 CRC32 校验和（可选）：

```
uint32_t crc32;  // 整个文件的 CRC32
```

## 示例

一个完整的 .pllm 文件解析：

```cpp
// 打开文件
FILE* f = fopen("model.pllm", "rb");

// 1. 验证 magic
char magic[4];
fread(magic, 1, 4, f);
assert(memcmp(magic, "PLLM", 4) == 0);

// 2. 读取版本
uint32_t version;
fread(&version, 4, 1, f);
assert(version == 1);

// 3. 读取 header size
uint32_t header_size;
fread(&header_size, 4, 1, f);

// 4. 读取 config
uint32_t config_size;
fread(&config_size, 4, 1, f);
std::string config_json(config_size, '\0');
fread(&config_json[0], 1, config_size, f);

// 解析 JSON
ModelConfig config = parse_json(config_json);

// 5. 读取 vocab
uint32_t vocab_size;
fread(&vocab_size, 4, 1, f);
for (int i = 0; i < vocab_size; i++) {
    uint32_t token_len;
    fread(&token_len, 4, 1, f);
    std::string token(token_len, '\0');
    fread(&token[0], 1, token_len, f);
    vocab.push_back(token);
}

// 6. 读取 tensor metadata
uint32_t num_tensors;
fread(&num_tensors, 4, 1, f);
for (int i = 0; i < num_tensors; i++) {
    Tensor tensor;
    
    // name
    uint32_t name_len;
    fread(&name_len, 4, 1, f);
    tensor.name.resize(name_len);
    fread(&tensor.name[0], 1, name_len, f);
    
    // shape
    uint32_t ndim;
    fread(&ndim, 4, 1, f);
    tensor.shape.resize(ndim);
    fread(&tensor.shape[0], 8, ndim, f);
    
    // qtype
    uint8_t qtype;
    fread(&qtype, 1, 1, f);
    tensor.qtype = (QuantType)qtype;
    
    // offset & size
    fread(&tensor.offset, 8, 1, f);
    fread(&tensor.data_size, 8, 1, f);
    
    tensors.push_back(tensor);
}

// 7. 读取或 mmap weights
// 选项 A: 读取到内存
for (auto& tensor : tensors) {
    fseek(f, tensor.offset, SEEK_SET);
    tensor.data = malloc(tensor.data_size);
    fread(tensor.data, 1, tensor.data_size, f);
}

// 选项 B: mmap (更高效)
void* mapped = mmap(...);
for (auto& tensor : tensors) {
    tensor.data = (uint8_t*)mapped + tensor.offset;
}

fclose(f);
```

## 版本兼容性

- **Major version** 变化：不兼容
- **Minor version** 变化：向后兼容

版本号格式：`major.minor`

当前版本：`1.0`

## 未来扩展

可能的扩展：

1. **压缩** - 使用 zstd 压缩 weights
2. **加密** - AES 加密敏感模型
3. **分片** - 大模型分片存储
4. **多架构** - 同一文件支持多个量化版本

## 参考

- GGUF: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- SafeTensors: https://github.com/huggingface/safetensors
