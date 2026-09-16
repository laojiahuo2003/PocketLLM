/**
 * Generator 实现 - 文本生成
 */

#include "pocket.h"
#include <iostream>

namespace pocket {

std::vector<token_t> Generator::generate_tokens(
    const std::vector<token_t>& prompt_tokens,
    const GenerateConfig& config
) {
    std::vector<token_t> output_tokens = prompt_tokens;

    // 清空 KV cache
    if (config.use_cache) {
        kv_cache_.clear();
    }

    // 创建采样器
    SamplerConfig sampler_config;
    sampler_config.temperature = config.temperature;
    sampler_config.top_p = config.top_p;
    sampler_config.top_k = config.top_k;
    sampler_config.repetition_penalty = config.repetition_penalty;
    sampler_config.seed = config.seed;

    Sampler sampler(sampler_config);

    // 生成循环
    for (int i = 0; i < config.max_new_tokens; i++) {
        // 前向传播
        std::vector<float> logits = model_->forward(
            output_tokens,
            config.use_cache ? &kv_cache_ : nullptr
        );

        // 采样下一个 token
        token_t next_token = sampler.sample(logits);

        // 检查是否结束
        if (next_token == tokenizer_->eos_token()) {
            break;
        }

        // 添加到输出
        output_tokens.push_back(next_token);

        // 回调
        if (config.callback) {
            config.callback(next_token, config.callback_data);
        }
    }

    return output_tokens;
}

std::string Generator::generate(
    const std::string& prompt,
    const GenerateConfig& config
) {
    // 编码 prompt
    std::vector<token_t> prompt_tokens = tokenizer_->encode(prompt, true);

    // 生成 tokens
    std::vector<token_t> output_tokens = generate_tokens(prompt_tokens, config);

    // 解码输出（跳过 prompt）
    std::vector<token_t> new_tokens(
        output_tokens.begin() + prompt_tokens.size(),
        output_tokens.end()
    );

    return tokenizer_->decode(new_tokens, true);
}

} // namespace pocket
