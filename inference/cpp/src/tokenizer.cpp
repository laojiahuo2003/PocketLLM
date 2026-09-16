/**
 * Tokenizer 实现 - BPE 分词器
 */

#include "pocket.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

namespace pocket {

bool Tokenizer::load(const std::string& path) {
    // 从 .pllm 文件中加载词表
    // 这里简化实现，实际应该从文件中读取

    // TODO: 完整实现
    // 暂时使用占位符
    vocab_.resize(6400);
    for (size_t i = 0; i < vocab_.size(); i++) {
        vocab_[i] = "<token_" + std::to_string(i) + ">";
    }

    bos_token_ = 1;
    eos_token_ = 2;
    pad_token_ = 0;

    return true;
}

std::vector<token_t> Tokenizer::encode(const std::string& text, bool add_special_tokens) {
    std::vector<token_t> tokens;

    if (add_special_tokens) {
        tokens.push_back(bos_token_);
    }

    // 简化版 BPE 编码
    // TODO: 完整的 BPE 算法实现

    // 这里先用简单的字符级编码
    for (char c : text) {
        // 将字符映射到 token
        token_t token_id = static_cast<token_t>(c) % vocab_.size();
        tokens.push_back(token_id);
    }

    if (add_special_tokens) {
        tokens.push_back(eos_token_);
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<token_t>& tokens, bool skip_special_tokens) {
    std::string text;

    for (token_t token_id : tokens) {
        if (skip_special_tokens) {
            if (token_id == bos_token_ || token_id == eos_token_ || token_id == pad_token_) {
                continue;
            }
        }

        text += decode(token_id);
    }

    return text;
}

std::string Tokenizer::decode(token_t token) {
    if (token >= 0 && token < static_cast<token_t>(vocab_.size())) {
        return vocab_[token];
    }
    return "<unk>";
}

} // namespace pocket
