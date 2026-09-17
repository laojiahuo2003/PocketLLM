/**
 * Tokenizer 实现 - 从 .pllm 加载真实词表
 *
 * 从 .pllm 文件的 header（config_json + vocab_section）解析词表和特殊 token。
 * 编码采用对词表的贪心最长匹配（逐 token 在剩余文本上找最长前缀）。
 * 该方案确定性强、无需 BPE merge 规则，可跑通端到端流程（smoke 用途）。
 */

#include "pocket.h"
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <iostream>

namespace pocket {

namespace {

std::string json_extract(const std::string& json, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return "";
    size_t colon = json.find(':', pos + pat.size());
    if (colon == std::string::npos) return "";
    size_t start = colon + 1;
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) start++;
    if (start < json.size() && (json[start] == '-' || json[start] == '.' || std::isdigit((unsigned char)json[start]))) {
        size_t end = start;
        while (end < json.size()) {
            char c = json[end];
            if (c == ',' || c == '}' || c == ']') break;
            end++;
        }
        return json.substr(start, end - start);
    }
    return json.substr(start, std::string::npos);
}

long long json_int(const std::string& json, const std::string& key, long long dflt) {
    std::string v = json_extract(json, key);
    if (v.empty() || v == "null") return dflt;
    try { return std::stoll(v); } catch (...) { return dflt; }
}

// UTF-8 字符的字节数（按首字节）
size_t utf8_char_len(const char* s) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

} // namespace

bool Tokenizer::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    // 跳过 magic(4) + version(4) + header_size(4)
    file.seekg(4 + 4 + 4);

    // config
    uint32_t config_size;
    file.read(reinterpret_cast<char*>(&config_size), sizeof(config_size));
    std::string config_json(config_size, '\0');
    file.read(&config_json[0], config_size);

    bos_token_ = (token_t)json_int(config_json, "bos_token_id", 1);
    eos_token_ = (token_t)json_int(config_json, "eos_token_id", 2);
    pad_token_ = (token_t)json_int(config_json, "pad_token_id", 0);

    // vocab
    uint32_t vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    vocab_.resize(vocab_size);
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint32_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string token(len, '\0');
        file.read(&token[0], len);
        vocab_[i] = std::move(token);
    }
    file.close();

    std::cout << "[Tokenizer] 加载词表: " << vocab_.size() << " tokens"
              << " (bos=" << bos_token_ << ", eos=" << eos_token_ << ")" << std::endl;
    return true;
}

std::vector<token_t> Tokenizer::encode(const std::string& text, bool add_special_tokens) {
    std::vector<token_t> tokens;
    if (tokens_desc_.empty()) {
        // 预构建：token->id 映射 + 按长度降序排列的索引
        for (size_t i = 0; i < vocab_.size(); i++) {
            if (!vocab_[i].empty()) {
                id_map_[vocab_[i]] = (token_t)i;
                tokens_desc_.push_back({vocab_[i], (token_t)i});
            }
        }
        std::sort(tokens_desc_.begin(), tokens_desc_.end(),
                  [](const std::pair<std::string, token_t>& a, const std::pair<std::string, token_t>& b) {
                      if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
                      return a.first < b.first;
                  });
    }

    if (add_special_tokens) {
        tokens.push_back(bos_token_);
    }

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        bool matched = false;
        for (const auto& kv : tokens_desc_) {
            const std::string& tok = kv.first;
            if (i + tok.size() <= n && text.compare(i, tok.size(), tok) == 0) {
                tokens.push_back(kv.second);
                i += tok.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            // 回退：按单个 UTF-8 字符匹配，找不到则用 unk/bos
            size_t clen = utf8_char_len(text.data() + i);
            if (i + clen > n) clen = 1;
            std::string ch = text.substr(i, clen);
            auto it = id_map_.find(ch);
            if (it != id_map_.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(bos_token_); // 未知字符兜底
            }
            i += clen;
        }
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<token_t>& tokens, bool skip_special_tokens) {
    std::string text;
    for (token_t id : tokens) {
        if (skip_special_tokens) {
            if (id == bos_token_ || id == eos_token_ || id == pad_token_) continue;
        }
        if (id >= 0 && id < (token_t)vocab_.size()) text += vocab_[(size_t)id];
    }
    return text;
}

std::string Tokenizer::decode(token_t token) {
    if (token >= 0 && token < (token_t)vocab_.size()) return vocab_[(size_t)token];
    return "<unk>";
}

} // namespace pocket