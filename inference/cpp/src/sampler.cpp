/**
 * Sampler 实现 - 采样策略
 */

#include "pocket.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>

namespace pocket {

token_t Sampler::argmax(const std::vector<float>& logits) {
    return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

token_t Sampler::sample_top_k(std::vector<float> logits, int k) {
    // 1. 找到 top-k 个最大值
    std::vector<std::pair<float, token_t>> indexed_logits;
    for (size_t i = 0; i < logits.size(); i++) {
        indexed_logits.push_back({logits[i], static_cast<token_t>(i)});
    }

    // 部分排序，只保留 top-k
    std::partial_sort(indexed_logits.begin(),
                     indexed_logits.begin() + k,
                     indexed_logits.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    // 2. 对 top-k 做 softmax
    std::vector<float> probs(k);
    float max_logit = indexed_logits[0].first;
    float sum = 0.0f;

    for (int i = 0; i < k; i++) {
        probs[i] = expf(indexed_logits[i].first - max_logit);
        sum += probs[i];
    }

    for (int i = 0; i < k; i++) {
        probs[i] /= sum;
    }

    // 3. 采样
    std::random_device rd;
    std::mt19937 gen(config_.seed != 0 ? config_.seed : rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    int sampled_idx = dist(gen);
    return indexed_logits[sampled_idx].second;
}

token_t Sampler::sample_top_p(std::vector<float> logits, float p) {
    // 1. 排序
    std::vector<std::pair<float, token_t>> indexed_logits;
    for (size_t i = 0; i < logits.size(); i++) {
        indexed_logits.push_back({logits[i], static_cast<token_t>(i)});
    }

    std::sort(indexed_logits.begin(), indexed_logits.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });

    // 2. Softmax
    float max_logit = indexed_logits[0].first;
    std::vector<float> probs(indexed_logits.size());
    float sum = 0.0f;

    for (size_t i = 0; i < indexed_logits.size(); i++) {
        probs[i] = expf(indexed_logits[i].first - max_logit);
        sum += probs[i];
    }

    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum;
    }

    // 3. 找到累积概率超过 p 的位置
    float cumsum = 0.0f;
    size_t cutoff = 0;

    for (size_t i = 0; i < probs.size(); i++) {
        cumsum += probs[i];
        cutoff = i;
        if (cumsum >= p) {
            break;
        }
    }

    cutoff++; // 至少保留一个

    // 4. 重新归一化并采样
    std::vector<float> filtered_probs(probs.begin(), probs.begin() + cutoff);
    float filtered_sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);

    for (auto& prob : filtered_probs) {
        prob /= filtered_sum;
    }

    std::random_device rd;
    std::mt19937 gen(config_.seed != 0 ? config_.seed : rd());
    std::discrete_distribution<> dist(filtered_probs.begin(), filtered_probs.end());

    int sampled_idx = dist(gen);
    return indexed_logits[sampled_idx].second;
}

token_t Sampler::sample(const std::vector<float>& logits) {
    // 应用 temperature
    std::vector<float> scaled_logits = logits;
    if (config_.temperature != 1.0f) {
        for (auto& logit : scaled_logits) {
            logit /= config_.temperature;
        }
    }

    // TODO: 应用 repetition penalty

    // 根据配置选择采样策略
    if (config_.temperature == 0.0f) {
        // Greedy
        return argmax(scaled_logits);
    } else if (config_.top_k > 0) {
        // Top-K
        return sample_top_k(scaled_logits, config_.top_k);
    } else if (config_.top_p < 1.0f) {
        // Top-P
        return sample_top_p(scaled_logits, config_.top_p);
    } else {
        // 普通采样
        return sample_top_p(scaled_logits, 1.0f);
    }
}

} // namespace pocket
