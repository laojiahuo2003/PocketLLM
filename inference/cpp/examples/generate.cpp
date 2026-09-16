/**
 * 简单的文本生成程序
 *
 * 用法:
 *   ./pocket-generate model.pllm --prompt "你好"
 */

#include "pocket.h"
#include <iostream>
#include <cstring>

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model.pllm> --prompt <text>\n\n"
              << "Options:\n"
              << "  --prompt <text>      Input prompt\n"
              << "  --max-tokens <n>     Maximum tokens to generate (default: 50)\n"
              << "  --temperature <f>    Sampling temperature (default: 0.8)\n"
              << "  --top-p <f>          Top-p sampling (default: 0.9)\n"
              << "  --top-k <n>          Top-k sampling (default: 50)\n"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt;

    // 解析参数
    pocket::GenerateConfig config;
    config.max_new_tokens = 50;
    config.temperature = 0.8f;
    config.top_p = 0.9f;
    config.top_k = 50;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            config.max_new_tokens = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            config.temperature = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            config.top_p = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            config.top_k = std::atoi(argv[++i]);
        }
    }

    if (prompt.empty()) {
        std::cerr << "Error: --prompt is required" << std::endl;
        return 1;
    }

    std::cout << "Loading model from " << model_path << "..." << std::endl;

    // 加载模型
    pocket::Model model;
    if (!model.load(model_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    // 加载分词器
    pocket::Tokenizer tokenizer;
    if (!tokenizer.load(model_path)) {
        std::cerr << "Failed to load tokenizer!" << std::endl;
        return 1;
    }

    // 打印模型信息
    pocket::print_model_info(model);
    std::cout << std::endl;

    // 创建生成器
    pocket::Generator generator(&model, &tokenizer);

    // 生成
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Generating..." << std::endl;
    std::cout << std::endl;

    std::cout << "Output: " << std::flush;

    // 设置流式输出
    config.callback = [](pocket::token_t token, void* user_data) {
        pocket::Tokenizer* tok = (pocket::Tokenizer*)user_data;
        std::cout << tok->decode(token) << std::flush;
    };
    config.callback_data = &tokenizer;

    std::string response = generator.generate(prompt, config);

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Done!" << std::endl;

    return 0;
}
