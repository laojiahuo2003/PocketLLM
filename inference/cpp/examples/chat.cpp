/**
 * Pocket-0.1 命令行对话程序
 *
 * 用法:
 *   ./pocket-chat model.pllm
 */

#include "pocket.h"
#include <iostream>
#include <string>
#include <cstring>

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model.pllm>\n\n"
              << "Options:\n"
              << "  --max-tokens <n>     Maximum tokens to generate (default: 100)\n"
              << "  --temperature <f>    Sampling temperature (default: 0.8)\n"
              << "  --top-p <f>          Top-p sampling (default: 0.9)\n"
              << "  --top-k <n>          Top-k sampling (default: 50)\n"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];

    // 解析参数
    pocket::GenerateConfig config;
    config.max_new_tokens = 100;
    config.temperature = 0.8f;
    config.top_p = 0.9f;
    config.top_k = 50;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            config.max_new_tokens = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            config.temperature = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            config.top_p = std::atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            config.top_k = std::atoi(argv[++i]);
        }
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

    // 创建生成器
    pocket::Generator generator(&model, &tokenizer);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "🤖 Pocket-0.1 Chat" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Type 'quit' or 'exit' to exit" << std::endl;
    std::cout << "Type 'clear' to clear conversation history" << std::endl;
    std::cout << std::endl;

    std::string conversation_history;

    while (true) {
        // 获取用户输入
        std::cout << "👤 You: ";
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            break;  // EOF/输入结束
        }

        if (user_input.empty()) {
            continue;
        }

        if (user_input == "quit" || user_input == "exit") {
            std::cout << "Goodbye!" << std::endl;
            break;
        }

        if (user_input == "clear") {
            conversation_history.clear();
            std::cout << "✅ Conversation history cleared" << std::endl;
            continue;
        }

        // 构建 prompt
        std::string prompt;
        if (!conversation_history.empty()) {
            prompt = conversation_history + "\nUser: " + user_input + "\nAssistant:";
        } else {
            prompt = "User: " + user_input + "\nAssistant:";
        }

        // 生成回复
        std::cout << "🤖 Assistant: " << std::flush;

        // 设置流式输出回调
        config.callback = [](pocket::token_t token, void* user_data) {
            pocket::Tokenizer* tok = (pocket::Tokenizer*)user_data;
            std::cout << tok->decode(token) << std::flush;
        };
        config.callback_data = &tokenizer;

        std::string response = generator.generate(prompt, config);

        std::cout << std::endl << std::endl;

        // 更新历史
        conversation_history += "\nUser: " + user_input;
        conversation_history += "\nAssistant: " + response;

        // 限制历史长度（保留最近 4 轮对话）
        size_t max_history_length = 1000;
        if (conversation_history.size() > max_history_length) {
            size_t pos = conversation_history.find("\nUser:",
                conversation_history.size() - max_history_length);
            if (pos != std::string::npos) {
                conversation_history = conversation_history.substr(pos);
            }
        }
    }

    return 0;
}
