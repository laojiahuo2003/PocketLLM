/**
 * 简单的 JSON 解析工具
 *
 * 只支持基本的键值对解析，足够读取模型配置
 */

#include <string>
#include <sstream>
#include <unordered_map>

namespace pocket {
namespace utils {

/**
 * 简化的 JSON 解析器
 * 只解析简单的 key-value 对
 */
class SimpleJSON {
public:
    std::unordered_map<std::string, std::string> data;

    bool parse(const std::string& json) {
        // 非常简化的解析，只处理 {"key": value, ...} 格式
        // 实际项目应该使用成熟的 JSON 库如 nlohmann/json

        size_t pos = 0;
        while (pos < json.size()) {
            // 找到 key
            size_t key_start = json.find('"', pos);
            if (key_start == std::string::npos) break;

            size_t key_end = json.find('"', key_start + 1);
            if (key_end == std::string::npos) break;

            std::string key = json.substr(key_start + 1, key_end - key_start - 1);

            // 找到 value
            size_t colon = json.find(':', key_end);
            if (colon == std::string::npos) break;

            size_t value_start = json.find_first_not_of(" \t\n", colon + 1);
            if (value_start == std::string::npos) break;

            size_t value_end;
            if (json[value_start] == '"') {
                // 字符串值
                value_end = json.find('"', value_start + 1);
                if (value_end == std::string::npos) break;
                data[key] = json.substr(value_start + 1, value_end - value_start - 1);
                pos = value_end + 1;
            } else {
                // 数字或布尔值
                value_end = json.find_first_of(",}", value_start);
                if (value_end == std::string::npos) break;
                data[key] = json.substr(value_start, value_end - value_start);
                pos = value_end;
            }
        }

        return !data.empty();
    }

    int getInt(const std::string& key, int default_value = 0) const {
        auto it = data.find(key);
        if (it != data.end()) {
            return std::stoi(it->second);
        }
        return default_value;
    }

    float getFloat(const std::string& key, float default_value = 0.0f) const {
        auto it = data.find(key);
        if (it != data.end()) {
            return std::stof(it->second);
        }
        return default_value;
    }

    std::string getString(const std::string& key, const std::string& default_value = "") const {
        auto it = data.find(key);
        if (it != data.end()) {
            return it->second;
        }
        return default_value;
    }
};

} // namespace utils
} // namespace pocket
