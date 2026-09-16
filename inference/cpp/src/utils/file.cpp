/**
 * 文件读写工具
 */

#include "pocket.h"
#include <fstream>
#include <vector>

namespace pocket {
namespace utils {

/**
 * 读取整个文件到内存
 */
std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);

    return buffer;
}

/**
 * 写入文件
 */
bool write_file(const std::string& path, const void* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(reinterpret_cast<const char*>(data), size);
    return file.good();
}

} // namespace utils
} // namespace pocket
