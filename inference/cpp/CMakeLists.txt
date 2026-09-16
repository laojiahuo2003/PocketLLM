cmake_minimum_required(VERSION 3.10)
project(PocketInference VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 选项
option(POCKET_BUILD_SHARED "Build shared library" ON)
option(POCKET_BUILD_EXAMPLES "Build examples" ON)
option(POCKET_BUILD_TESTS "Build tests" OFF)
option(POCKET_ARM_NEON "Enable ARM NEON optimizations" ON)

# 编译选项
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    if(POCKET_ARM_NEON)
        add_compile_options(-mfpu=neon)
        add_definitions(-DPOCKET_ARM_NEON)
    endif()
endif()

# 优化选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native)
endif()

# 警告选项
add_compile_options(-Wall -Wextra -Wpedantic)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 源文件
set(POCKET_SOURCES
    src/tensor.cpp
    src/model.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/generator.cpp
    src/ops/matmul.cpp
    src/ops/norm.cpp
    src/ops/rope.cpp
    src/ops/attention.cpp
    src/ops/ffn.cpp
    src/utils/file.cpp
    src/utils/json.cpp
)

# 头文件
set(POCKET_HEADERS
    include/pocket.h
)

# 库目标
if(POCKET_BUILD_SHARED)
    add_library(pocket SHARED ${POCKET_SOURCES})
else()
    add_library(pocket STATIC ${POCKET_SOURCES})
endif()

target_include_directories(pocket PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 链接库
target_link_libraries(pocket PRIVATE pthread)

# 示例程序
if(POCKET_BUILD_EXAMPLES)
    add_executable(pocket-chat examples/chat.cpp)
    target_link_libraries(pocket-chat pocket)

    add_executable(pocket-generate examples/generate.cpp)
    target_link_libraries(pocket-generate pocket)
endif()

# 测试
if(POCKET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 安装
install(TARGETS pocket
    EXPORT PocketTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${POCKET_HEADERS}
    DESTINATION include
)

# 导出配置
install(EXPORT PocketTargets
    FILE PocketTargets.cmake
    NAMESPACE Pocket::
    DESTINATION lib/cmake/Pocket
)

# 生成版本信息
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pocket_version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/pocket_version.h
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/pocket_version.h
    DESTINATION include
)
