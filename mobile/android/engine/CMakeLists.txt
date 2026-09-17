cmake_minimum_required(VERSION 3.22)
project(PocketAndroid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 引擎源码根目录（相对于本 CMakeLists）
set(ENGINE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../inference/cpp")
set(ENGINE_INC_DIR "${ENGINE_SRC_DIR}/include")

# arm64 禁用 -march=native（NDK 交叉编译下非法），显式开启 NEON 宏
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_definitions(POCKET_ARM_NEON)
endif()

add_compile_options(-O2 -Wall -Wextra -fvisibility=hidden)

# 引擎 + JNI 桥接统一编成一个 libpocket.so
add_library(pocket_engine SHARED
    jni_bridge.cpp
    ${ENGINE_SRC_DIR}/src/tensor.cpp
    ${ENGINE_SRC_DIR}/src/model.cpp
    ${ENGINE_SRC_DIR}/src/tokenizer.cpp
    ${ENGINE_SRC_DIR}/src/sampler.cpp
    ${ENGINE_SRC_DIR}/src/generator.cpp
    ${ENGINE_SRC_DIR}/src/ops/matmul.cpp
    ${ENGINE_SRC_DIR}/src/ops/norm.cpp
    ${ENGINE_SRC_DIR}/src/ops/rope.cpp
    ${ENGINE_SRC_DIR}/src/ops/attention.cpp
    ${ENGINE_SRC_DIR}/src/ops/ffn.cpp
    ${ENGINE_SRC_DIR}/src/utils/file.cpp
    ${ENGINE_SRC_DIR}/src/utils/json.cpp
)

target_include_directories(pocket_engine PRIVATE ${ENGINE_INC_DIR})

# Android 原生库依赖 log 库
find_library(ANDROID_LOG_LIB log)
target_link_libraries(pocket_engine ${ANDROID_LOG_LIB})

# 输出命名统一为 libpocket.so
set_target_properties(pocket_engine PROPERTIES
    OUTPUT_NAME "pocket"
)