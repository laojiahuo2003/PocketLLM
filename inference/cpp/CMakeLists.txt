cmake_minimum_required(VERSION 3.10)
project(PocketLLM_Inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
    if(ANDROID)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    endif()
endif()

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 源文件
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/ops/*.cpp"
    "src/backends/cpu/*.cpp"
)

# 如果是 ARM 平台，添加 NEON 后端
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" OR ANDROID)
    file(GLOB_RECURSE NEON_SOURCES "src/backends/neon/*.cpp")
    list(APPEND SOURCES ${NEON_SOURCES})
    add_definitions(-DENABLE_NEON)
endif()

# 静态库
add_library(pllm_inference STATIC ${SOURCES})

# 动态库
add_library(pllm_inference_shared SHARED ${SOURCES})
set_target_properties(pllm_inference_shared PROPERTIES OUTPUT_NAME pllm_inference)

# 测试可执行文件（可选）
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    file(GLOB TEST_SOURCES "tests/*.cpp")
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} pllm_inference)
    endforeach()
endif()

# 安装
install(TARGETS pllm_inference pllm_inference_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/pllm)

# Android NDK 支持
if(ANDROID)
    find_library(log-lib log)
    target_link_libraries(pllm_inference ${log-lib})
    target_link_libraries(pllm_inference_shared ${log-lib})
endif()
