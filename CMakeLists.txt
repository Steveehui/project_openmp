cmake_minimum_required(VERSION 3.18)
project(ImageConvolution LANGUAGES CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(MSVC)
    # MSVC: 添加 UTF-8 支持,解决中文编码问题
    add_compile_options(/W3 /O2 /openmp /utf-8)
else()
    add_compile_options(-Wall -O3 -fopenmp)
endif()

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 源文件
set(COMMON_SOURCES
    src/image_util.cpp
    src/convolution.cpp
)

# 串行版本
add_executable(conv_serial ${COMMON_SOURCES} src/main_serial.cpp)

# OpenMP 版本
add_executable(conv_omp ${COMMON_SOURCES} src/main_omp.cpp)
if(MSVC)
    target_compile_options(conv_omp PRIVATE /openmp)
else()
    target_compile_options(conv_omp PRIVATE -fopenmp)
    target_link_libraries(conv_omp PRIVATE -fopenmp)
endif()

# 检测 CUDA
# 尝试直接启用 CUDA 语言
if(DEFINED ENV{CUDA_PATH} OR EXISTS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    enable_language(CUDA)
endif()

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER OR DEFINED CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # 设置 CUDA 标准和架构
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 75 86)  # RTX 4070 是 Ada架构(8.9), 向下兼容 Turing(7.5)
    
    # CUDA 版本
    add_executable(conv_cuda 
        ${COMMON_SOURCES}
        src/convolution_cuda.cu
        src/main_cuda.cpp
    )
    set_target_properties(conv_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(conv_cuda PRIVATE USE_CUDA)
    
    # 混合版本
    add_executable(conv_hybrid
        ${COMMON_SOURCES}
        src/convolution_cuda.cu
        src/main_hybrid.cpp
    )
    set_target_properties(conv_hybrid PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(conv_hybrid PRIVATE USE_CUDA)
    
    if(MSVC)
        target_compile_options(conv_hybrid PRIVATE /openmp)
    else()
        target_compile_options(conv_hybrid PRIVATE -fopenmp)
        target_link_libraries(conv_hybrid PRIVATE -fopenmp)
    endif()
    
    message(STATUS "CUDA 支持已启用")
else()
    message(WARNING "未检测到 CUDA,跳过 CUDA 版本编译")
endif()

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ 编译器: ${CMAKE_CXX_COMPILER}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA 编译器: ${CMAKE_CUDA_COMPILER}")
endif()
