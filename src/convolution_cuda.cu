/**
 * @file convolution_cuda_simple.cu
 * @brief CUDA 加速卷积实现 - 简化优化版
 */

#include "convolution.h"
#include "timer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <cstring>
#include <atomic>
#include <thread>
#include <algorithm>
#include <numeric>

// 常量内存卷积核（支持至多 32x32）
__constant__ float d_const_kernel[1024];

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return Image(); \
        } \
    } while(0)

#define CUDA_CHECK_VEC(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return {}; \
        } \
    } while(0)

// 线程块大小
#define BLOCK_SIZE 16
#define BLOCK_SIZE_SMALL 8

/**
 * @brief CUDA 卷积核 - 朴素实现
 */
__global__ void convolve_kernel_naive(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_width,
    int input_height,
    int kernel_size,
    int output_width,
    int output_height
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x >= output_width || out_y >= output_height) return;
    
    float sum = 0.0f;
    
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_x = out_x + kx;
            int in_y = out_y + ky;
            sum += input[in_y * input_width + in_x] * kernel[ky * kernel_size + kx];
        }
    }
    
    output[out_y * output_width + out_x] = fmaxf(0.0f, fminf(255.0f, sum));
}

/**
 * @brief CUDA 卷积核 - 共享内存优化
 */
__global__ void convolve_kernel_shared(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_width,
    int input_height,
    int kernel_size,
    int output_width,
    int output_height
) {
    // 动态共享内存
    extern __shared__ float shared_mem[];
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 共享内存布局: 前面存储卷积核，后面存储输入块
    float* shared_kernel = shared_mem;
    int shared_tile_dim = BLOCK_SIZE + kernel_size - 1;
    float* shared_input = shared_mem + kernel_size * kernel_size;
    
    // 加载卷积核到共享内存
    int kernel_elems = kernel_size * kernel_size;
    int tid = ty * BLOCK_SIZE + tx;
    int total_threads = BLOCK_SIZE * BLOCK_SIZE;
    for (int i = tid; i < kernel_elems; i += total_threads) {
        shared_kernel[i] = kernel[i];
    }
    
    // 加载输入块到共享内存
    int tiles_to_load = (shared_tile_dim * shared_tile_dim + total_threads - 1) / total_threads;
    for (int t = 0; t < tiles_to_load; ++t) {
        int idx = tid + t * total_threads;
        if (idx < shared_tile_dim * shared_tile_dim) {
            int sy = idx / shared_tile_dim;
            int sx = idx % shared_tile_dim;
            int in_x = blockIdx.x * BLOCK_SIZE + sx;
            int in_y = blockIdx.y * BLOCK_SIZE + sy;
            
            if (in_x < input_width && in_y < input_height) {
                shared_input[sy * shared_tile_dim + sx] = input[in_y * input_width + in_x];
            } else {
                shared_input[sy * shared_tile_dim + sx] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (out_x >= output_width || out_y >= output_height) return;
    
    // 计算卷积
    float sum = 0.0f;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            sum += shared_input[(ty + ky) * shared_tile_dim + (tx + kx)] * 
                   shared_kernel[ky * kernel_size + kx];
        }
    }
    
    output[out_y * output_width + out_x] = fmaxf(0.0f, fminf(255.0f, sum));
}

// BLOCK_SIZE_SMALL 版本，用于大核/共享内存紧张场景
__global__ void convolve_kernel_shared_small(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_width,
    int input_height,
    int kernel_size,
    int output_width,
    int output_height
) {
    extern __shared__ float shared_mem[];
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float* shared_kernel = shared_mem;
    int shared_tile_dim = BLOCK_SIZE_SMALL + kernel_size - 1;
    float* shared_input = shared_mem + kernel_size * kernel_size;

    int kernel_elems = kernel_size * kernel_size;
    int tid = ty * BLOCK_SIZE_SMALL + tx;
    int total_threads = BLOCK_SIZE_SMALL * BLOCK_SIZE_SMALL;
    for (int i = tid; i < kernel_elems; i += total_threads) {
        shared_kernel[i] = kernel[i];
    }

    int tiles_to_load = (shared_tile_dim * shared_tile_dim + total_threads - 1) / total_threads;
    for (int t = 0; t < tiles_to_load; ++t) {
        int idx = tid + t * total_threads;
        if (idx < shared_tile_dim * shared_tile_dim) {
            int sy = idx / shared_tile_dim;
            int sx = idx % shared_tile_dim;
            int in_x = blockIdx.x * BLOCK_SIZE_SMALL + sx;
            int in_y = blockIdx.y * BLOCK_SIZE_SMALL + sy;
            if (in_x < input_width && in_y < input_height) {
                shared_input[sy * shared_tile_dim + sx] = input[in_y * input_width + in_x];
            } else {
                shared_input[sy * shared_tile_dim + sx] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (out_x >= output_width || out_y >= output_height) return;

    float sum = 0.0f;
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            sum += shared_input[(ty + ky) * shared_tile_dim + (tx + kx)] *
                   shared_kernel[ky * kernel_size + kx];
        }
    }

    output[out_y * output_width + out_x] = fmaxf(0.0f, fminf(255.0f, sum));
}

/**
 * @brief CUDA 卷积核 - 常量内存小核优化
 * 适用于 kernel_size^2 <= 1024 的场景
 */
__global__ void convolve_kernel_const(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_width,
    int input_height,
    int kernel_size,
    int output_width,
    int output_height
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= output_width || out_y >= output_height) return;

    float sum = 0.0f;
    for (int ky = 0; ky < kernel_size; ++ky) {
        const float* in_row = input + (out_y + ky) * input_width + out_x;
        const float* k_row = d_const_kernel + ky * kernel_size;
#pragma unroll
        for (int kx = 0; kx < kernel_size; ++kx) {
            sum += in_row[kx] * k_row[kx];
        }
    }
    output[out_y * output_width + out_x] = fmaxf(0.0f, fminf(255.0f, sum));
}

namespace Convolution {

// === Helper: 简单稀疏度检测（占位版，未来可接 block-sparse） ===
static float estimate_sparsity(const Kernel& k) {
    int n = k.size * k.size;
    if (n == 0) return 0.0f;
    int zero_like = 0;
    for (float v : k.data) {
        if (fabsf(v) < 1e-6f) zero_like++;
    }
    return static_cast<float>(zero_like) / n;
}

// === Helper: 选择共享内存核 block 尺寸并 launch ===
static void launch_shared_kernel(
    const float* d_input,
    const float* d_kernel,
    float* d_output,
    int input_width,
    int input_height,
    int kernel_size,
    int output_width,
    int output_height,
    cudaStream_t stream = 0
) {
    // 估算共享内存需求，若超 48KB 或 kernel 较大，改用 8x8 block 减少 SMEM
    auto run_block = [&](int block_dim) {
        int tile_dim = block_dim + kernel_size - 1;
        size_t smem = (kernel_size * kernel_size + tile_dim * tile_dim) * sizeof(float);
        dim3 block(block_dim, block_dim);
        dim3 grid((output_width + block_dim - 1) / block_dim,
                  (output_height + block_dim - 1) / block_dim);
        if (block_dim == BLOCK_SIZE) {
            convolve_kernel_shared<<<grid, block, smem, stream>>>(
                d_input, d_kernel, d_output,
                input_width, input_height,
                kernel_size,
                output_width, output_height
            );
        } else {
            convolve_kernel_shared_small<<<grid, block, smem, stream>>>(
                d_input, d_kernel, d_output,
                input_width, input_height,
                kernel_size,
                output_width, output_height
            );
        }
    };

    int tile_dim_16 = BLOCK_SIZE + kernel_size - 1;
    size_t smem_16 = (kernel_size * kernel_size + tile_dim_16 * tile_dim_16) * sizeof(float);
    if (smem_16 > 48 * 1024 || kernel_size >= 17) {
        run_block(BLOCK_SIZE_SMALL);
    } else {
        run_block(BLOCK_SIZE);
    }
}

// === 卷积策略枚举（未来可扩展为 RL/代价模型） ===
enum class ConvPolicy {
    Auto,
    ConstSmall,   // 小核常量内存（Winograd/TF32 小核可扩展）
    Shared,       // 共享内存 tiling
    FFTLike,      // 占位：大核/大图 FFT 路径
    SparseBlock   // 占位：块稀疏
};

/**
 * @brief CUDA 卷积 - 朴素实现
 */
Image convolve_cuda(const Image& input, const Kernel& kernel) {
    int kernel_size = kernel.size;
    int output_width = input.width - kernel_size + 1;
    int output_height = input.height - kernel_size + 1;
    
    if (output_width <= 0 || output_height <= 0) {
        return Image();
    }
    
    Image output(output_width, output_height);
    
    // 分配设备内存
    float *d_input, *d_kernel, *d_output;
    size_t input_size = input.width * input.height * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);
    size_t output_size = output_width * output_height * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, input.data.data(), input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data.data(), kernel_size_bytes, cudaMemcpyHostToDevice));
    
    // 配置执行参数
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 启动内核
    convolve_kernel_naive<<<grid, block>>>(
        d_input, d_kernel, d_output,
        input.width, input.height,
        kernel_size,
        output_width, output_height
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_output, output_size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return output;
}

/**
 * @brief CUDA 卷积 - 常量内存优化（小核）
 * kernel_size^2 必须 <= 1024，否则回退到共享内存实现。
 */
Image convolve_cuda_const(const Image& input, const Kernel& kernel) {
    int kernel_size = kernel.size;
    int output_width = input.width - kernel_size + 1;
    int output_height = input.height - kernel_size + 1;

    if (output_width <= 0 || output_height <= 0) {
        return Image();
    }

    // 若核过大，直接回退共享内存版本
    if (kernel_size * kernel_size > 1024) {
        return convolve_cuda_shared(input, kernel);
    }

    Image output(output_width, output_height);

    float *d_input = nullptr, *d_output = nullptr;
    size_t input_size = input.width * input.height * sizeof(float);
    size_t output_size = output_width * output_height * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    CUDA_CHECK(cudaMemcpy(d_input, input.data.data(), input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel, kernel.data.data(), kernel_size * kernel_size * sizeof(float), 0, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolve_kernel_const<<<grid, block>>>(
        d_input, d_output,
        input.width, input.height,
        kernel_size,
        output_width, output_height
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data.data(), d_output, output_size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

/**
 * @brief CUDA 卷积 - 共享内存优化
 */
Image convolve_cuda_shared(const Image& input, const Kernel& kernel) {
    int kernel_size = kernel.size;
    int output_width = input.width - kernel_size + 1;
    int output_height = input.height - kernel_size + 1;
    
    if (output_width <= 0 || output_height <= 0) {
        return Image();
    }
    
    Image output(output_width, output_height);
    
    // 分配设备内存
    float *d_input, *d_kernel, *d_output;
    size_t input_size = input.width * input.height * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);
    size_t output_size = output_width * output_height * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    
    // 复制数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, input.data.data(), input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data.data(), kernel_size_bytes, cudaMemcpyHostToDevice));
    
    // 启动自适应 block 尺寸的共享内存核
    launch_shared_kernel(
        d_input, d_kernel, d_output,
        input.width, input.height,
        kernel_size,
        output_width, output_height,
        0
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回主机
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_output, output_size, cudaMemcpyDeviceToHost));
    
    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return output;
}

/**
 * @brief 自适应 GPU 卷积策略调度
 * - 小核(<=7)优先常量内存 (可扩展 Winograd)
 * - 大核(>=31 或 大图>=4K) 占位 FFT 分支（当前回退共享内存）
 * - 稀疏核(>60% 近零) 占位 block-sparse（当前回退共享内存）
 */
Image convolve_cuda_policy(const Image& input, const Kernel& kernel, ConvPolicy policy) {
    int ks = kernel.size;
    bool large_image = (input.width >= 4096 || input.height >= 4096);
    float sparsity = estimate_sparsity(kernel);

    auto use_const = (ks * ks <= 1024);
    auto use_fft_like = (ks >= 31) || (large_image && ks >= 17);
    auto use_sparse = (sparsity > 0.6f);

    if (policy == ConvPolicy::ConstSmall || (policy == ConvPolicy::Auto && use_const && ks <= 7)) {
        return convolve_cuda_const(input, kernel);
    }

    if (policy == ConvPolicy::SparseBlock || (policy == ConvPolicy::Auto && use_sparse)) {
        // 占位：未来接入 block-sparse；当前回退共享内存实现
        return convolve_cuda_shared(input, kernel);
    }

    if (policy == ConvPolicy::FFTLike || (policy == ConvPolicy::Auto && use_fft_like)) {
        // 占位：未来接入 FFT/Winograd 大核路径；当前回退共享内存
        return convolve_cuda_shared(input, kernel);
    }

    // 默认共享内存
    return convolve_cuda_shared(input, kernel);
}

/**
 * @brief CUDA 批量卷积 - 减少内存分配开销
 */
std::vector<Image> convolve_batch_cuda(
    const std::vector<Image>& images,
    const std::vector<Kernel>& kernels
) {
    if (images.empty() || kernels.empty()) {
        return {};
    }
    
    std::vector<Image> results;
    results.reserve(images.size() * kernels.size());
    
    // 假设所有图像尺寸相同
    int img_width = images[0].width;
    int img_height = images[0].height;
    int max_kernel_size = 0;
    for (const auto& k : kernels) {
        max_kernel_size = std::max(max_kernel_size, k.size);
    }
    
    int min_output_width = img_width - max_kernel_size + 1;
    int min_output_height = img_height - max_kernel_size + 1;
    
    if (min_output_width <= 0 || min_output_height <= 0) {
        return {};
    }
    
    // 预分配设备内存
    float *d_input, *d_kernel, *d_output;
    size_t input_size = img_width * img_height * sizeof(float);
    size_t max_kernel_bytes = max_kernel_size * max_kernel_size * sizeof(float);
    size_t max_output_size = img_width * img_height * sizeof(float);  // 保守估计
    
    CUDA_CHECK_VEC(cudaMalloc(&d_input, input_size));
    CUDA_CHECK_VEC(cudaMalloc(&d_kernel, max_kernel_bytes));
    CUDA_CHECK_VEC(cudaMalloc(&d_output, max_output_size));
    
    // 批量处理
    for (const auto& img : images) {
        // 复制图像到设备
        CUDA_CHECK_VEC(cudaMemcpy(d_input, img.data.data(), input_size, cudaMemcpyHostToDevice));
        
        for (const auto& kernel : kernels) {
            int kernel_size = kernel.size;
            int output_width = img_width - kernel_size + 1;
            int output_height = img_height - kernel_size + 1;
            
            Image output(output_width, output_height);
            
            // 复制卷积核
            size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
            CUDA_CHECK_VEC(cudaMemcpy(d_kernel, kernel.data.data(), kernel_bytes, cudaMemcpyHostToDevice));
            
            // 配置执行参数
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
            
            // 启动内核
            convolve_kernel_naive<<<grid, block>>>(
                d_input, d_kernel, d_output,
                img_width, img_height,
                kernel_size,
                output_width, output_height
            );
            
            CUDA_CHECK_VEC(cudaGetLastError());
            CUDA_CHECK_VEC(cudaDeviceSynchronize());
            
            // 复制结果
            size_t output_bytes = output_width * output_height * sizeof(float);
            CUDA_CHECK_VEC(cudaMemcpy(output.data.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
            
            results.push_back(std::move(output));
        }
    }
    
    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return results;
}

/**
 * @brief CUDA 批量卷积 - 共享内存优化版本
 */
std::vector<Image> convolve_batch_cuda_shared(
    const std::vector<Image>& images,
    const std::vector<Kernel>& kernels
) {
    if (images.empty() || kernels.empty()) {
        return {};
    }
    
    std::vector<Image> results;
    results.reserve(images.size() * kernels.size());
    
    int img_width = images[0].width;
    int img_height = images[0].height;
    int max_kernel_size = 0;
    for (const auto& k : kernels) {
        max_kernel_size = std::max(max_kernel_size, k.size);
    }
    
    // 预分配设备内存
    float *d_input, *d_kernel, *d_output;
    size_t input_size = img_width * img_height * sizeof(float);
    size_t max_kernel_bytes = max_kernel_size * max_kernel_size * sizeof(float);
    size_t max_output_size = img_width * img_height * sizeof(float);
    
    CUDA_CHECK_VEC(cudaMalloc(&d_input, input_size));
    CUDA_CHECK_VEC(cudaMalloc(&d_kernel, max_kernel_bytes));
    CUDA_CHECK_VEC(cudaMalloc(&d_output, max_output_size));
    
    // 批量处理 - 图像复用
    for (const auto& img : images) {
        // 每张图像只传输一次
        CUDA_CHECK_VEC(cudaMemcpy(d_input, img.data.data(), input_size, cudaMemcpyHostToDevice));
        
        for (const auto& kernel : kernels) {
            int kernel_size = kernel.size;
            int output_width = img_width - kernel_size + 1;
            int output_height = img_height - kernel_size + 1;
            
            Image output(output_width, output_height);
            
            // 复制卷积核
            size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
            CUDA_CHECK_VEC(cudaMemcpy(d_kernel, kernel.data.data(), kernel_bytes, cudaMemcpyHostToDevice));
            
            // 自适应 block 尺寸的共享内存核
            launch_shared_kernel(
                d_input, d_kernel, d_output,
                img_width, img_height,
                kernel_size,
                output_width, output_height,
                0
            );
            
            CUDA_CHECK_VEC(cudaGetLastError());
            CUDA_CHECK_VEC(cudaDeviceSynchronize());
            
            // 复制结果
            size_t output_bytes = output_width * output_height * sizeof(float);
            CUDA_CHECK_VEC(cudaMemcpy(output.data.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
            
            results.push_back(std::move(output));
        }
    }
    
    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return results;
}

/**
 * @brief CUDA 批量卷积 - 流水线融合优化 (Pipelined Fusion)
 * 
 * ============== 核心优化策略 ==============
 * 
 * 1. 全异步流水线: 所有 stream 独立工作，无同步等待
 * 2. 双缓冲输出: 每个 stream 有两组输出缓冲，交替使用
 * 3. 延迟回收: 计算完成后异步传输，下一次 launch 前才同步
 * 4. 完全重叠: 传输与计算完全重叠
 * 
 * 时间线示例 (4 streams):
 * Stream0: [K0计算][传输K0][K4计算][传输K4]...
 * Stream1: [K1计算][传输K1][K5计算][传输K5]...
 * Stream2: [K2计算][传输K2][K6计算][传输K6]...
 * Stream3: [K3计算][传输K3][K7计算][传输K7]...
 */
std::vector<Image> convolve_batch_cuda_streams(
    const std::vector<Image>& images,
    const std::vector<Kernel>& kernels,
    int num_streams
) {
    if (images.empty() || kernels.empty()) {
        return {};
    }
    
    // 检查 GPU 状态
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[convolve_batch_cuda_streams] CUDA error at start: " 
                  << cudaGetErrorString(err) << "\n";
        cudaDeviceReset();
    }
    
    // 优化 stream 数量: 8-16 个效果最好
    if (num_streams <= 0) num_streams = 12;
    num_streams = std::min(num_streams, 16);
    
    int img_width = images[0].width;
    int img_height = images[0].height;
    int num_kernels = (int)kernels.size();
    int num_images = (int)images.size();
    size_t total_tasks = num_images * num_kernels;
    
    std::vector<Image> results(total_tasks);
    
    // 找最大卷积核尺寸
    int max_kernel_size = 0;
    for (const auto& k : kernels) {
        max_kernel_size = std::max(max_kernel_size, k.size);
    }
    
    size_t input_size = img_width * img_height * sizeof(float);
    size_t max_output_size = img_width * img_height * sizeof(float);
    size_t max_kernel_bytes = max_kernel_size * max_kernel_size * sizeof(float);
    
    // ========== 资源分配 ==========
    
    // 创建高优先级 streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK_VEC(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }
    
    // 图像缓冲 (每张图像一份)
    std::vector<float*> d_inputs(num_images);
    for (int i = 0; i < num_images; ++i) {
        CUDA_CHECK_VEC(cudaMalloc(&d_inputs[i], input_size));
    }
    
    // 每个 stream 的资源: 卷积核 + 双缓冲输出
    std::vector<float*> d_kernels(num_streams);
    std::vector<float*> d_outputs_A(num_streams);  // 双缓冲 A
    std::vector<float*> d_outputs_B(num_streams);  // 双缓冲 B
    std::vector<float*> h_outputs_A(num_streams);  // Pinned 内存 A
    std::vector<float*> h_outputs_B(num_streams);  // Pinned 内存 B
    
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK_VEC(cudaMalloc(&d_kernels[i], max_kernel_bytes));
        CUDA_CHECK_VEC(cudaMalloc(&d_outputs_A[i], max_output_size));
        CUDA_CHECK_VEC(cudaMalloc(&d_outputs_B[i], max_output_size));
        CUDA_CHECK_VEC(cudaMallocHost(&h_outputs_A[i], max_output_size));
        CUDA_CHECK_VEC(cudaMallocHost(&h_outputs_B[i], max_output_size));
    }
    
    // ========== 预传输所有图像到 GPU ==========
    for (int i = 0; i < num_images; ++i) {
        CUDA_CHECK_VEC(cudaMemcpy(d_inputs[i], images[i].data.data(), 
                                  input_size, cudaMemcpyHostToDevice));
    }
    
    // 已移除冗余的首值打印，避免额外开销
    
    // ========== 构建任务列表 ==========
    struct PendingTask {
        int result_idx;
        int out_w, out_h;
        float* h_buffer;
        bool valid;
    };
    std::vector<PendingTask> pending(num_streams, {-1, 0, 0, nullptr, false});
    std::vector<int> buffer_idx(num_streams, 0);  // 0=A, 1=B
    
    // ========== 主流水线循环 ==========
    int task_id = 0;
    int recovered_count = 0;
    int submitted_count = 0;
    
    while (task_id < (int)total_tasks) {
        for (int s = 0; s < num_streams && task_id < (int)total_tasks; ++s) {
            // 1. 先回收上一个任务的结果 (如果有)
            if (pending[s].valid) {
                CUDA_CHECK_VEC(cudaStreamSynchronize(streams[s]));
                size_t out_bytes = pending[s].out_w * pending[s].out_h * sizeof(float);
                
                // Verify result image is allocated correctly
                if (results[pending[s].result_idx].data.size() != (size_t)(pending[s].out_w * pending[s].out_h)) {
                    std::cerr << "ERROR: result[" << pending[s].result_idx << "] size mismatch: "
                              << results[pending[s].result_idx].data.size() << " vs expected "
                              << pending[s].out_w * pending[s].out_h << "\n";
                }
                
                memcpy(results[pending[s].result_idx].data.data(), pending[s].h_buffer, out_bytes);
                pending[s].valid = false;
                recovered_count++;
            }
            
            // 2. 启动新任务
            int img_idx = task_id / num_kernels;
            int k_idx = task_id % num_kernels;
            const Kernel& kernel = kernels[k_idx];
            int ks = kernel.size;
            int out_w = img_width - ks + 1;
            int out_h = img_height - ks + 1;
            
            // 分配输出图像
            results[task_id] = Image(out_w, out_h);
            
            // 选择当前 stream 的缓冲
            int buf = buffer_idx[s];
            float* d_out = (buf == 0) ? d_outputs_A[s] : d_outputs_B[s];
            float* h_out = (buf == 0) ? h_outputs_A[s] : h_outputs_B[s];
            buffer_idx[s] = 1 - buf;  // 切换缓冲
            
            // 异步传输卷积核
            CUDA_CHECK_VEC(cudaMemcpyAsync(d_kernels[s], kernel.data.data(),
                                           ks * ks * sizeof(float),
                                           cudaMemcpyHostToDevice, streams[s]));
            
            // 启动计算（自适应 block 尺寸共享内存核）
            launch_shared_kernel(
                d_inputs[img_idx], d_kernels[s], d_out,
                img_width, img_height, ks, out_w, out_h,
                streams[s]
            );
            
            // 捕捉潜在的内核启动错误，避免静默失败
            CUDA_CHECK_VEC(cudaGetLastError());

            // 异步传回结果
            size_t out_bytes = out_w * out_h * sizeof(float);
            CUDA_CHECK_VEC(cudaMemcpyAsync(h_out, d_out, out_bytes, 
                                           cudaMemcpyDeviceToHost, streams[s]));
            
            // 记录待处理任务
            pending[s] = {task_id, out_w, out_h, h_out, true};
            task_id++;
            submitted_count++;
        }
    }
    
    // 回收所有剩余任务
    // 确保所有 GPU 操作完成
    cudaDeviceSynchronize();
    
    for (int s = 0; s < num_streams; ++s) {
        if (pending[s].valid) {
            // 双保险等待当前 stream 完成，防止异步传输尚未落盘
            CUDA_CHECK_VEC(cudaStreamSynchronize(streams[s]));
            size_t out_bytes = pending[s].out_w * pending[s].out_h * sizeof(float);
            memcpy(results[pending[s].result_idx].data.data(), pending[s].h_buffer, out_bytes);
            recovered_count++;
        }
    }
    
    // ========== 清理 ==========
    for (int i = 0; i < num_images; ++i) {
        cudaFree(d_inputs[i]);
    }
    for (int i = 0; i < num_streams; ++i) {
        cudaFreeHost(h_outputs_A[i]);
        cudaFreeHost(h_outputs_B[i]);
        cudaFree(d_outputs_A[i]);
        cudaFree(d_outputs_B[i]);
        cudaFree(d_kernels[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return results;
}

/**
 * @brief CUDA 自适应最优实现
 * 小核(<= small_kernel_threshold)直接用共享内存单流执行，大核交给多流批处理，兼顾低开销与高吞吐。
 */
std::vector<Image> convolve_cuda_best(
    const std::vector<Image>& images,
    const std::vector<Kernel>& kernels,
    int num_streams
) {
    if (images.empty() || kernels.empty()) {
        return {};
    }

    if (num_streams <= 0) num_streams = 12;
    num_streams = std::min(num_streams, 16);

    const int num_images = (int)images.size();
    const int num_kernels = (int)kernels.size();
    const int total_tasks = num_images * num_kernels;

    int img_width = images[0].width;
    int img_height = images[0].height;

    // 找最大输出尺寸与最大 kernel，分配统一缓冲
    int max_kernel_size = 0;
    for (const auto& k : kernels) {
        max_kernel_size = std::max(max_kernel_size, k.size);
    }

    size_t input_size = img_width * img_height * sizeof(float);
    size_t max_output_size = img_width * img_height * sizeof(float);

    std::vector<Image> results(total_tasks);

    // ===== 预传图像 =====
    std::vector<float*> d_inputs(num_images);
    for (int i = 0; i < num_images; ++i) {
        CUDA_CHECK_VEC(cudaMalloc(&d_inputs[i], input_size));
        CUDA_CHECK_VEC(cudaMemcpy(d_inputs[i], images[i].data.data(), input_size, cudaMemcpyHostToDevice));
    }

    // ===== 预传卷积核（每个 kernel 一份，避免重复传输） =====
    std::vector<float*> d_kernel_table(num_kernels, nullptr);
    for (int k = 0; k < num_kernels; ++k) {
        size_t bytes = kernels[k].size * kernels[k].size * sizeof(float);
        CUDA_CHECK_VEC(cudaMalloc(&d_kernel_table[k], bytes));
        CUDA_CHECK_VEC(cudaMemcpy(d_kernel_table[k], kernels[k].data.data(), bytes, cudaMemcpyHostToDevice));
    }

    // ===== 每个 stream 资源：双缓冲输出 =====
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK_VEC(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    }

    std::vector<float*> d_outputs_A(num_streams);
    std::vector<float*> d_outputs_B(num_streams);
    std::vector<float*> h_outputs_A(num_streams);
    std::vector<float*> h_outputs_B(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK_VEC(cudaMalloc(&d_outputs_A[i], max_output_size));
        CUDA_CHECK_VEC(cudaMalloc(&d_outputs_B[i], max_output_size));
        CUDA_CHECK_VEC(cudaMallocHost(&h_outputs_A[i], max_output_size));
        CUDA_CHECK_VEC(cudaMallocHost(&h_outputs_B[i], max_output_size));
    }

    struct PendingTask {
        int result_idx;
        int out_w, out_h;
        float* h_buffer;
        bool valid;
    };
    std::vector<PendingTask> pending(num_streams, {-1, 0, 0, nullptr, false});
    std::vector<int> buffer_idx(num_streams, 0);

    // ===== 主循环：单一流水线，按任务顺序遍历 =====
    int task_id = 0;
    while (task_id < total_tasks) {
        for (int s = 0; s < num_streams && task_id < total_tasks; ++s) {
            // 回收
            if (pending[s].valid) {
                CUDA_CHECK_VEC(cudaStreamSynchronize(streams[s]));
                size_t out_bytes = pending[s].out_w * pending[s].out_h * sizeof(float);
                memcpy(results[pending[s].result_idx].data.data(), pending[s].h_buffer, out_bytes);
                pending[s].valid = false;
            }

            int img_idx = task_id / num_kernels;
            int k_idx = task_id % num_kernels;
            const Kernel& kernel = kernels[k_idx];
            int ks = kernel.size;
            int out_w = img_width - ks + 1;
            int out_h = img_height - ks + 1;

            results[task_id] = Image(out_w, out_h);

            int buf = buffer_idx[s];
            float* d_out = (buf == 0) ? d_outputs_A[s] : d_outputs_B[s];
            float* h_out = (buf == 0) ? h_outputs_A[s] : h_outputs_B[s];
            buffer_idx[s] = 1 - buf;

            // 小核走常量内存路径；否则自适应共享内存 block 尺寸
            if (ks * ks <= 1024) {
                CUDA_CHECK_VEC(cudaMemcpyToSymbolAsync(d_const_kernel, kernels[k_idx].data.data(),
                                                      ks * ks * sizeof(float), 0,
                                                      cudaMemcpyHostToDevice, streams[s]));
                dim3 block(BLOCK_SIZE, BLOCK_SIZE);
                dim3 grid((out_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE);
                convolve_kernel_const<<<grid, block, 0, streams[s]>>>(
                    d_inputs[img_idx], d_out,
                    img_width, img_height, ks, out_w, out_h
                );
            } else {
                launch_shared_kernel(
                    d_inputs[img_idx], d_kernel_table[k_idx], d_out,
                    img_width, img_height, ks, out_w, out_h,
                    streams[s]
                );
            }
            CUDA_CHECK_VEC(cudaGetLastError());

            size_t out_bytes = out_w * out_h * sizeof(float);
            CUDA_CHECK_VEC(cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, streams[s]));

            pending[s] = {task_id, out_w, out_h, h_out, true};
            task_id++;
        }
    }

    // 回收剩余
    cudaDeviceSynchronize();
    for (int s = 0; s < num_streams; ++s) {
        if (pending[s].valid) {
            CUDA_CHECK_VEC(cudaStreamSynchronize(streams[s]));
            size_t out_bytes = pending[s].out_w * pending[s].out_h * sizeof(float);
            memcpy(results[pending[s].result_idx].data.data(), pending[s].h_buffer, out_bytes);
        }
    }

    // 清理
    for (int i = 0; i < num_images; ++i) {
        cudaFree(d_inputs[i]);
    }
    for (int k = 0; k < num_kernels; ++k) {
        cudaFree(d_kernel_table[k]);
    }
    for (int i = 0; i < num_streams; ++i) {
        cudaFreeHost(h_outputs_A[i]);
        cudaFreeHost(h_outputs_B[i]);
        cudaFree(d_outputs_A[i]);
        cudaFree(d_outputs_B[i]);
        cudaStreamDestroy(streams[i]);
    }

    return results;
}

/**
 * @brief CPU-GPU 混合并行 - 极限流水线优化
 * 
 * ============== 新设计理念 ==============
 * 
 * 核心思想: 让 GPU 和 CPU 各自发挥最大性能，同时完全并行
 * 
 * GPU 端优化:
 * 1. 多 Stream 流水线: 12 个 stream 并行处理
 * 2. 双缓冲异步传输: 传输与计算完全重叠
 * 3. 预传输所有图像: 消除图像传输开销
 * 
 * CPU 端优化:
 * 1. OpenMP 多线程 + SIMD
 * 2. 只处理小卷积核 (3×3, 5×5) - 计算量少但启动快
 * 
 * 任务分配:
 * - CPU: 小卷积核 (3×3 到 7×7) - 约 10% 任务
 * - GPU: 大卷积核 (9×9 到 31×31) - 约 90% 任务
 */
std::vector<Image> convolve_hybrid(
    const std::vector<Image>& images,
    const std::vector<Kernel>& kernels,
    int num_threads
) {
    if (images.empty() || kernels.empty()) {
        return {};
    }
    
    // 确保 GPU 状态干净
    cudaDeviceSynchronize();
    
    if (num_threads <= 0) num_threads = omp_get_max_threads();
    
    int img_width = images[0].width;
    int img_height = images[0].height;
    int num_images = (int)images.size();
    int num_kernels = (int)kernels.size();
    int total_tasks = num_images * num_kernels;
    
    std::vector<Image> results(total_tasks);
    
    // ========== 构建任务列表并按卷积核大小排序 ==========
    struct Task {
        int img_idx;
        int kernel_idx;
        int result_idx;
        int kernel_size;
    };
    
    std::vector<Task> tasks;
    tasks.reserve(total_tasks);
    for (int i = 0; i < num_images; ++i) {
        for (int k = 0; k < num_kernels; ++k) {
            tasks.push_back({i, k, i * num_kernels + k, kernels[k].size});
        }
    }
    
    // 不排序！保持原始顺序以避免复杂性
    // std::sort(tasks.begin(), tasks.end(), 
    //           [](const Task& a, const Task& b) { return a.kernel_size < b.kernel_size; });
    
    // ========== 任务分配: CPU 处理前一部分, GPU 处理后一部分 ==========
    // 给 CPU 分配 0 个任务来测试 - DEBUG
    int cpu_tasks = 0;  // std::max(1, (total_tasks * 10) / 100);
    int gpu_start = cpu_tasks;
    
    std::cerr << "[Hybrid] total=" << total_tasks << ", cpu_tasks=" << cpu_tasks << ", gpu_start=" << gpu_start << "\n";
    
    // 如果当前策略将所有任务都交给 GPU，直接复用已验证的流式实现以减少重复代码
    if (gpu_start >= total_tasks || cpu_tasks == 0) {
        return convolve_batch_cuda_streams(images, kernels, 12);
    }
    
    int max_kernel_size = 0;
    for (const auto& k : kernels) {
        max_kernel_size = std::max(max_kernel_size, k.size);
    }
    
    size_t input_size = img_width * img_height * sizeof(float);
    size_t max_output_size = img_width * img_height * sizeof(float);
    size_t max_kernel_bytes = max_kernel_size * max_kernel_size * sizeof(float);
    
    // ========== GPU 线程 (流水线优化) ==========
    std::atomic<int> gpu_completed{0};
    std::thread gpu_thread([&]() {
        const int num_streams = 12;
        
        // 创建 streams
        std::vector<cudaStream_t> streams(num_streams);
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }
        
        // 预传输所有图像
        std::vector<float*> d_inputs(num_images);
        for (int i = 0; i < num_images; ++i) {
            cudaMalloc(&d_inputs[i], input_size);
            cudaMemcpy(d_inputs[i], images[i].data.data(), input_size, cudaMemcpyHostToDevice);
        }
        
        // 每个 stream 的资源
        std::vector<float*> d_kernels(num_streams);
        std::vector<float*> d_outputs_A(num_streams);
        std::vector<float*> d_outputs_B(num_streams);
        std::vector<float*> h_outputs_A(num_streams);
        std::vector<float*> h_outputs_B(num_streams);
        
        for (int i = 0; i < num_streams; ++i) {
            cudaMalloc(&d_kernels[i], max_kernel_bytes);
            cudaMalloc(&d_outputs_A[i], max_output_size);
            cudaMalloc(&d_outputs_B[i], max_output_size);
            cudaMallocHost(&h_outputs_A[i], max_output_size);
            cudaMallocHost(&h_outputs_B[i], max_output_size);
        }
        
        // 待处理任务跟踪
        struct PendingTask {
            int result_idx;
            int out_w, out_h;
            float* h_buffer;
            bool valid;
        };
        std::vector<PendingTask> pending(num_streams, {-1, 0, 0, nullptr, false});
        std::vector<int> buffer_idx(num_streams, 0);
        
        // 流水线处理 GPU 任务
        int task_idx = gpu_start;
        
        while (task_idx < total_tasks) {
            for (int s = 0; s < num_streams && task_idx < total_tasks; ++s) {
                // 回收上一个任务
                if (pending[s].valid) {
                    cudaStreamSynchronize(streams[s]);
                    size_t out_bytes = pending[s].out_w * pending[s].out_h * sizeof(float);
                    
                    // DEBUG: Check first value
                    if (gpu_completed == 0) {
                        std::cerr << "[Hybrid GPU] First recovered: idx=" << pending[s].result_idx 
                                  << ", h_buffer[0]=" << pending[s].h_buffer[0]
                                  << ", h_buffer[100]=" << pending[s].h_buffer[100] << "\n";
                    }
                    
                    memcpy(results[pending[s].result_idx].data.data(), pending[s].h_buffer, out_bytes);
                    pending[s].valid = false;
                    gpu_completed++;
                }
                
                // 启动新任务
                const Task& task = tasks[task_idx];
                const Kernel& kernel = kernels[task.kernel_idx];
                int ks = kernel.size;
                int out_w = img_width - ks + 1;
                int out_h = img_height - ks + 1;
                
                results[task.result_idx] = Image(out_w, out_h);
                
                int buf = buffer_idx[s];
                float* d_out = (buf == 0) ? d_outputs_A[s] : d_outputs_B[s];
                float* h_out = (buf == 0) ? h_outputs_A[s] : h_outputs_B[s];
                buffer_idx[s] = 1 - buf;
                
                cudaMemcpyAsync(d_kernels[s], kernel.data.data(),
                               ks * ks * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
                
                launch_shared_kernel(
                    d_inputs[task.img_idx], d_kernels[s], d_out,
                    img_width, img_height, ks, out_w, out_h,
                    streams[s]
                );
                
                size_t out_bytes = out_w * out_h * sizeof(float);
                cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, streams[s]);
                
                pending[s] = {task.result_idx, out_w, out_h, h_out, true};
                task_idx++;
            }
        }
        
        // 回收所有剩余任务
        for (int s = 0; s < num_streams; ++s) {
            if (pending[s].valid) {
                cudaStreamSynchronize(streams[s]);
                size_t out_bytes = pending[s].out_w * pending[s].out_h * sizeof(float);
                memcpy(results[pending[s].result_idx].data.data(), pending[s].h_buffer, out_bytes);
                gpu_completed++;
            }
        }
        
        std::cerr << "[Hybrid GPU] completed=" << gpu_completed.load() << " tasks\n";
        
        // 确保所有 CUDA 操作完成
        cudaDeviceSynchronize();
        
        // 清理
        for (int i = 0; i < num_images; ++i) {
            cudaFree(d_inputs[i]);
        }
        for (int i = 0; i < num_streams; ++i) {
            cudaFreeHost(h_outputs_A[i]);
            cudaFreeHost(h_outputs_B[i]);
            cudaFree(d_outputs_A[i]);
            cudaFree(d_outputs_B[i]);
            cudaFree(d_kernels[i]);
            cudaStreamDestroy(streams[i]);
        }
    });
    
    // ========== CPU 线程 (OpenMP 处理小卷积核) ==========
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int t = 0; t < cpu_tasks; ++t) {
        const Task& task = tasks[t];
        const Image& img = images[task.img_idx];
        const Kernel& kernel = kernels[task.kernel_idx];
        int ks = kernel.size;
        int out_w = img_width - ks + 1;
        int out_h = img_height - ks + 1;
        
        Image output(out_w, out_h);
        
        // 卷积计算
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float sum = 0.0f;
                for (int ky = 0; ky < ks; ++ky) {
                    const float* img_row = &img.data[(y + ky) * img_width + x];
                    const float* ker_row = &kernel.data[ky * ks];
                    for (int kx = 0; kx < ks; ++kx) {
                        sum += img_row[kx] * ker_row[kx];
                    }
                }
                output.data[y * out_w + x] = std::max(0.0f, std::min(255.0f, sum));
            }
        }
        
        results[task.result_idx] = std::move(output);
    }
    
    // 等待 GPU 完成
    gpu_thread.join();
    
    return results;
}

/**
 * @brief 自适应混合并行
 */
std::vector<Image> convolve_adaptive(
    const std::vector<Image>& images,
    const std::vector<Kernel>& kernels,
    int num_threads,
    int gpu_threshold
) {
    if (images.empty() || kernels.empty()) {
        return {};
    }
    
    // 判断是否使用 GPU
    bool use_gpu = (images[0].width >= gpu_threshold && images[0].height >= gpu_threshold);
    
    if (use_gpu) {
        return convolve_batch_cuda(images, kernels);
    } else {
        // 使用 CPU OpenMP 版本
        return convolve_batch_omp(images, kernels, num_threads);
    }
}

} // namespace Convolution
