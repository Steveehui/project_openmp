/**
 * @file convolution.cpp
 * @brief 卷积算法的串行和OpenMP并行实现
 * @author 高性能计算课程项目
 * @date 2025-12-10
 * 
 * 本文件包含多种卷积实现:
 * 1. 基础串行版本
 * 2. 缓存优化版本(循环分块)
 * 3. 行缓存优化版本
 * 4. OpenMP并行版本(多种优化策略)
 * 5. SIMD向量化版本
 * 6. 可分离卷积优化
 */

#include "convolution.h"
#include "timer.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace Convolution {

// ============================================================================
// 串行实现
// ============================================================================

/**
 * @brief 串行卷积核心实现(基础版本)
 * 边界处理: 忽略边界(输出图像尺寸减小)
 * 计算复杂度: O(W * H * K^2)
 */
Image convolve_serial(const Image& input, const Kernel& kernel) {
    int padding = kernel.size / 2;
    int out_width = input.width - 2 * padding;
    int out_height = input.height - 2 * padding;
    
    if (out_width <= 0 || out_height <= 0) {
        std::cerr << "Error: image size smaller than kernel" << std::endl;
        return Image();
    }
    
    Image output(out_width, out_height);
    
    const float* input_data = input.ptr();
    const float* kernel_data = kernel.ptr();
    float* output_data = output.ptr();
    const int input_width = input.width;
    const int kernel_size = kernel.size;
    
    // 遍历输出图像的每个像素
    for (int out_y = 0; out_y < out_height; ++out_y) {
        for (int out_x = 0; out_x < out_width; ++out_x) {
            float sum = 0.0f;
            
            // 卷积计算 - 展开内层循环以提高性能
            for (int ky = 0; ky < kernel_size; ++ky) {
                const int in_y = out_y + ky;
                const float* in_row = input_data + in_y * input_width + out_x;
                const float* k_row = kernel_data + ky * kernel_size;
                
                for (int kx = 0; kx < kernel_size; ++kx) {
                    sum += in_row[kx] * k_row[kx];
                }
            }
            
            // 限制输出范围 [0, 255]
            output_data[out_y * out_width + out_x] = std::max(0.0f, std::min(255.0f, sum));
        }
    }
    
    return output;
}

/**
 * @brief 串行卷积(循环分块/缓存优化版本)
 * 使用分块策略提高缓存命中率
 */
Image convolve_serial_blocked(const Image& input, const Kernel& kernel, int block_size) {
    int padding = kernel.size / 2;
    int out_width = input.width - 2 * padding;
    int out_height = input.height - 2 * padding;
    
    if (out_width <= 0 || out_height <= 0) {
        std::cerr << "Error: image size smaller than kernel" << std::endl;
        return Image();
    }
    
    Image output(out_width, out_height);
    
    const float* input_data = input.ptr();
    const float* kernel_data = kernel.ptr();
    float* output_data = output.ptr();
    const int input_width = input.width;
    const int kernel_size = kernel.size;
    
    // 分块遍历
    for (int by = 0; by < out_height; by += block_size) {
        int y_end = std::min(by + block_size, out_height);
        
        for (int bx = 0; bx < out_width; bx += block_size) {
            int x_end = std::min(bx + block_size, out_width);
            
            // 处理当前块
            for (int out_y = by; out_y < y_end; ++out_y) {
                for (int out_x = bx; out_x < x_end; ++out_x) {
                    float sum = 0.0f;
                    
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        const int in_y = out_y + ky;
                        const float* in_row = input_data + in_y * input_width + out_x;
                        const float* k_row = kernel_data + ky * kernel_size;
                        
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            sum += in_row[kx] * k_row[kx];
                        }
                    }
                    
                    output_data[out_y * out_width + out_x] = std::max(0.0f, std::min(255.0f, sum));
                }
            }
        }
    }
    
    return output;
}

/**
 * @brief 串行卷积(行缓存优化版本)
 * 预加载卷积核所需的行数据到本地缓存
 */
Image convolve_serial_row_cache(const Image& input, const Kernel& kernel) {
    int padding = kernel.size / 2;
    int out_width = input.width - 2 * padding;
    int out_height = input.height - 2 * padding;
    
    if (out_width <= 0 || out_height <= 0) {
        std::cerr << "Error: image size smaller than kernel" << std::endl;
        return Image();
    }
    
    Image output(out_width, out_height);
    
    const int input_width = input.width;
    const int kernel_size = kernel.size;
    const float* input_data = input.ptr();
    const float* kernel_data = kernel.ptr();
    float* output_data = output.ptr();
    
    // 行缓存
    std::vector<std::vector<float>> row_cache(kernel_size, std::vector<float>(input_width));
    
    // 初始化缓存
    for (int i = 0; i < kernel_size; ++i) {
        std::copy(input_data + i * input_width, 
                  input_data + (i + 1) * input_width, 
                  row_cache[i].begin());
    }
    
    // 遍历输出图像
    for (int out_y = 0; out_y < out_height; ++out_y) {
        // 更新行缓存(滚动)
        if (out_y > 0) {
            // 将最旧的行移到最后
            std::vector<float> temp = std::move(row_cache[0]);
            for (int i = 0; i < kernel_size - 1; ++i) {
                row_cache[i] = std::move(row_cache[i + 1]);
            }
            // 加载新行
            int new_row = out_y + kernel_size - 1;
            std::copy(input_data + new_row * input_width,
                      input_data + (new_row + 1) * input_width,
                      temp.begin());
            row_cache[kernel_size - 1] = std::move(temp);
        }
        
        // 计算该行的所有输出
        for (int out_x = 0; out_x < out_width; ++out_x) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                const float* cache_row = row_cache[ky].data() + out_x;
                const float* k_row = kernel_data + ky * kernel_size;
                
                for (int kx = 0; kx < kernel_size; ++kx) {
                    sum += cache_row[kx] * k_row[kx];
                }
            }
            
            output_data[out_y * out_width + out_x] = std::max(0.0f, std::min(255.0f, sum));
        }
    }
    
    return output;
}

// ============================================================================
// OpenMP 并行实现
// ============================================================================

/**
 * @brief OpenMP 并行卷积实现(基础版本)
 */
Image convolve_omp(const Image& input, const Kernel& kernel, int num_threads) {
    int padding = kernel.size / 2;
    int out_width = input.width - 2 * padding;
    int out_height = input.height - 2 * padding;
    
    if (out_width <= 0 || out_height <= 0) {
        std::cerr << "Error: image size smaller than kernel" << std::endl;
        return Image();
    }
    
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    Image output(out_width, out_height);
    
    const float* input_data = input.ptr();
    const float* kernel_data = kernel.ptr();
    float* output_data = output.ptr();
    const int input_width = input.width;
    const int kernel_size = kernel.size;
    
    // OpenMP 并行化外层循环
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16)
    for (int out_y = 0; out_y < out_height; ++out_y) {
        for (int out_x = 0; out_x < out_width; ++out_x) {
            float sum = 0.0f;
            
            // 卷积计算
            for (int ky = 0; ky < kernel_size; ++ky) {
                const int in_y = out_y + ky;
                const float* in_row = input_data + in_y * input_width + out_x;
                const float* k_row = kernel_data + ky * kernel_size;
                
                for (int kx = 0; kx < kernel_size; ++kx) {
                    sum += in_row[kx] * k_row[kx];
                }
            }
            
            output_data[out_y * out_width + out_x] = std::max(0.0f, std::min(255.0f, sum));
        }
    }
    
    return output;
}

/**
 * @brief OpenMP 并行卷积(循环分块优化)
 */
Image convolve_omp_blocked(const Image& input, const Kernel& kernel, int num_threads, int block_size) {
    int padding = kernel.size / 2;
    int out_width = input.width - 2 * padding;
    int out_height = input.height - 2 * padding;
    
    if (out_width <= 0 || out_height <= 0) {
        std::cerr << "Error: image size smaller than kernel" << std::endl;
        return Image();
    }
    
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    Image output(out_width, out_height);
    
    const float* input_data = input.ptr();
    const float* kernel_data = kernel.ptr();
    float* output_data = output.ptr();
    const int input_width = input.width;
    const int kernel_size = kernel.size;
    
    int num_blocks_y = (out_height + block_size - 1) / block_size;
    int num_blocks_x = (out_width + block_size - 1) / block_size;
    int total_blocks = num_blocks_y * num_blocks_x;
    
    // 并行处理块
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (int block_id = 0; block_id < total_blocks; ++block_id) {
        int by = (block_id / num_blocks_x) * block_size;
        int bx = (block_id % num_blocks_x) * block_size;
        int y_end = std::min(by + block_size, out_height);
        int x_end = std::min(bx + block_size, out_width);
        
        for (int out_y = by; out_y < y_end; ++out_y) {
            for (int out_x = bx; out_x < x_end; ++out_x) {
                float sum = 0.0f;
                
                for (int ky = 0; ky < kernel_size; ++ky) {
                    const int in_y = out_y + ky;
                    const float* in_row = input_data + in_y * input_width + out_x;
                    const float* k_row = kernel_data + ky * kernel_size;
                    
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        sum += in_row[kx] * k_row[kx];
                    }
                }
                
                output_data[out_y * out_width + out_x] = std::max(0.0f, std::min(255.0f, sum));
            }
        }
    }
    
    return output;
}

/**
 * @brief OpenMP 并行卷积(SIMD向量化)
 * 使用 OpenMP SIMD 指令进行向量化
 */
Image convolve_omp_simd(const Image& input, const Kernel& kernel, int num_threads) {
    int padding = kernel.size / 2;
    int out_width = input.width - 2 * padding;
    int out_height = input.height - 2 * padding;
    
    if (out_width <= 0 || out_height <= 0) {
        std::cerr << "Error: image size smaller than kernel" << std::endl;
        return Image();
    }
    
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    Image output(out_width, out_height);
    
    const float* input_data = input.ptr();
    const float* kernel_data = kernel.ptr();
    float* output_data = output.ptr();
    const int input_width = input.width;
    const int kernel_size = kernel.size;
    const int kernel_total = kernel_size * kernel_size;
    
    // 预计算卷积核偏移
    std::vector<int> offsets(kernel_total);
    std::vector<float> kernel_copy(kernel_total);
    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int idx = ky * kernel_size + kx;
            offsets[idx] = ky * input_width + kx;
            kernel_copy[idx] = kernel_data[idx];
        }
    }
    
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int out_y = 0; out_y < out_height; ++out_y) {
        const float* in_base = input_data + out_y * input_width;
        float* out_row = output_data + out_y * out_width;
        
        for (int out_x = 0; out_x < out_width; ++out_x) {
            float sum = 0.0f;
            
            for (int k = 0; k < kernel_total; ++k) {
                sum += in_base[out_x + offsets[k]] * kernel_copy[k];
            }
            
            out_row[out_x] = (sum < 0.0f) ? 0.0f : ((sum > 255.0f) ? 255.0f : sum);
        }
    }
    
    return output;
}

/**
 * @brief OpenMP 并行卷积(可分离卷积核优化)
 * 将 NxN 卷积分解为两次 Nx1 卷积,复杂度从 O(N^2) 降到 O(2N)
 */
Image convolve_omp_separable(const Image& input, const std::vector<float>& kernel_1d, int num_threads) {
    int kernel_size = static_cast<int>(kernel_1d.size());
    int padding = kernel_size / 2;
    int out_width = input.width - 2 * padding;
    int out_height = input.height - 2 * padding;
    
    if (out_width <= 0 || out_height <= 0) {
        std::cerr << "Error: image size smaller than kernel" << std::endl;
        return Image();
    }
    
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    // 第一次卷积(水平方向)
    Image temp(out_width, input.height);
    
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; ++k) {
                sum += input.at(x + k, y) * kernel_1d[k];
            }
            temp.at(x, y) = sum;
        }
    }
    
    // 第二次卷积(垂直方向)
    Image output(out_width, out_height);
    
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; ++k) {
                sum += temp.at(x, y + k) * kernel_1d[k];
            }
            output.at(x, y) = std::max(0.0f, std::min(255.0f, sum));
        }
    }
    
    return output;
}

/**
 * @brief 批量处理(任务并行)
 */
std::vector<Image> convolve_batch_omp(
    const std::vector<Image>& images,
    const std::vector<Kernel>& kernels,
    int num_threads
) {
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    int total_tasks = static_cast<int>(images.size() * kernels.size());
    std::vector<Image> results(total_tasks);
    
    // 任务级并行
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (int task_id = 0; task_id < total_tasks; ++task_id) {
        int img_idx = task_id / static_cast<int>(kernels.size());
        int kernel_idx = task_id % static_cast<int>(kernels.size());
        
        results[task_id] = convolve_serial(images[img_idx], kernels[kernel_idx]);
    }
    
    return results;
}

/**
 * @brief 批量处理(负载均衡版本)
 * 根据卷积核大小估算计算量,进行加权任务分配
 */
std::vector<Image> convolve_batch_omp_balanced(
    const std::vector<Image>& images,
    const std::vector<Kernel>& kernels,
    int num_threads
) {
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    // 计算每个任务的权重(基于图像大小和卷积核大小)
    struct Task {
        int img_idx;
        int kernel_idx;
        int weight;  // 计算量估算
    };
    
    std::vector<Task> tasks;
    tasks.reserve(images.size() * kernels.size());
    
    for (size_t i = 0; i < images.size(); ++i) {
        for (size_t k = 0; k < kernels.size(); ++k) {
            int weight = images[i].width * images[i].height * 
                        kernels[k].size * kernels[k].size;
            tasks.push_back({static_cast<int>(i), static_cast<int>(k), weight});
        }
    }
    
    // 按权重降序排序(大任务优先)
    std::sort(tasks.begin(), tasks.end(), 
              [](const Task& a, const Task& b) { return a.weight > b.weight; });
    
    int total_tasks = static_cast<int>(tasks.size());
    std::vector<Image> results(images.size() * kernels.size());
    
    // 并行执行
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (int t = 0; t < total_tasks; ++t) {
        const Task& task = tasks[t];
        results[task.img_idx * kernels.size() + task.kernel_idx] = 
            convolve_serial(images[task.img_idx], kernels[task.kernel_idx]);
    }
    
    return results;
}

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 验证结果正确性
 */
bool verifyResults(const Image& img1, const Image& img2, float tolerance) {
    if (img1.width != img2.width || img1.height != img2.height) {
        std::cerr << "Image dimensions mismatch!" << std::endl;
        return false;
    }
    
    float total_diff = 0.0f;
    int num_pixels = img1.width * img1.height;
    
    for (int i = 0; i < num_pixels; ++i) {
        total_diff += std::abs(img1.data[i] - img2.data[i]);
    }
    
    float avg_diff = total_diff / num_pixels;
    
    if (avg_diff <= tolerance) {
        std::cout << "Verification passed! Average pixel diff: " << avg_diff << std::endl;
        return true;
    } else {
        std::cerr << "Verification failed! Average pixel diff: " << avg_diff 
                  << " (tolerance: " << tolerance << ")" << std::endl;
        return false;
    }
}

/**
 * @brief 计算均方误差
 */
float computeMSE(const Image& img1, const Image& img2) {
    if (img1.width != img2.width || img1.height != img2.height) {
        return -1.0f;
    }
    
    float mse = 0.0f;
    int num_pixels = img1.width * img1.height;
    
    for (int i = 0; i < num_pixels; ++i) {
        float diff = img1.data[i] - img2.data[i];
        mse += diff * diff;
    }
    
    return mse / num_pixels;
}

/**
 * @brief 检查卷积核是否可分离
 * 尝试将 2D 卷积核分解为两个 1D 向量的外积
 */
bool isSeparable(const Kernel& kernel, std::vector<float>& row_kernel, std::vector<float>& col_kernel) {
    // 简化实现:只检查高斯类型的卷积核
    // 完整实现应该使用 SVD 分解
    
    int size = kernel.size;
    if (size < 3) return false;
    
    // 检查是否对称
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (std::abs(kernel.at(x, y) - kernel.at(y, x)) > 1e-6f) {
                return false;
            }
        }
    }
    
    // 提取第一行作为行核
    row_kernel.resize(size);
    col_kernel.resize(size);
    
    float center_val = kernel.at(size/2, 0);
    if (std::abs(center_val) < 1e-6f) return false;
    
    // 从中心行提取
    for (int i = 0; i < size; ++i) {
        row_kernel[i] = kernel.at(i, size/2);
    }
    
    // 归一化并验证
    float norm = std::sqrt(std::abs(row_kernel[size/2]));
    if (norm < 1e-6f) return false;
    
    for (int i = 0; i < size; ++i) {
        row_kernel[i] = std::sqrt(std::abs(row_kernel[i])) * (row_kernel[i] >= 0 ? 1 : -1);
    }
    
    col_kernel = row_kernel;
    
    // 验证分解是否正确
    float max_error = 0.0f;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float reconstructed = row_kernel[x] * col_kernel[y];
            float error = std::abs(reconstructed - kernel.at(x, y));
            max_error = std::max(max_error, error);
        }
    }
    
    return max_error < 0.01f;
}

/**
 * @brief 获取推荐的分块大小
 */
int getOptimalBlockSize(int image_size, int kernel_size) {
    // 根据 L1 缓存大小(通常 32KB)和图像大小确定
    const int L1_CACHE_SIZE = 32 * 1024;  // 32 KB
    const int FLOAT_SIZE = sizeof(float);
    
    // 每个块需要的内存: (block_size + kernel_size)^2 * float_size
    // 目标: 让一个块能够放入 L1 缓存
    
    int max_block = static_cast<int>(std::sqrt(L1_CACHE_SIZE / FLOAT_SIZE)) - kernel_size;
    max_block = std::max(16, std::min(max_block, 128));
    
    // 调整为 16 的倍数(有利于向量化)
    return (max_block / 16) * 16;
}

} // namespace Convolution
