#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "image_util.h"
#include <vector>

/**
 * @brief 卷积操作的统一接口
 */
namespace Convolution {
    // ==================== 串行实现 ====================
    
    /**
     * @brief 串行卷积实现(基础版本)
     */
    Image convolve_serial(const Image& input, const Kernel& kernel);
    
    /**
     * @brief 串行卷积(缓存优化版本) - 循环分块
     */
    Image convolve_serial_blocked(const Image& input, const Kernel& kernel, int block_size = 64);
    
    /**
     * @brief 串行卷积(行缓存优化版本)
     */
    Image convolve_serial_row_cache(const Image& input, const Kernel& kernel);
    
    // ==================== OpenMP 并行实现 ====================
    
    /**
     * @brief OpenMP 并行卷积(基础版本)
     */
    Image convolve_omp(const Image& input, const Kernel& kernel, int num_threads = 0);
    
    /**
     * @brief OpenMP 并行卷积(循环分块优化)
     */
    Image convolve_omp_blocked(const Image& input, const Kernel& kernel, int num_threads = 0, int block_size = 64);
    
    /**
     * @brief OpenMP 并行卷积(SIMD向量化)
     */
    Image convolve_omp_simd(const Image& input, const Kernel& kernel, int num_threads = 0);
    
    /**
     * @brief OpenMP 并行卷积(可分离卷积核优化)
     * 仅适用于可分离的卷积核(如高斯核),将 NxN 卷积分解为两次 Nx1 卷积
     */
    Image convolve_omp_separable(const Image& input, const std::vector<float>& kernel_1d, int num_threads = 0);
    
    /**
     * @brief OpenMP 批量卷积(任务并行)
     */
    std::vector<Image> convolve_batch_omp(
        const std::vector<Image>& images,
        const std::vector<Kernel>& kernels,
        int num_threads = 0
    );
    
    /**
     * @brief OpenMP 批量卷积(数据并行 + 负载均衡)
     */
    std::vector<Image> convolve_batch_omp_balanced(
        const std::vector<Image>& images,
        const std::vector<Kernel>& kernels,
        int num_threads = 0
    );

#ifdef USE_CUDA
    // ==================== CUDA GPU 实现 ====================
    
    /**
     * @brief CUDA 卷积(朴素版本)
     */
    Image convolve_cuda(const Image& input, const Kernel& kernel);
    
    /**
     * @brief CUDA 卷积(共享内存优化)
     */
    Image convolve_cuda_shared(const Image& input, const Kernel& kernel);

    /**
     * @brief CUDA 卷积(常量内存优化，小核)
     */
    Image convolve_cuda_const(const Image& input, const Kernel& kernel);

    /**
     * @brief 自适应策略卷积（占位含 Winograd/FFT/Sparse/Tensor-Core 调度）
     */
    enum class ConvPolicy {
        Auto,
        ConstSmall,
        Shared,
        FFTLike,
        SparseBlock
    };
    Image convolve_cuda_policy(const Image& input, const Kernel& kernel, ConvPolicy policy = ConvPolicy::Auto);
    
    /**
     * @brief CUDA 批量卷积(朴素版本)
     */
    std::vector<Image> convolve_batch_cuda(
        const std::vector<Image>& images,
        const std::vector<Kernel>& kernels
    );
    
    /**
     * @brief CUDA 批量卷积(共享内存优化)
     */
    std::vector<Image> convolve_batch_cuda_shared(
        const std::vector<Image>& images,
        const std::vector<Kernel>& kernels
    );
    
    /**
     * @brief CUDA 卷积(常量内存优化 - 适合小卷积核)
     */
    Image convolve_cuda_const(const Image& input, const Kernel& kernel);
    
    /**
     * @brief CUDA 卷积(纹理内存优化)
     */
    Image convolve_cuda_texture(const Image& input, const Kernel& kernel);
    
    /**
     * @brief CUDA 批量卷积(流水线优化)
     */
    std::vector<Image> convolve_batch_cuda_streams(
        const std::vector<Image>& images,
        const std::vector<Kernel>& kernels,
        int num_streams = 4
    );

    /**
     * @brief CUDA 自适应最优实现
     * 小核(<=7)直接用共享内存单流执行，大核使用多流批处理，兼顾低开销与高吞吐。
     */
    std::vector<Image> convolve_cuda_best(
        const std::vector<Image>& images,
        const std::vector<Kernel>& kernels,
        int num_streams = 12
    );
    
    /**
     * @brief 混合并行: OpenMP 调度 + CUDA 执行
     */
    std::vector<Image> convolve_hybrid(
        const std::vector<Image>& images,
        const std::vector<Kernel>& kernels,
        int num_threads = 0
    );
    
    /**
     * @brief 自适应混合: 根据任务大小选择 CPU 或 GPU
     */
    std::vector<Image> convolve_adaptive(
        const std::vector<Image>& images,
        const std::vector<Kernel>& kernels,
        int num_threads = 0,
        int gpu_threshold = 512  // 图像尺寸阈值
    );
#endif

    // ==================== 工具函数 ====================
    
    /**
     * @brief 验证两个图像是否相似
     */
    bool verifyResults(const Image& img1, const Image& img2, float tolerance = 1.0f);
    
    /**
     * @brief 计算两个图像的均方误差
     */
    float computeMSE(const Image& img1, const Image& img2);
    
    /**
     * @brief 检查卷积核是否可分离
     */
    bool isSeparable(const Kernel& kernel, std::vector<float>& row_kernel, std::vector<float>& col_kernel);
    
    /**
     * @brief 获取推荐的分块大小
     */
    int getOptimalBlockSize(int image_size, int kernel_size);
}

#endif // CONVOLUTION_H
