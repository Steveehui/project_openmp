/**
 * @file benchmark_cuda.cpp
 * @brief CUDA 算法专项测试程序
 * 
 * 只测试 CUDA 相关的算法:
 * - CUDA_Naive
 * - CUDA_Shared
 * - CUDA_Streams
 * - Hybrid
 */

#include "convolution.h"
#include "image_util.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

// 验证结果
double calculateMSE(const std::vector<Image>& a, const std::vector<Image>& b) {
    if (a.size() != b.size()) return -1;
    double total_mse = 0;
    size_t total_pixels = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].width != b[i].width || a[i].height != b[i].height) return -1;
        for (size_t j = 0; j < a[i].data.size(); ++j) {
            double diff = a[i].data[j] - b[i].data[j];
            total_mse += diff * diff;
        }
        total_pixels += a[i].data.size();
    }
    return total_mse / total_pixels;
}

int main(int argc, char* argv[]) {
    // 默认参数
    int image_size = 4096;
    int num_images = 3;
    int num_threads = 16;
    int iterations = 1;
    
    if (argc >= 2) image_size = std::atoi(argv[1]);
    if (argc >= 3) num_images = std::atoi(argv[2]);
    if (argc >= 4) num_threads = std::atoi(argv[3]);
    if (argc >= 5) iterations = std::atoi(argv[4]);
    
    std::cout << "========================================================\n";
    std::cout << "  CUDA Convolution Benchmark\n";
    std::cout << "========================================================\n\n";
    
    std::cout << "Test Configuration:\n";
    std::cout << "  Image Size: " << image_size << " x " << image_size << "\n";
    std::cout << "  Number of Images: " << num_images << "\n";
    std::cout << "  CPU Threads: " << num_threads << "\n";
    std::cout << "  Iterations: " << iterations << "\n\n";
    
    // 生成测试数据
    std::cout << "[1/5] Generating test images...\n";
    Timer timer;
    timer.start();
    std::vector<Image> images = ImageUtil::generateTestImages(num_images, image_size, image_size, "random");
    timer.stop();
    std::cout << "  Done in " << std::fixed << std::setprecision(2) << timer.elapsed() << " sec\n\n";
    
    // 创建卷积核
    std::cout << "[2/5] Creating convolution kernels...\n";
    std::vector<Kernel> kernels = KernelFactory::getLargeKernelSet();
    int total_tasks = num_images * kernels.size();
    std::cout << "  Total kernels: " << kernels.size() << "\n";
    std::cout << "  Total tasks: " << total_tasks << " convolutions\n\n";
    
    // 计算基准 (使用 CUDA_Shared)
    std::cout << "[3/5] Computing reference results (CUDA_Shared)...\n";
    timer.start();
    std::vector<Image> reference = Convolution::convolve_batch_cuda_shared(images, kernels);
    timer.stop();
    double ref_time = timer.elapsed();
    std::cout << "  Reference time: " << std::fixed << std::setprecision(2) << ref_time << " sec\n\n";
    
    // 测试结果存储
    struct Result {
        std::string name;
        double time;
        double speedup;
        bool verified;
        double mse;
    };
    std::vector<Result> results;
    
    std::cout << "[4/5] Testing CUDA implementations...\n\n";
    
    // 测试 CUDA_Naive
    {
        std::cout << "  Testing CUDA_Naive...\n";
        double total_time = 0;
        std::vector<Image> output;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            output = Convolution::convolve_batch_cuda(images, kernels);
            timer.stop();
            total_time += timer.elapsed();
        }
        double avg_time = total_time / iterations;
        double mse = calculateMSE(reference, output);
        results.push_back({"CUDA_Naive", avg_time, ref_time / avg_time, mse < 1e-6, mse});
        std::cout << "    Time: " << std::fixed << std::setprecision(2) << avg_time << " sec, "
                  << "Speedup vs Shared: " << std::setprecision(2) << (ref_time / avg_time) << "x\n";
    }
    
    // 测试 CUDA_Shared (再次确认)
    {
        std::cout << "  Testing CUDA_Shared...\n";
        double total_time = 0;
        std::vector<Image> output;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            output = Convolution::convolve_batch_cuda_shared(images, kernels);
            timer.stop();
            total_time += timer.elapsed();
        }
        double avg_time = total_time / iterations;
        double mse = calculateMSE(reference, output);
        results.push_back({"CUDA_Shared", avg_time, ref_time / avg_time, mse < 1e-6, mse});
        std::cout << "    Time: " << std::fixed << std::setprecision(2) << avg_time << " sec, "
                  << "Speedup vs Shared: " << std::setprecision(2) << (ref_time / avg_time) << "x (baseline)\n";
    }
    
    // 测试 CUDA_Streams
    {
        std::cout << "  Testing CUDA_Streams...\n";
        double total_time = 0;
        std::vector<Image> output;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            output = Convolution::convolve_batch_cuda_streams(images, kernels, 8);
            timer.stop();
            total_time += timer.elapsed();
        }
        double avg_time = total_time / iterations;
        double mse = calculateMSE(reference, output);
        results.push_back({"CUDA_Streams", avg_time, ref_time / avg_time, mse < 1e-6, mse});
        std::cout << "    Time: " << std::fixed << std::setprecision(2) << avg_time << " sec, "
                  << "Speedup vs Shared: " << std::setprecision(2) << (ref_time / avg_time) << "x\n";
    }
    
    // 测试 Hybrid
    {
        std::cout << "  Testing Hybrid (CPU+GPU)...\n";
        double total_time = 0;
        std::vector<Image> output;
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            output = Convolution::convolve_hybrid(images, kernels, num_threads);
            timer.stop();
            total_time += timer.elapsed();
        }
        double avg_time = total_time / iterations;
        double mse = calculateMSE(reference, output);
        results.push_back({"Hybrid", avg_time, ref_time / avg_time, mse < 1e-6, mse});
        std::cout << "    Time: " << std::fixed << std::setprecision(2) << avg_time << " sec, "
                  << "Speedup vs Shared: " << std::setprecision(2) << (ref_time / avg_time) << "x\n";
    }
    
    // 输出结果汇总
    std::cout << "\n[5/5] Results Summary\n";
    std::cout << "========================================================\n";
    std::cout << std::left << std::setw(18) << "Method" 
              << std::right << std::setw(10) << "Time(s)"
              << std::setw(12) << "vs Shared"
              << std::setw(12) << "Verified"
              << std::setw(14) << "MSE" << "\n";
    std::cout << "--------------------------------------------------------\n";
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(18) << r.name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.time
                  << std::setw(10) << std::setprecision(2) << r.speedup << "x"
                  << std::setw(12) << (r.verified ? "Yes" : "NO!")
                  << std::setw(14) << std::scientific << std::setprecision(2) << r.mse << "\n";
    }
    std::cout << "========================================================\n";
    
    // 找出最佳
    double best_time = 1e9;
    std::string best_name;
    for (const auto& r : results) {
        if (r.time < best_time && r.verified) {
            best_time = r.time;
            best_name = r.name;
        }
    }
    
    std::cout << "\nBest Method: " << best_name << " (" << std::fixed << std::setprecision(2) << best_time << " sec)\n";
    std::cout << "========================================================\n";
    
    return 0;
}
