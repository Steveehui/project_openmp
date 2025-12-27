/**
 * @file main_cuda.cpp
 * @brief CUDA 版本卷积测试程序 - 增强版
 * @author OpenMP/CUDA 并行计算项目
 * @date 2025-12-08
 * 
 * 测试多种 CUDA 优化策略:
 * - 朴素 CUDA 实现
 * - 共享内存优化
 * - CUDA Streams 流水线
 * - 混合调度
 */

#include "convolution.h"
#include "image_util.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>

void printGPUInfo() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "错误: 未检测到 CUDA 设备!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::cout << "检测到 " << device_count << " 个 CUDA 设备" << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "\n设备 " << i << ": " << prop.name << std::endl;
        std::cout << "  - 计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  - 全局内存: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  - SM 数量: " << prop.multiProcessorCount << std::endl;
        std::cout << "  - 共享内存/块: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "  - 最大线程块: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  - 内存带宽: " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
    }
    std::cout << std::endl;
}

double computeGFLOPs(int img_width, int img_height, int kernel_size, double time_seconds) {
    long long flops = 2LL * (img_width - kernel_size + 1) * (img_height - kernel_size + 1) * kernel_size * kernel_size;
    return flops / time_seconds / 1e9;
}

int main(int argc, char** argv) {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       图像卷积 - CUDA 加速测试 (多策略对比)                  ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    // 打印 GPU 信息
    printGPUInfo();
    
    // 默认参数
    int img_size = 2048;
    int num_images = 2;
    int num_streams = 4;
    bool use_large_kernels = false;  // 使用标准核集合避免超大核导致问题
    
    if (argc > 1) img_size = std::atoi(argv[1]);
    if (argc > 2) num_images = std::atoi(argv[2]);
    if (argc > 3) num_streams = std::atoi(argv[3]);
    
    std::cout << "测试配置:" << std::endl;
    std::cout << "  - 图像尺寸: " << img_size << " x " << img_size << std::endl;
    std::cout << "  - 图像数量: " << num_images << std::endl;
    std::cout << "  - CUDA Streams: " << num_streams << std::endl;
    
    // ===== 步骤 1: 生成测试数据 =====
    std::cout << "\n[1/6] 生成测试图像..." << std::endl;
    std::vector<Image> images;
    std::vector<std::string> patterns = {"random", "gradient", "checkerboard"};
    
    for (int i = 0; i < num_images; ++i) {
        std::string pattern = patterns[i % patterns.size()];
        images.push_back(ImageUtil::generateTestImage(img_size, img_size, pattern));
        std::cout << "  - 图像 " << (i + 1) << ": " << pattern << std::endl;
    }
    
    // ===== 步骤 2: 创建卷积核 =====
    std::cout << "\n[2/6] 创建卷积核集合..." << std::endl;
    std::vector<Kernel> kernels = use_large_kernels ? 
        KernelFactory::getLargeKernelSet() : 
        KernelFactory::getStandardKernelSet();
    
    int total_tasks = num_images * kernels.size();
    std::cout << "  - 卷积核数量: " << kernels.size() << std::endl;
    std::cout << "  - 总任务数: " << total_tasks << std::endl;
    
    // 显示卷积核大小分布
    std::cout << "  - 卷积核大小: ";
    for (size_t i = 0; i < kernels.size() && i < 10; ++i) {
        std::cout << kernels[i].size << "x" << kernels[i].size;
        if (i < 9 && i < kernels.size() - 1) std::cout << ", ";
    }
    if (kernels.size() > 10) std::cout << " ... (共" << kernels.size() << "个)";
    std::cout << std::endl;
    std::cout.flush();
    
    // ===== 步骤 3: 预热 GPU =====
    std::cout << "\n[3/6] 预热 GPU..."; std::cout.flush();
    for (int i = 0; i < 3; ++i) {
        std::cout << " " << (i+1); std::cout.flush();
        Image warmup = Convolution::convolve_cuda(images[0], kernels[0]);
        cudaDeviceSynchronize();
    }
    std::cout << " 完成" << std::endl; std::cout.flush();
    
    // ===== 步骤 4: 串行基准测试(单个卷积用于验证) =====
    std::cout << "\n[4/6] CPU 串行基准测试..." << std::endl;
    Timer timer;
    timer.start();
    Image serial_result = Convolution::convolve_serial(images[0], kernels[0]);
    timer.stop();
    double serial_single_time = timer.elapsed();
    std::cout << "  - 单次串行卷积: " << std::fixed << std::setprecision(4) 
              << serial_single_time << " 秒" << std::endl;
    
    // ===== 步骤 5: CUDA 优化策略对比 =====
    std::cout << "\n[5/6] CUDA 优化策略对比测试..." << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::left << std::setw(25) << "策略"
              << std::setw(15) << "时间(秒)"
              << std::setw(15) << "加速比"
              << std::setw(12) << "GFLOP/s"
              << "状态" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    struct TestResult {
        std::string name;
        double time;
        double speedup;
        double gflops;
        bool verified;
    };
    std::vector<TestResult> results;
    
    // 估算串行总时间
    double estimated_serial_total = serial_single_time * total_tasks;
    std::cout << "  预估串行总时间: " << std::fixed << std::setprecision(2) 
              << estimated_serial_total << " 秒" << std::endl << std::endl;
    
    // 策略 1: 朴素 CUDA (逐个处理)
    {
        timer.start();
        std::vector<Image> cuda_results;
        for (const auto& img : images) {
            for (const auto& kernel : kernels) {
                cuda_results.push_back(Convolution::convolve_cuda(img, kernel));
            }
        }
        cudaDeviceSynchronize();
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        double gflops = 0;
        for (const auto& k : kernels) {
            gflops += computeGFLOPs(img_size, img_size, k.size, elapsed / total_tasks);
        }
        gflops /= kernels.size();
        
        // 验证第一个结果
        double mse = Convolution::computeMSE(cuda_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::cout << std::left << std::setw(25) << "CUDA_Naive"
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(2) << gflops
                  << (verified ? "✓ 验证通过" : "✗ 验证失败") << std::endl;
        
        results.push_back({"CUDA_Naive", elapsed, speedup, gflops, verified});
    }
    
    // 策略 2: 共享内存优化
    {
        timer.start();
        std::vector<Image> cuda_shared_results;
        for (const auto& img : images) {
            for (const auto& kernel : kernels) {
                cuda_shared_results.push_back(Convolution::convolve_cuda_shared(img, kernel));
            }
        }
        cudaDeviceSynchronize();
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        double gflops = 0;
        for (const auto& k : kernels) {
            gflops += computeGFLOPs(img_size, img_size, k.size, elapsed / total_tasks);
        }
        gflops /= kernels.size();
        
        double mse = Convolution::computeMSE(cuda_shared_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::cout << std::left << std::setw(25) << "CUDA_Shared"
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(2) << gflops
                  << (verified ? "✓ 验证通过" : "✗ 验证失败") << std::endl;
        
        results.push_back({"CUDA_Shared", elapsed, speedup, gflops, verified});
    }
    
    // 策略 3: CUDA Streams 流水线
    {
        timer.start();
        std::vector<Image> stream_results = Convolution::convolve_batch_cuda_streams(
            images, kernels, num_streams);
        cudaDeviceSynchronize();
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        double gflops = 0;
        for (const auto& k : kernels) {
            gflops += computeGFLOPs(img_size, img_size, k.size, elapsed / total_tasks);
        }
        gflops /= kernels.size();
        
        double mse = Convolution::computeMSE(stream_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::cout << std::left << std::setw(25) << ("CUDA_Streams_" + std::to_string(num_streams))
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(2) << gflops
                  << (verified ? "✓ 验证通过" : "✗ 验证失败") << std::endl;
        
        results.push_back({"CUDA_Streams", elapsed, speedup, gflops, verified});
    }
    
    // 策略 4: 混合 OpenMP + CUDA
    {
        int omp_threads = 4;
        timer.start();
        std::vector<Image> hybrid_results = Convolution::convolve_hybrid(
            images, kernels, omp_threads);
        cudaDeviceSynchronize();
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        double gflops = 0;
        for (const auto& k : kernels) {
            gflops += computeGFLOPs(img_size, img_size, k.size, elapsed / total_tasks);
        }
        gflops /= kernels.size();
        
        double mse = Convolution::computeMSE(hybrid_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::cout << std::left << std::setw(25) << ("Hybrid_OMP" + std::to_string(omp_threads) + "_CUDA")
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(2) << gflops
                  << (verified ? "✓ 验证通过" : "✗ 验证失败") << std::endl;
        
        results.push_back({"Hybrid_OMP_CUDA", elapsed, speedup, gflops, verified});
    }
    
    std::cout << std::string(75, '-') << std::endl;
    
    // ===== 步骤 6: 结果汇总 =====
    std::cout << "\n[6/6] 结果汇总" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 找出最佳策略
    auto best = std::min_element(results.begin(), results.end(),
        [](const TestResult& a, const TestResult& b) { return a.time < b.time; });
    
    std::cout << "最佳 CUDA 策略: " << best->name << std::endl;
    std::cout << "  - 执行时间: " << std::fixed << std::setprecision(4) << best->time << " 秒" << std::endl;
    std::cout << "  - 相比串行加速: " << std::setprecision(2) << best->speedup << "x" << std::endl;
    std::cout << "  - 计算性能: " << best->gflops << " GFLOP/s" << std::endl;
    
    // 带宽利用率估算
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double peak_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6; // GB/s
    double achieved_bandwidth = (double)(img_size * img_size * sizeof(float) * 2 * total_tasks) / best->time / 1e9;
    std::cout << "  - 内存带宽: " << std::setprecision(2) << achieved_bandwidth 
              << " GB/s (峰值 " << peak_bandwidth << " GB/s)" << std::endl;
    
    // 性能对比表格
    std::cout << "\n策略性能排名:" << std::endl;
    std::vector<TestResult> sorted_results = results;
    std::sort(sorted_results.begin(), sorted_results.end(),
        [](const TestResult& a, const TestResult& b) { return a.time < b.time; });
    
    for (size_t i = 0; i < sorted_results.size(); ++i) {
        double relative = sorted_results[i].time / sorted_results[0].time;
        std::cout << "  " << (i + 1) << ". " << std::left << std::setw(20) << sorted_results[i].name
                  << " - " << std::fixed << std::setprecision(4) << sorted_results[i].time << "s"
                  << " (相对最佳: " << std::setprecision(2) << relative << "x)" << std::endl;
    }
    
    // 输出 CSV 格式结果
    std::ofstream csv_file("cuda_benchmark_results.csv");
    if (csv_file.is_open()) {
        csv_file << "Strategy,Time_sec,Speedup,GFLOPs,Verified\n";
        for (const auto& r : results) {
            csv_file << r.name << "," << r.time << "," << r.speedup << "," 
                     << r.gflops << "," << (r.verified ? "Yes" : "No") << "\n";
        }
        csv_file.close();
        std::cout << "\n结果已保存到: cuda_benchmark_results.csv" << std::endl;
    }
    
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "测试完成!" << std::endl;
    
    return 0;
}
