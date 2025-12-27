/**
 * @file main_omp.cpp
 * @brief OpenMP 并行版本卷积测试程序 - 多种优化策略对比
 * @author 高性能计算课程项目
 * @date 2025-12-10
 * 
 * 该程序测试多种OpenMP优化策略:
 * 1. 基础并行版本
 * 2. 循环分块优化
 * 3. SIMD向量化
 * 4. 任务级并行(批处理)
 * 5. 负载均衡优化
 */

#include "convolution.h"
#include "image_util.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <fstream>
#include <numeric>

// 测试配置
struct TestConfig {
    int image_size;
    int num_images;
    bool use_large_kernels;
    int iterations;
    int num_threads;
};

// 性能结果
struct PerfResult {
    std::string method;
    double time;
    double speedup;
    bool verified;
};

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Image Convolution - OpenMP Version" << std::endl;
    std::cout << "  (Multiple Optimization Strategies)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 默认配置
    TestConfig config;
    config.image_size = 4096;
    config.num_images = 3;
    config.use_large_kernels = true;
    config.iterations = 2;
    config.num_threads = omp_get_max_threads();
    
    std::string pattern = "random";
    std::string mode = "full";
    
    // 解析命令行参数
    if (argc > 1) config.image_size = std::atoi(argv[1]);
    if (argc > 2) config.num_images = std::atoi(argv[2]);
    if (argc > 3) config.num_threads = std::atoi(argv[3]);
    if (argc > 4) pattern = argv[4];
    if (argc > 5) mode = argv[5];
    
    if (mode == "quick") {
        config.image_size = 2048;
        config.num_images = 1;
        config.iterations = 1;
        config.use_large_kernels = false;
    }
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Image Size: " << config.image_size << "x" << config.image_size << std::endl;
    std::cout << "  Number of Images: " << config.num_images << std::endl;
    std::cout << "  Threads: " << config.num_threads << " (max: " << omp_get_max_threads() << ")" << std::endl;
    std::cout << "  Pattern: " << pattern << std::endl;
    std::cout << "  Mode: " << mode << std::endl;
    std::cout << "  Iterations: " << config.iterations << std::endl;
    
    // ========== 阶段1: 生成测试数据 ==========
    std::cout << "\n[1/6] Generating test images..." << std::endl;
    Timer timer;
    timer.start();
    
    std::vector<Image> images = ImageUtil::generateTestImages(
        config.num_images, config.image_size, config.image_size, pattern);
    
    timer.stop();
    std::cout << "  Generated " << images.size() << " images in " 
              << std::fixed << std::setprecision(2) << timer.elapsed() << " sec" << std::endl;
    
    // ========== 阶段2: 创建卷积核 ==========
    std::cout << "\n[2/6] Creating convolution kernels..." << std::endl;
    
    std::vector<Kernel> kernels;
    if (config.use_large_kernels) {
        kernels = KernelFactory::getLargeKernelSet();
    } else {
        kernels = KernelFactory::getStandardKernelSet();
    }
    
    std::cout << "  Total kernels: " << kernels.size() << std::endl;
    int total_kernel_ops = 0;
    for (const auto& k : kernels) {
        total_kernel_ops += k.size * k.size;
    }
    
    // 计算任务量
    int total_tasks = config.num_images * kernels.size() * config.iterations;
    long long total_ops = (long long)config.image_size * config.image_size * 
                          config.num_images * total_kernel_ops * config.iterations;
    
    std::cout << "  Total tasks: " << total_tasks << std::endl;
    std::cout << "  Estimated FLOPs: " << total_ops / 1e9 << " GFLOP" << std::endl;
    
    // ========== 阶段3: 串行基准测试 ==========
    std::cout << "\n[3/6] Running serial baseline..." << std::endl;
    
    double serial_time = 0;
    std::vector<Image> serial_results;
    
    {
        Timer serial_timer;
        serial_timer.start();
        
        for (int iter = 0; iter < config.iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_serial(img, kernel);
                    if (iter == config.iterations - 1) {
                        serial_results.push_back(std::move(result));
                    }
                }
            }
        }
        
        serial_timer.stop();
        serial_time = serial_timer.elapsed();
        
        std::cout << "  Serial time: " << std::setprecision(2) << serial_time << " sec" << std::endl;
    }
    
    // ========== 阶段4: OpenMP 优化策略测试 ==========
    std::cout << "\n[4/6] Testing OpenMP optimization strategies..." << std::endl;
    
    std::vector<PerfResult> results;
    
    // 方法1: 基础并行版本
    std::cout << "\n  [Method 1] Basic OpenMP parallel..." << std::endl;
    {
        Timer t;
        t.start();
        
        std::vector<Image> omp_results;
        for (int iter = 0; iter < config.iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_omp(img, kernel, config.num_threads);
                    if (iter == config.iterations - 1) {
                        omp_results.push_back(std::move(result));
                    }
                }
            }
        }
        
        t.stop();
        double time = t.elapsed();
        
        // 验证正确性
        bool verified = true;
        if (!serial_results.empty() && !omp_results.empty()) {
            verified = Convolution::verifyResults(serial_results[0], omp_results[0], 1.0f);
        }
        
        results.push_back({"OMP_Basic", time, serial_time / time, verified});
        std::cout << "    Time: " << std::setprecision(2) << time << " sec"
                  << " | Speedup: " << std::setprecision(2) << serial_time / time << "x" << std::endl;
    }
    
    // 方法2: 循环分块优化
    std::cout << "\n  [Method 2] OMP with cache blocking..." << std::endl;
    {
        int block_size = Convolution::getOptimalBlockSize(config.image_size, kernels[0].size);
        std::cout << "    Block size: " << block_size << std::endl;
        
        Timer t;
        t.start();
        
        std::vector<Image> blocked_results;
        for (int iter = 0; iter < config.iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_omp_blocked(img, kernel, config.num_threads, block_size);
                    if (iter == config.iterations - 1) {
                        blocked_results.push_back(std::move(result));
                    }
                }
            }
        }
        
        t.stop();
        double time = t.elapsed();
        
        bool verified = true;
        if (!serial_results.empty() && !blocked_results.empty()) {
            verified = Convolution::verifyResults(serial_results[0], blocked_results[0], 1.0f);
        }
        
        results.push_back({"OMP_Blocked", time, serial_time / time, verified});
        std::cout << "    Time: " << std::setprecision(2) << time << " sec"
                  << " | Speedup: " << std::setprecision(2) << serial_time / time << "x" << std::endl;
    }
    
    // 方法3: SIMD向量化
    std::cout << "\n  [Method 3] OMP with SIMD..." << std::endl;
    {
        Timer t;
        t.start();
        
        std::vector<Image> simd_results;
        for (int iter = 0; iter < config.iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_omp_simd(img, kernel, config.num_threads);
                    if (iter == config.iterations - 1) {
                        simd_results.push_back(std::move(result));
                    }
                }
            }
        }
        
        t.stop();
        double time = t.elapsed();
        
        bool verified = true;
        if (!serial_results.empty() && !simd_results.empty()) {
            verified = Convolution::verifyResults(serial_results[0], simd_results[0], 1.0f);
        }
        
        results.push_back({"OMP_SIMD", time, serial_time / time, verified});
        std::cout << "    Time: " << std::setprecision(2) << time << " sec"
                  << " | Speedup: " << std::setprecision(2) << serial_time / time << "x" << std::endl;
    }
    
    // 方法4: 任务级并行(批处理)
    std::cout << "\n  [Method 4] OMP batch processing (task parallel)..." << std::endl;
    {
        Timer t;
        t.start();
        
        std::vector<Image> batch_results;
        for (int iter = 0; iter < config.iterations; ++iter) {
            auto results_iter = Convolution::convolve_batch_omp(images, kernels, config.num_threads);
            if (iter == config.iterations - 1) {
                batch_results = std::move(results_iter);
            }
        }
        
        t.stop();
        double time = t.elapsed();
        
        bool verified = true;
        if (!serial_results.empty() && !batch_results.empty()) {
            verified = Convolution::verifyResults(serial_results[0], batch_results[0], 1.0f);
        }
        
        results.push_back({"OMP_Batch", time, serial_time / time, verified});
        std::cout << "    Time: " << std::setprecision(2) << time << " sec"
                  << " | Speedup: " << std::setprecision(2) << serial_time / time << "x" << std::endl;
    }
    
    // 方法5: 负载均衡批处理
    std::cout << "\n  [Method 5] OMP batch with load balancing..." << std::endl;
    {
        Timer t;
        t.start();
        
        std::vector<Image> balanced_results;
        for (int iter = 0; iter < config.iterations; ++iter) {
            auto results_iter = Convolution::convolve_batch_omp_balanced(images, kernels, config.num_threads);
            if (iter == config.iterations - 1) {
                balanced_results = std::move(results_iter);
            }
        }
        
        t.stop();
        double time = t.elapsed();
        
        bool verified = true;
        if (!serial_results.empty() && !balanced_results.empty()) {
            verified = Convolution::verifyResults(serial_results[0], balanced_results[0], 1.0f);
        }
        
        results.push_back({"OMP_Balanced", time, serial_time / time, verified});
        std::cout << "    Time: " << std::setprecision(2) << time << " sec"
                  << " | Speedup: " << std::setprecision(2) << serial_time / time << "x" << std::endl;
    }
    
    // ========== 阶段5: 线程扩展性测试 ==========
    std::cout << "\n[5/6] Thread scalability test..." << std::endl;
    
    int max_threads = omp_get_max_threads();
    std::vector<std::pair<int, double>> scalability;
    
    // 使用单张图像和标准卷积核进行扩展性测试
    Image test_img = images[0];
    Kernel test_kernel = kernels[0];
    
    for (int threads = 1; threads <= max_threads; threads *= 2) {
        Timer t;
        t.start();
        
        for (int i = 0; i < 3; ++i) {  // 3次取平均
            Convolution::convolve_omp(test_img, test_kernel, threads);
        }
        
        t.stop();
        double avg_time = t.elapsed() / 3.0;
        scalability.push_back({threads, avg_time});
        
        std::cout << "  " << std::setw(2) << threads << " threads: " 
                  << std::setprecision(4) << avg_time << " sec" << std::endl;
    }
    
    // ========== 阶段6: 结果汇总 ==========
    std::cout << "\n[6/6] Results Summary:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nPerformance Comparison:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::left << std::setw(20) << "Method" 
              << std::setw(12) << "Time(s)" 
              << std::setw(12) << "Speedup" 
              << std::setw(10) << "Verified" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::setw(20) << "Serial" 
              << std::setw(12) << std::fixed << std::setprecision(2) << serial_time 
              << std::setw(12) << "1.00x" 
              << std::setw(10) << "N/A" << std::endl;
    
    for (const auto& r : results) {
        std::cout << std::setw(20) << r.method 
                  << std::setw(12) << std::setprecision(2) << r.time 
                  << std::setw(12) << std::setprecision(2) << r.speedup << "x" 
                  << std::setw(10) << (r.verified ? "Yes" : "No") << std::endl;
    }
    
    // 找出最佳方法
    auto best = std::max_element(results.begin(), results.end(),
        [](const PerfResult& a, const PerfResult& b) { return a.speedup < b.speedup; });
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Best method: " << best->method << " (" << std::setprecision(2) 
              << best->speedup << "x speedup)" << std::endl;
    
    // 计算效率
    double efficiency = best->speedup / config.num_threads * 100;
    std::cout << "Parallel efficiency: " << std::setprecision(1) << efficiency << "%" << std::endl;
    
    // 性能指标
    double pixels_per_sec = (double)config.image_size * config.image_size * 
                            config.num_images * kernels.size() * config.iterations / best->time;
    double gflops = total_ops / best->time / 1e9;
    
    std::cout << "\nPerformance Metrics (Best Method):" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(2) << pixels_per_sec / 1e6 << " MP/s" << std::endl;
    std::cout << "  GFLOP/s: " << std::setprecision(3) << gflops << std::endl;
    
    std::cout << "========================================" << std::endl;
    
    // CSV格式输出
    std::cout << "\n[CSV] omp," << config.image_size << "," << config.num_images << ","
              << kernels.size() << "," << config.num_threads << ","
              << std::setprecision(4) << best->time << ","
              << std::setprecision(2) << best->speedup << ","
              << std::setprecision(3) << gflops << std::endl;
    
    return 0;
}
