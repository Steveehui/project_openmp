/**
 * @file main_hybrid.cpp
 * @brief 混合并行版本(OpenMP + CUDA)测试程序 - 增强版
 * @author OpenMP/CUDA 并行计算项目
 * @date 2025-12-08
 * 
 * 测试 CPU+GPU 混合并行策略:
 * - OpenMP 任务调度 + CUDA 执行
 * - 自适应负载均衡
 * - CPU/GPU 协作
 */

#include "convolution.h"
#include "image_util.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

void printSystemInfo() {
    // CPU 信息
    std::cout << "\n系统配置:" << std::endl;
    std::cout << "  CPU:" << std::endl;
    std::cout << "    - 最大线程数: " << omp_get_max_threads() << std::endl;
    
    // GPU 信息
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "  GPU:" << std::endl;
        std::cout << "    - 设备: " << prop.name << std::endl;
        std::cout << "    - 计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "    - 显存: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "    - SM 数量: " << prop.multiProcessorCount << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       图像卷积 - 混合并行测试 (OpenMP + CUDA)                ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    
    printSystemInfo();
    
    // 默认参数
    int img_size = 4096;
    int num_images = 3;
    int num_threads = std::min(omp_get_max_threads(), 8);  // 混合模式不需要太多线程
    bool use_large_kernels = true;
    
    if (argc > 1) img_size = std::atoi(argv[1]);
    if (argc > 2) num_images = std::atoi(argv[2]);
    if (argc > 3) num_threads = std::atoi(argv[3]);
    
    std::cout << "\n测试配置:" << std::endl;
    std::cout << "  - 图像尺寸: " << img_size << " x " << img_size << std::endl;
    std::cout << "  - 图像数量: " << num_images << std::endl;
    std::cout << "  - 调度线程: " << num_threads << std::endl;
    
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
    
    // ===== 步骤 3: 基准测试 =====
    std::cout << "\n[3/6] 基准测试 (单次串行卷积)..." << std::endl;
    Timer timer;
    timer.start();
    Image serial_result = Convolution::convolve_serial(images[0], kernels[0]);
    timer.stop();
    double serial_single_time = timer.elapsed();
    double estimated_serial_total = serial_single_time * total_tasks;
    std::cout << "  - 单次串行: " << std::fixed << std::setprecision(4) 
              << serial_single_time << " 秒" << std::endl;
    std::cout << "  - 预估串行总时间: " << estimated_serial_total << " 秒" << std::endl;
    
    // ===== 步骤 4: 策略对比测试 =====
    std::cout << "\n[4/6] 混合并行策略对比测试..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(30) << "策略"
              << std::setw(15) << "时间(秒)"
              << std::setw(15) << "vs串行"
              << std::setw(12) << "任务/秒"
              << "状态" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    struct TestResult {
        std::string name;
        double time;
        double speedup;
        double throughput;
        bool verified;
    };
    std::vector<TestResult> results;
    
    // 策略 1: 纯 OpenMP (最佳策略)
    {
        timer.start();
        std::vector<Image> omp_results = Convolution::convolve_batch_omp_balanced(
            images, kernels, omp_get_max_threads());
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        double throughput = total_tasks / elapsed;
        
        double mse = Convolution::computeMSE(omp_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::cout << std::left << std::setw(30) << "OpenMP_Only (Balanced)"
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << throughput
                  << (verified ? "✓" : "✗") << std::endl;
        
        results.push_back({"OpenMP_Balanced", elapsed, speedup, throughput, verified});
    }
    
    // 策略 2: 纯 CUDA
    {
        // 预热
        for (int i = 0; i < 3; ++i) {
            Convolution::convolve_cuda(images[0], kernels[0]);
        }
        cudaDeviceSynchronize();
        
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
        double throughput = total_tasks / elapsed;
        
        double mse = Convolution::computeMSE(cuda_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::cout << std::left << std::setw(30) << "CUDA_Only"
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << throughput
                  << (verified ? "✓" : "✗") << std::endl;
        
        results.push_back({"CUDA_Only", elapsed, speedup, throughput, verified});
    }
    
    // 策略 3: 混合 OpenMP + CUDA (OpenMP调度)
    for (int threads : {2, 4, num_threads}) {
        timer.start();
        std::vector<Image> hybrid_results = Convolution::convolve_hybrid(
            images, kernels, threads);
        cudaDeviceSynchronize();
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        double throughput = total_tasks / elapsed;
        
        double mse = Convolution::computeMSE(hybrid_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::string name = "Hybrid_OMP" + std::to_string(threads) + "_CUDA";
        std::cout << std::left << std::setw(30) << name
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << throughput
                  << (verified ? "✓" : "✗") << std::endl;
        
        results.push_back({name, elapsed, speedup, throughput, verified});
    }
    
    // 策略 4: 自适应混合 (小任务CPU,大任务GPU)
    {
        int threshold = 512;  // 512x512 以上用 GPU
        timer.start();
        std::vector<Image> adaptive_results = Convolution::convolve_adaptive(
            images, kernels, omp_get_max_threads(), threshold);
        cudaDeviceSynchronize();
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        double throughput = total_tasks / elapsed;
        
        double mse = Convolution::computeMSE(adaptive_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::cout << std::left << std::setw(30) << "Adaptive (GPU>=512)"
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << throughput
                  << (verified ? "✓" : "✗") << std::endl;
        
        results.push_back({"Adaptive", elapsed, speedup, throughput, verified});
    }
    
    // 策略 5: CUDA Streams
    {
        int num_streams = 4;
        timer.start();
        std::vector<Image> stream_results = Convolution::convolve_batch_cuda_streams(
            images, kernels, num_streams);
        cudaDeviceSynchronize();
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        double throughput = total_tasks / elapsed;
        
        double mse = Convolution::computeMSE(stream_results[0], serial_result);
        bool verified = mse < 1e-6;
        
        std::cout << std::left << std::setw(30) << "CUDA_Streams_4"
                  << std::setw(15) << std::fixed << std::setprecision(4) << elapsed
                  << std::setw(15) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << throughput
                  << (verified ? "✓" : "✗") << std::endl;
        
        results.push_back({"CUDA_Streams", elapsed, speedup, throughput, verified});
    }
    
    std::cout << std::string(80, '-') << std::endl;
    
    // ===== 步骤 5: CPU/GPU 协作测试 =====
    std::cout << "\n[5/6] CPU/GPU 协作模式测试..." << std::endl;
    
    // 测试不同的 CPU/GPU 任务分配比例
    std::cout << "  任务分配比例测试 (CPU% : GPU%):" << std::endl;
    
    std::vector<std::pair<double, double>> split_results;
    for (int cpu_pct : {0, 25, 50, 75, 100}) {
        int cpu_tasks = total_tasks * cpu_pct / 100;
        int gpu_tasks = total_tasks - cpu_tasks;
        
        timer.start();
        
        std::vector<Image> combined_results(total_tasks);
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // CPU 处理部分任务
                for (int i = 0; i < cpu_tasks; ++i) {
                    int img_idx = i / kernels.size();
                    int kernel_idx = i % kernels.size();
                    combined_results[i] = Convolution::convolve_serial(
                        images[img_idx], kernels[kernel_idx]);
                }
            }
            #pragma omp section
            {
                // GPU 处理部分任务
                for (int i = cpu_tasks; i < total_tasks; ++i) {
                    int img_idx = i / kernels.size();
                    int kernel_idx = i % kernels.size();
                    combined_results[i] = Convolution::convolve_cuda(
                        images[img_idx], kernels[kernel_idx]);
                }
            }
        }
        
        cudaDeviceSynchronize();
        timer.stop();
        
        double elapsed = timer.elapsed();
        double speedup = estimated_serial_total / elapsed;
        split_results.push_back({cpu_pct, elapsed});
        
        std::cout << "    " << std::setw(3) << cpu_pct << "% : " << std::setw(3) << (100-cpu_pct) << "%"
                  << "  -> " << std::fixed << std::setprecision(4) << elapsed << " 秒"
                  << " (加速 " << std::setprecision(2) << speedup << "x)" << std::endl;
    }
    
    // 找出最佳分配比例
    auto best_split = std::min_element(split_results.begin(), split_results.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    std::cout << "  最佳分配: CPU " << (int)best_split->first << "% : GPU " 
              << (int)(100-best_split->first) << "%" << std::endl;
    
    // ===== 步骤 6: 结果汇总 =====
    std::cout << "\n[6/6] 结果汇总" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 找出总体最佳策略
    auto best = std::min_element(results.begin(), results.end(),
        [](const TestResult& a, const TestResult& b) { return a.time < b.time; });
    
    std::cout << "总体最佳策略: " << best->name << std::endl;
    std::cout << "  - 执行时间: " << std::fixed << std::setprecision(4) << best->time << " 秒" << std::endl;
    std::cout << "  - 相比串行加速: " << std::setprecision(2) << best->speedup << "x" << std::endl;
    std::cout << "  - 吞吐量: " << std::setprecision(1) << best->throughput << " 任务/秒" << std::endl;
    
    // 性能排名
    std::cout << "\n策略性能排名:" << std::endl;
    std::vector<TestResult> sorted_results = results;
    std::sort(sorted_results.begin(), sorted_results.end(),
        [](const TestResult& a, const TestResult& b) { return a.time < b.time; });
    
    for (size_t i = 0; i < sorted_results.size(); ++i) {
        double relative = sorted_results[i].time / sorted_results[0].time;
        std::cout << "  " << (i + 1) << ". " << std::left << std::setw(25) << sorted_results[i].name
                  << " - " << std::fixed << std::setprecision(4) << sorted_results[i].time << "s"
                  << " (相对最佳: " << std::setprecision(2) << relative << "x)" << std::endl;
    }
    
    // 输出 CSV
    std::ofstream csv_file("hybrid_benchmark_results.csv");
    if (csv_file.is_open()) {
        csv_file << "Strategy,Time_sec,Speedup,Throughput,Verified\n";
        for (const auto& r : results) {
            csv_file << r.name << "," << r.time << "," << r.speedup << "," 
                     << r.throughput << "," << (r.verified ? "Yes" : "No") << "\n";
        }
        csv_file.close();
        std::cout << "\n结果已保存到: hybrid_benchmark_results.csv" << std::endl;
    }
    
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "测试完成!" << std::endl;
    
    return 0;
}
