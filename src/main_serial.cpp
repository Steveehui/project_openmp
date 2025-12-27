/**
 * @file main_serial.cpp
 * @brief 串行版本卷积测试程序 - 完整性能测试
 * @author 高性能计算课程项目
 * @date 2025-12-10
 * 
 * 该程序执行大规模图像卷积任务，设计运行时间约5分钟
 * 用于与并行版本进行性能对比
 */

#include "convolution.h"
#include "image_util.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

// 测试配置结构
struct TestConfig {
    int image_size;         // 图像尺寸
    int num_images;         // 图像数量
    bool use_large_kernels; // 是否使用大卷积核
    int iterations;         // 重复迭代次数
};

void printProgress(int current, int total, const std::string& task_name) {
    int percent = (current * 100) / total;
    std::cout << "\r  [" << std::setw(3) << percent << "%] " 
              << task_name << " (" << current << "/" << total << ")" << std::flush;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Image Convolution - Serial Version" << std::endl;
    std::cout << "  (Designed for ~5 min runtime)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 默认配置 - 设计约5分钟运行时间
    TestConfig config;
    config.image_size = 4096;       // 4K 图像
    config.num_images = 3;          // 3张图像
    config.use_large_kernels = true;// 使用大卷积核
    config.iterations = 2;          // 迭代2次
    
    std::string pattern = "random";
    std::string mode = "full";      // full: 完整测试, quick: 快速测试
    
    // 解析命令行参数
    if (argc > 1) config.image_size = std::atoi(argv[1]);
    if (argc > 2) config.num_images = std::atoi(argv[2]);
    if (argc > 3) pattern = argv[3];
    if (argc > 4) mode = argv[4];
    
    // 快速模式使用较小的测试配置
    if (mode == "quick") {
        config.image_size = 2048;
        config.num_images = 1;
        config.iterations = 1;
        config.use_large_kernels = false;
    }
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Image Size: " << config.image_size << "x" << config.image_size << std::endl;
    std::cout << "  Number of Images: " << config.num_images << std::endl;
    std::cout << "  Pattern: " << pattern << std::endl;
    std::cout << "  Mode: " << mode << std::endl;
    std::cout << "  Iterations: " << config.iterations << std::endl;
    
    // 估算运行时间
    long long total_pixels = (long long)config.image_size * config.image_size * config.num_images;
    std::cout << "  Total Pixels: " << total_pixels / 1000000.0 << " MP" << std::endl;
    
    // ========== 阶段1: 生成测试图像 ==========
    std::cout << "\n[1/5] Generating test images..." << std::endl;
    Timer timer;
    timer.start();
    
    std::vector<Image> images = ImageUtil::generateTestImages(
        config.num_images, config.image_size, config.image_size, pattern);
    
    timer.stop();
    std::cout << "  Generated " << images.size() << " images in " 
              << std::fixed << std::setprecision(2) << timer.elapsed() << " sec" << std::endl;
    
    // ========== 阶段2: 创建卷积核 ==========
    std::cout << "\n[2/5] Creating convolution kernels..." << std::endl;
    
    std::vector<Kernel> kernels;
    if (config.use_large_kernels) {
        // 大型卷积核集合 - 增加计算量
        kernels = KernelFactory::getLargeKernelSet();
    } else {
        // 标准卷积核集合
        kernels = KernelFactory::getStandardKernelSet();
    }
    
    // 显示卷积核信息
    std::cout << "  Total kernels: " << kernels.size() << std::endl;
    int total_kernel_ops = 0;
    for (const auto& k : kernels) {
        total_kernel_ops += k.size * k.size;
        std::cout << "    - " << k.name << " (" << k.size << "x" << k.size << ")" << std::endl;
    }
    
    // 估算总计算量
    long long total_ops = (long long)config.image_size * config.image_size * 
                          config.num_images * total_kernel_ops * config.iterations;
    std::cout << "  Estimated FLOPs: " << total_ops / 1e9 << " GFLOP" << std::endl;
    
    // ========== 阶段3: 预热 ==========
    std::cout << "\n[3/5] Warmup run..." << std::endl;
    {
        Image small_img = ImageUtil::generateTestImage(512, 512, "random");
        Kernel small_kernel = KernelFactory::createGaussian3x3();
        Convolution::convolve_serial(small_img, small_kernel);
        std::cout << "  Warmup complete." << std::endl;
    }
    
    // ========== 阶段4: 执行串行卷积 ==========
    std::cout << "\n[4/5] Executing serial convolution..." << std::endl;
    
    int total_tasks = config.num_images * kernels.size() * config.iterations;
    std::cout << "  Total tasks: " << total_tasks << std::endl;
    
    Timer main_timer;
    main_timer.start();
    
    std::vector<Image> results;
    results.reserve(config.num_images * kernels.size());
    
    int completed = 0;
    double last_report_time = 0;
    
    for (int iter = 0; iter < config.iterations; ++iter) {
        if (config.iterations > 1) {
            std::cout << "\n  Iteration " << (iter + 1) << "/" << config.iterations << ":" << std::endl;
        }
        
        for (int img_idx = 0; img_idx < config.num_images; ++img_idx) {
            for (size_t k_idx = 0; k_idx < kernels.size(); ++k_idx) {
                // 执行卷积
                timer.start();
                Image output = Convolution::convolve_serial(images[img_idx], kernels[k_idx]);
                timer.stop();
                
                // 保存最后一次迭代的结果
                if (iter == config.iterations - 1) {
                    results.push_back(std::move(output));
                }
                
                completed++;
                
                // 每5%或每30秒报告一次进度
                double elapsed = main_timer.elapsed();
                if (completed % std::max(1, total_tasks / 20) == 0 || 
                    elapsed - last_report_time > 30.0) {
                    int percent = (completed * 100) / total_tasks;
                    double eta = (elapsed / completed) * (total_tasks - completed);
                    
                    std::cout << "\r  [" << std::setw(3) << percent << "%] "
                              << "Completed " << completed << "/" << total_tasks
                              << " | Elapsed: " << std::fixed << std::setprecision(1) << elapsed << "s"
                              << " | ETA: " << eta << "s"
                              << " | Current: " << kernels[k_idx].name
                              << "        " << std::flush;
                    last_report_time = elapsed;
                }
            }
        }
    }
    
    main_timer.stop();
    double total_time = main_timer.elapsed();
    
    std::cout << std::endl;
    
    // ========== 阶段5: 结果统计 ==========
    std::cout << "\n[5/5] Results Summary:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Total Runtime:        " << std::fixed << std::setprecision(2) 
              << total_time << " seconds" << std::endl;
    std::cout << "                        " << std::setprecision(2) 
              << total_time / 60.0 << " minutes" << std::endl;
    std::cout << "  Tasks Completed:      " << total_tasks << std::endl;
    std::cout << "  Avg Time per Task:    " << std::setprecision(4) 
              << total_time / total_tasks << " sec" << std::endl;
    
    // 性能指标
    double pixels_per_sec = (double)total_pixels * kernels.size() * config.iterations / total_time;
    double gflops = total_ops / total_time / 1e9;
    
    std::cout << "  Throughput:           " << std::setprecision(2) 
              << pixels_per_sec / 1e6 << " MP/s" << std::endl;
    std::cout << "  Performance:          " << std::setprecision(3) 
              << gflops << " GFLOP/s" << std::endl;
    
    // 输出图像信息
    if (!results.empty()) {
        std::cout << "  Output Image Size:    " << results[0].width << "x" << results[0].height << std::endl;
        std::cout << "  Total Output Images:  " << results.size() << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
    
    // 保存结果(可选)
    if (argc > 5 && std::string(argv[5]) == "--save") {
        std::cout << "\nSaving results..." << std::endl;
        for (size_t i = 0; i < std::min(results.size(), (size_t)10); ++i) {
            std::string filename = "output_serial_" + std::to_string(i) + ".txt";
            ImageUtil::saveImageAsText(results[i], filename);
        }
        std::cout << "Results saved." << std::endl;
    }
    
    // 输出用于记录的CSV格式数据
    std::cout << "\n[CSV] serial," << config.image_size << "," << config.num_images << ","
              << kernels.size() << "," << config.iterations << ","
              << std::setprecision(4) << total_time << ","
              << std::setprecision(2) << pixels_per_sec / 1e6 << ","
              << std::setprecision(3) << gflops << std::endl;
    
    return 0;
}
