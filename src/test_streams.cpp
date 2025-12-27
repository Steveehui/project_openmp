/**
 * @brief 测试 CUDA Streams 结果正确性
 */

#include "convolution.h"
#include "image_util.h"
#include <iostream>
#include <cmath>

int main() {
    // 小规模测试
    int image_size = 1024;
    int num_images = 2;
    
    std::cout << "Generating test data...\n";
    std::vector<Image> images = ImageUtil::generateTestImages(num_images, image_size, image_size, "random");
    
    // 使用更多的卷积核
    std::vector<Kernel> kernels = KernelFactory::getLargeKernelSet();
    
    std::cout << "Images: " << num_images << ", Kernels: " << kernels.size() << "\n";
    std::cout << "Total tasks: " << num_images * kernels.size() << "\n\n";
    
    // 输出图像数据的前几个值
    for (int i = 0; i < num_images; ++i) {
        std::cout << "Image " << i << " first 5 values: ";
        for (int j = 0; j < 5 && j < (int)images[i].data.size(); ++j) {
            std::cout << images[i].data[j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Serial 基准
    std::cout << "Computing Serial results...\n";
    std::vector<Image> serial_results;
    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        for (int k_idx = 0; k_idx < (int)kernels.size(); ++k_idx) {
            int result_idx = img_idx * kernels.size() + k_idx;
            std::cout << "  Serial task " << result_idx << ": img=" << img_idx << ", k=" << k_idx << "\n";
            serial_results.push_back(Convolution::convolve_serial(images[img_idx], kernels[k_idx]));
        }
    }
    
    // CUDA Streams
    std::cout << "\nComputing CUDA_Streams results (num_streams=4)...\n";
    std::vector<Image> streams_results = Convolution::convolve_batch_cuda_streams(images, kernels, 4);
    
    // 比较
    std::cout << "\n=== Results Comparison ===\n";
    std::cout << "Serial size: " << serial_results.size() << "\n";
    std::cout << "Streams size: " << streams_results.size() << "\n\n";
    
    // 检查每个结果
    bool all_pass = true;
    for (size_t i = 0; i < serial_results.size(); ++i) {
        int img_idx = i / kernels.size();
        int k_idx = i % kernels.size();
        
        // 检查尺寸
        if (serial_results[i].width != streams_results[i].width ||
            serial_results[i].height != streams_results[i].height) {
            std::cout << "[" << i << "] Size MISMATCH! img=" << img_idx << ", k=" << k_idx << "\n";
            std::cout << "  Serial: " << serial_results[i].width << "x" << serial_results[i].height << "\n";
            std::cout << "  Streams: " << streams_results[i].width << "x" << streams_results[i].height << "\n";
            all_pass = false;
            continue;
        }
        
        // 计算差异
        double max_diff = 0;
        for (size_t j = 0; j < serial_results[i].data.size(); ++j) {
            double diff = std::abs(streams_results[i].data[j] - serial_results[i].data[j]);
            max_diff = std::max(max_diff, diff);
        }
        
        bool pass = max_diff < 1.0;
        std::cout << "[" << i << "] img=" << img_idx << ", k=" << k_idx 
                  << " (size=" << kernels[k_idx].size << "): max_diff=" << max_diff 
                  << " " << (pass ? "OK" : "FAIL") << "\n";
        
        if (!pass) {
            all_pass = false;
            std::cout << "  First 5 values: Serial vs Streams\n";
            for (int j = 0; j < 5 && j < (int)serial_results[i].data.size(); ++j) {
                std::cout << "    [" << j << "] " << serial_results[i].data[j] 
                          << " vs " << streams_results[i].data[j] << "\n";
            }
        }
    }
    
    std::cout << "\nOverall: " << (all_pass ? "PASS" : "FAIL") << "\n";
    
    return all_pass ? 0 : 1;
}
