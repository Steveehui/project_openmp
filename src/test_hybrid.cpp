/**
 * @file test_hybrid.cpp
 * @brief Quick test for Hybrid function
 */

#include "image_util.h"
#include "convolution.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "Testing Hybrid function...\n";
    
    // Match benchmark_full config
    int img_size = 4096;
    int num_images = 3;
    int num_kernels = 31;  // 3x3 to 61x61 (or however many)
    
    // Generate test images
    std::vector<Image> images = ImageUtil::generateTestImages(num_images, img_size, img_size, "random");
    
    // Generate kernels
    std::vector<Kernel> kernels;
    for (int k = 3; k <= 3 + 2 * (num_kernels - 1); k += 2) {
        kernels.push_back(KernelFactory::createGaussian(k, 1.0f));
    }
    
    std::cout << "  Images: " << num_images << " x " << img_size << "x" << img_size << "\n";
    std::cout << "  Kernels: " << kernels.size() << " (";
    for (size_t i = 0; i < kernels.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << kernels[i].size << "x" << kernels[i].size;
    }
    std::cout << ")\n";
    std::cout << "  Total tasks: " << num_images * kernels.size() << "\n\n";
    
    // Compute serial reference
    std::cout << "Computing serial reference...\n";
    std::vector<Image> serial_results;
    for (int i = 0; i < num_images; ++i) {
        for (size_t k = 0; k < kernels.size(); ++k) {
            serial_results.push_back(Convolution::convolve_serial(images[i], kernels[k]));
        }
    }
    std::cout << "  Serial done: " << serial_results.size() << " results\n\n";
    
    // Test Hybrid
    std::cout << "Running Hybrid...\n";
    std::vector<Image> hybrid_results = Convolution::convolve_hybrid(images, kernels, 16);
    std::cout << "  Hybrid done: " << hybrid_results.size() << " results\n\n";
    
    // Verify
    std::cout << "Verifying...\n";
    double max_diff = 0;
    int max_idx = -1, max_pixel = -1;
    
    for (size_t i = 0; i < serial_results.size(); ++i) {
        if (hybrid_results[i].width != serial_results[i].width ||
            hybrid_results[i].height != serial_results[i].height) {
            std::cout << "  [FAIL] Size mismatch at index " << i << "\n";
            std::cout << "    Hybrid: " << hybrid_results[i].width << "x" << hybrid_results[i].height << "\n";
            std::cout << "    Serial: " << serial_results[i].width << "x" << serial_results[i].height << "\n";
            continue;
        }
        
        for (size_t j = 0; j < serial_results[i].data.size(); ++j) {
            double diff = std::abs(hybrid_results[i].data[j] - serial_results[i].data[j]);
            if (diff > max_diff) {
                max_diff = diff;
                max_idx = (int)i;
                max_pixel = (int)j;
            }
        }
    }
    
    std::cout << "  Max diff: " << max_diff << "\n";
    if (max_diff >= 1.0 && max_idx >= 0) {
        std::cout << "  At index " << max_idx << ", pixel " << max_pixel << "\n";
        std::cout << "  Hybrid: " << hybrid_results[max_idx].data[max_pixel] 
                  << ", Serial: " << serial_results[max_idx].data[max_pixel] << "\n";
        
        // More debug info - show which image/kernel this is
        int img_idx = max_idx / num_kernels;
        int kernel_idx = max_idx % num_kernels;
        std::cout << "  This is image " << img_idx << ", kernel " << kernel_idx 
                  << " (size " << kernels[kernel_idx].size << "x" << kernels[kernel_idx].size << ")\n";
    }
    
    if (max_diff < 1.0) {
        std::cout << "\n[PASS] Hybrid verification succeeded!\n";
    } else {
        std::cout << "\n[FAIL] Hybrid verification failed!\n";
    }
    
    return (max_diff < 1.0) ? 0 : 1;
}
