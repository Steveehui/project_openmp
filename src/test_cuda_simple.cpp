/**
 * @file test_cuda_simple.cpp
 * @brief 简化的 CUDA 测试程序
 */

#include "convolution.h"
#include "image_util.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

int main() {
    std::cout << "=== CUDA Simple Test ===" << std::endl;
    std::cout.flush();
    
    // 检查 CUDA 设备
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    std::cout.flush();
    
    // 生成小图像
    std::cout << "Generating 256x256 test image..." << std::endl;
    std::cout.flush();
    Image input = ImageUtil::generateTestImage(256, 256, "random");
    std::cout << "  Done. Size: " << input.width << "x" << input.height << std::endl;
    std::cout.flush();
    
    // 创建简单卷积核
    std::cout << "Creating 3x3 Sobel kernel..." << std::endl;
    std::cout.flush();
    Kernel kernel = KernelFactory::createSobelX();
    std::cout << "  Done. Kernel size: " << kernel.size << "x" << kernel.size << std::endl;
    std::cout.flush();
    
    // 测试串行版本
    std::cout << "Running serial convolution..." << std::endl;
    std::cout.flush();
    Timer timer;
    timer.start();
    Image serial_result = Convolution::convolve_serial(input, kernel);
    timer.stop();
    std::cout << "  Serial time: " << timer.elapsed() << " sec" << std::endl;
    std::cout << "  Output size: " << serial_result.width << "x" << serial_result.height << std::endl;
    std::cout.flush();
    
    // 测试 CUDA 版本
    std::cout << "Running CUDA convolution..." << std::endl;
    std::cout.flush();
    timer.start();
    Image cuda_result = Convolution::convolve_cuda(input, kernel);
    timer.stop();
    std::cout << "  CUDA time: " << timer.elapsed() << " sec" << std::endl;
    std::cout << "  Output size: " << cuda_result.width << "x" << cuda_result.height << std::endl;
    std::cout.flush();
    
    // 验证结果
    double mse = Convolution::computeMSE(cuda_result, serial_result);
    std::cout << "MSE between CUDA and Serial: " << mse << std::endl;
    
    if (mse < 1e-4) {
        std::cout << "CUDA Test PASSED!" << std::endl;
    } else {
        std::cout << "CUDA Test FAILED!" << std::endl;
    }
    
    return 0;
}
