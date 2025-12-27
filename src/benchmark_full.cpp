/**
 * @file benchmark_full.cpp
 * @brief Full performance benchmark - Serial ~5 minutes baseline
 * 
 * Test configuration:
 * - Image size: 4096 x 4096
 * - Number of images: 3
 * - Kernels: 31 types (3x3 to 31x31)
 * - Iterations: 2
 */

#include "image_util.h"
#include "convolution.h"
#include "timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <cmath>
#include <omp.h>

// Test result structure
struct TestResult {
    std::string name;
    double time;
    double speedup;
    std::string type;
    bool verified;
};

// Verify results
bool verifyResults(const std::vector<Image>& results, const std::vector<Image>& reference, bool verbose = false) {
    if (results.size() != reference.size()) {
        if (verbose) std::cout << "  Size mismatch: " << results.size() << " vs " << reference.size() << "\n";
        return false;
    }
    
    double max_diff = 0;
    int max_diff_idx = -1;
    int max_diff_pixel = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].width != reference[i].width || 
            results[i].height != reference[i].height) {
            if (verbose) std::cout << "  Dim mismatch at " << i << ": " 
                << results[i].width << "x" << results[i].height << " vs "
                << reference[i].width << "x" << reference[i].height << "\n";
            return false;
        }
        for (size_t j = 0; j < results[i].data.size(); ++j) {
            double diff = std::abs(results[i].data[j] - reference[i].data[j]);
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_idx = (int)i;
                max_diff_pixel = (int)j;
            }
        }
    }
    if (verbose && max_diff >= 1.0) {
        std::cout << "Max diff " << max_diff << " at index " << max_diff_idx 
                  << ", pixel " << max_diff_pixel << "\n";
        if (max_diff_idx >= 0 && max_diff_idx < (int)results.size()) {
            std::cout << "    Values: result=" << results[max_diff_idx].data[max_diff_pixel] 
                << " ref=" << reference[max_diff_idx].data[max_diff_pixel] << "\n";
            // Print image dimensions
            std::cout << "    Image size: " << results[max_diff_idx].width << "x" 
                      << results[max_diff_idx].height << " (" 
                      << results[max_diff_idx].data.size() << " pixels)\n";
        }
    }
    return max_diff < 1.0;
}

// Calculate MSE
double calculateMSE(const std::vector<Image>& results, const std::vector<Image>& reference) {
    if (results.size() != reference.size()) return -1;
    
    double mse = 0;
    size_t total = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        for (size_t j = 0; j < results[i].data.size(); ++j) {
            double diff = results[i].data[j] - reference[i].data[j];
            mse += diff * diff;
            total++;
        }
    }
    return mse / total;
}

int main(int argc, char* argv[]) {
    // Default configuration - target: serial ~5 minutes
    int image_size = 4096;
    int num_images = 3;
    int num_threads = 16;
    int iterations = 2;
    
    // Parse command line arguments
    if (argc >= 2) image_size = std::atoi(argv[1]);
    if (argc >= 3) num_images = std::atoi(argv[2]);
    if (argc >= 4) num_threads = std::atoi(argv[3]);
    if (argc >= 5) iterations = std::atoi(argv[4]);
    
    std::cout << "========================================================\n";
    std::cout << "  Image Convolution - Full Benchmark\n";
    std::cout << "  (Serial / OpenMP / CUDA / Hybrid)\n";
    std::cout << "========================================================\n\n";
    
    // System info
    std::cout << "====== System Configuration ======\n";
    std::cout << "CPU Threads Available: " << omp_get_max_threads() << "\n";
    
    std::cout << "\nTest Configuration:\n";
    std::cout << "  Image Size: " << image_size << " x " << image_size << "\n";
    std::cout << "  Number of Images: " << num_images << "\n";
    std::cout << "  CPU Threads: " << num_threads << "\n";
    std::cout << "  Iterations: " << iterations << "\n";
    
    omp_set_num_threads(num_threads);
    
    // ========== Phase 1: Generate test images ==========
    std::cout << "\n[1/6] Generating test images...\n";
    Timer timer;
    timer.start();
    
    std::vector<Image> images = ImageUtil::generateTestImages(num_images, image_size, image_size, "random");
    
    timer.stop();
    std::cout << "  Done in " << std::fixed << std::setprecision(2) 
              << timer.elapsed() << " sec\n";
    
    // ========== Phase 2: Create convolution kernels ==========
    std::cout << "\n[2/6] Creating convolution kernels (Large Set)...\n";
    
    std::vector<Kernel> kernels = KernelFactory::getLargeKernelSet();
    
    std::cout << "  Total kernels: " << kernels.size() << "\n";
    
    // Calculate kernel statistics
    long long total_kernel_ops = 0;
    int min_size = 999, max_size = 0;
    for (const auto& k : kernels) {
        total_kernel_ops += k.size * k.size;
        min_size = std::min(min_size, k.size);
        max_size = std::max(max_size, k.size);
    }
    
    std::cout << "  Kernel sizes: " << min_size << "x" << min_size 
              << " to " << max_size << "x" << max_size << "\n";
    
    // Calculate workload
    int total_tasks = num_images * (int)kernels.size() * iterations;
    long long total_flops = (long long)image_size * image_size * 
                            num_images * total_kernel_ops * iterations * 2;
    
    std::cout << "  Total tasks: " << total_tasks << " convolutions\n";
    std::cout << "  Estimated FLOPs: " << std::fixed << std::setprecision(2) 
              << total_flops / 1e9 << " GFLOP\n";
    
    // Estimate time
    double estimated_serial_time = total_flops / 3e9;
    std::cout << "  Estimated serial time: ~" << (int)(estimated_serial_time / 60) 
              << " min " << (int)((int)estimated_serial_time % 60) << " sec\n";
    
    std::vector<TestResult> results;
    std::vector<Image> serial_results;
    double serial_time = 0;
    
    // ========== Phase 3: Serial baseline test ==========
    std::cout << "\n[3/6] Running SERIAL baseline...\n";
    std::cout << "  (This will take approximately " << (int)(estimated_serial_time / 60) 
              << " minutes)\n";
    std::cout << "  Progress: " << std::flush;
    
    timer.start();
    
    int progress = 0;
    int total_progress = num_images * (int)kernels.size() * iterations;
    int last_percent = -1;
    
    for (int iter = 0; iter < iterations; ++iter) {
        for (const auto& img : images) {
            for (const auto& kernel : kernels) {
                Image result = Convolution::convolve_serial(img, kernel);
                if (iter == 0) {
                    serial_results.push_back(std::move(result));
                }
                
                progress++;
                int percent = progress * 100 / total_progress;
                if (percent != last_percent && percent % 5 == 0) {
                    std::cout << percent << "% " << std::flush;
                    last_percent = percent;
                }
            }
        }
    }
    
    timer.stop();
    serial_time = timer.elapsed();
    std::cout << "\n  Serial time: " << std::fixed << std::setprecision(2) 
              << serial_time << " sec (" << serial_time / 60 << " min)\n";
    
    results.push_back({"Serial", serial_time, 1.0, "CPU-Single", true});
    
    // ========== Phase 4: OpenMP tests ==========
    std::cout << "\n[4/6] Testing OpenMP optimizations...\n";
    
    // OMP Basic
    {
        std::cout << "  [OMP_Basic] " << std::flush;
        std::vector<Image> omp_results;
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_omp(img, kernel, num_threads);
                    if (iter == 0) omp_results.push_back(std::move(result));
                }
            }
        }
        timer.stop();
        
        double t = timer.elapsed();
        bool verified = verifyResults(omp_results, serial_results);
        std::cout << std::fixed << std::setprecision(2) << t << " sec, Speedup: " 
                  << serial_time / t << "x" << (verified ? " [OK]" : " [FAIL]") << "\n";
        results.push_back({"OMP_Basic", t, serial_time / t, "CPU-Multi", verified});
    }
    
    // OMP Blocked
    {
        std::cout << "  [OMP_Blocked] " << std::flush;
        std::vector<Image> omp_results;
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_omp_blocked(img, kernel, num_threads, 64);
                    if (iter == 0) omp_results.push_back(std::move(result));
                }
            }
        }
        timer.stop();
        
        double t = timer.elapsed();
        bool verified = verifyResults(omp_results, serial_results);
        std::cout << std::fixed << std::setprecision(2) << t << " sec, Speedup: " 
                  << serial_time / t << "x" << (verified ? " [OK]" : " [FAIL]") << "\n";
        results.push_back({"OMP_Blocked", t, serial_time / t, "CPU-Multi", verified});
    }
    
    // OMP SIMD
    {
        std::cout << "  [OMP_SIMD] " << std::flush;
        std::vector<Image> omp_results;
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_omp_simd(img, kernel, num_threads);
                    if (iter == 0) omp_results.push_back(std::move(result));
                }
            }
        }
        timer.stop();
        
        double t = timer.elapsed();
        bool verified = verifyResults(omp_results, serial_results);
        std::cout << std::fixed << std::setprecision(2) << t << " sec, Speedup: " 
                  << serial_time / t << "x" << (verified ? " [OK]" : " [FAIL]") << "\n";
        results.push_back({"OMP_SIMD", t, serial_time / t, "CPU-Multi", verified});
    }
    
    // ========== Phase 5: CUDA tests ==========
    std::cout << "\n[5/6] Testing CUDA implementations...\n";
    
#ifdef USE_CUDA
    // CUDA Naive
    {
        std::cout << "  [CUDA_Naive] " << std::flush;
        std::vector<Image> cuda_results;
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_cuda(img, kernel);
                    if (iter == 0) cuda_results.push_back(std::move(result));
                }
            }
        }
        timer.stop();
        
        double t = timer.elapsed();
        bool verified = verifyResults(cuda_results, serial_results);
        double mse = calculateMSE(cuda_results, serial_results);
        std::cout << std::fixed << std::setprecision(2) << t << " sec, Speedup: " 
                  << serial_time / t << "x" << (verified ? " [OK]" : " [FAIL]") 
                  << " (MSE=" << std::scientific << std::setprecision(2) << mse << ")\n";
        results.push_back({"CUDA_Naive", t, serial_time / t, "GPU", verified});
    }
    
    // CUDA Shared Memory
    {
        std::cout << "  [CUDA_Shared] " << std::flush;
        std::vector<Image> cuda_results;
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            for (const auto& img : images) {
                for (const auto& kernel : kernels) {
                    Image result = Convolution::convolve_cuda_shared(img, kernel);
                    if (iter == 0) cuda_results.push_back(std::move(result));
                }
            }
        }
        timer.stop();
        
        double t = timer.elapsed();
        bool verified = verifyResults(cuda_results, serial_results);
        std::cout << std::fixed << std::setprecision(2) << t << " sec, Speedup: " 
                  << serial_time / t << "x" << (verified ? " [OK]" : " [FAIL]") << "\n";
        results.push_back({"CUDA_Shared", t, serial_time / t, "GPU", verified});
    }

    // CUDA Best (single pipeline, streams, pre-uploaded kernels)
    {
        std::cout << "  [CUDA_Best] " << std::flush;
        std::vector<Image> cuda_results;
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            auto batch_results = Convolution::convolve_cuda_best(images, kernels, 12);
            if (iter == 0) cuda_results = std::move(batch_results);
        }
        timer.stop();
        double t = timer.elapsed();
        bool verified = verifyResults(cuda_results, serial_results);
        std::cout << std::fixed << std::setprecision(2) << t << " sec, Speedup: "
                  << serial_time / t << "x" << (verified ? " [OK]" : " [FAIL]") << "\n";
        results.push_back({"CUDA_Best", t, serial_time / t, "GPU", verified});
    }
    
    // Reset GPU state before Streams test
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
    
    // CUDA Batch (Streams)
#ifdef USE_CUDA
    // 确保前序测试的所有 GPU 工作已完成并重置状态，避免残留错误影响多流管线
    cudaDeviceSynchronize();
    cudaDeviceReset();
#endif
    {
        std::cout << "  [CUDA_Streams] " << std::flush;
        std::vector<Image> cuda_results;
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            auto batch_results = Convolution::convolve_batch_cuda_streams(images, kernels, 4);
            if (iter == 0) cuda_results = std::move(batch_results);
        }
        timer.stop();
        
        double t = timer.elapsed();
        bool verified = verifyResults(cuda_results, serial_results, true);  // verbose
        std::cout << std::fixed << std::setprecision(2) << t << " sec, Speedup: " 
                  << serial_time / t << "x" << (verified ? " [OK]" : " [FAIL]") << "\n";
        results.push_back({"CUDA_Streams", t, serial_time / t, "GPU", verified});
    }
    
    // Hybrid
#ifdef USE_CUDA
    cudaDeviceSynchronize();
    cudaDeviceReset();
#endif
    {
        std::cout << "  [Hybrid] " << std::flush;
        std::vector<Image> hybrid_results;
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            auto batch_results = Convolution::convolve_hybrid(images, kernels, num_threads);
            if (iter == 0) hybrid_results = std::move(batch_results);
        }
        timer.stop();
        
        double t = timer.elapsed();
        bool verified = verifyResults(hybrid_results, serial_results);
        std::cout << std::fixed << std::setprecision(2) << t << " sec, Speedup: " 
                  << serial_time / t << "x" << (verified ? " [OK]" : " [FAIL]") << "\n";
        results.push_back({"Hybrid", t, serial_time / t, "CPU+GPU", verified});
    }
#else
    std::cout << "  CUDA not enabled (compile with -DUSE_CUDA)\n";
#endif
    
    // ========== Phase 6: Results summary ==========
    std::cout << "\n[6/6] Results Summary\n";
    std::cout << "========================================================\n";
    std::cout << std::left << std::setw(18) << "Method" 
              << std::right << std::setw(12) << "Time(s)"
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Type"
              << std::setw(10) << "Verified" << "\n";
    std::cout << "--------------------------------------------------------\n";
    
    TestResult best_result = results[0];
    for (const auto& r : results) {
        std::cout << std::left << std::setw(18) << r.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << r.time
                  << std::setw(9) << r.speedup << "x"
                  << std::setw(12) << r.type
                  << std::setw(10) << (r.verified ? "Yes" : "No") << "\n";
        
        if (r.speedup > best_result.speedup && r.name != "Serial") {
            best_result = r;
        }
    }
    
    std::cout << "========================================================\n";
    std::cout << "\nBest Method: " << best_result.name << "\n";
    std::cout << "  Time: " << best_result.time << " sec\n";
    std::cout << "  Speedup vs Serial: " << best_result.speedup << "x\n";
    
    // Calculate performance metrics
    double throughput = (double)image_size * image_size * num_images * kernels.size() 
                        * iterations / best_result.time / 1e6;
    double gflops = (double)total_flops / best_result.time / 1e9;
    
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
              << throughput << " MP/s\n";
    std::cout << "  Performance: " << gflops << " GFLOP/s\n";
    
    // Save CSV results
    std::ofstream csv("benchmark_full_results.csv");
    if (csv.is_open()) {
        csv << "Method,Time_sec,Speedup,Type,Verified\n";
        for (const auto& r : results) {
            csv << r.name << "," << r.time << "," << r.speedup << "," 
                << r.type << "," << (r.verified ? "Yes" : "No") << "\n";
        }
        csv.close();
        std::cout << "\nResults saved to: benchmark_full_results.csv\n";
    }
    
    std::cout << "========================================================\n";
    
    return 0;
}
