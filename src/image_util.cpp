#include "image_util.h"
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ImageUtil {

Image generateTestImage(int width, int height, const std::string& pattern) {
    Image img(width, height);
    
    if (pattern == "random") {
        // 随机噪声图像
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 255.0f);
        
        for (int i = 0; i < width * height; ++i) {
            img.data[i] = dis(gen);
        }
    } else if (pattern == "gradient") {
        // 水平渐变图像
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                img.at(x, y) = (float)x / width * 255.0f;
            }
        }
    } else if (pattern == "checkerboard") {
        // 棋盘图案
        int block_size = std::max(width / 16, 1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool white = ((x / block_size) + (y / block_size)) % 2 == 0;
                img.at(x, y) = white ? 255.0f : 0.0f;
            }
        }
    } else if (pattern == "circle") {
        // 同心圆图案
        float cx = width / 2.0f;
        float cy = height / 2.0f;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float dx = x - cx;
                float dy = y - cy;
                float dist = std::sqrt(dx * dx + dy * dy);
                img.at(x, y) = (std::sin(dist * 0.1f) + 1.0f) * 127.5f;
            }
        }
    } else if (pattern == "noise_gradient") {
        // 渐变+噪声
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> noise(-30.0f, 30.0f);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float base = (float)(x + y) / (width + height) * 255.0f;
                img.at(x, y) = std::max(0.0f, std::min(255.0f, base + noise(gen)));
            }
        }
    }
    
    return img;
}

std::vector<Image> generateTestImages(int count, int width, int height, const std::string& pattern) {
    std::vector<Image> images;
    images.reserve(count);
    
    std::vector<std::string> patterns = {"random", "gradient", "checkerboard", "circle", "noise_gradient"};
    
    for (int i = 0; i < count; ++i) {
        std::string p = (pattern == "mixed") ? patterns[i % patterns.size()] : pattern;
        images.push_back(generateTestImage(width, height, p));
    }
    
    return images;
}

bool saveImageAsText(const Image& img, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }
    
    file << img.width << " " << img.height << "\n";
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            file << (int)img.at(x, y);
            if (x < img.width - 1) file << " ";
        }
        file << "\n";
    }
    
    file.close();
    return true;
}

Image loadImageFromText(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return Image();
    }
    
    int width, height;
    file >> width >> height;
    
    Image img(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int val;
            file >> val;
            img.at(x, y) = (float)val;
        }
    }
    
    file.close();
    return img;
}

void normalizeImage(Image& img) {
    float min_val = *std::min_element(img.data.begin(), img.data.end());
    float max_val = *std::max_element(img.data.begin(), img.data.end());
    
    if (max_val - min_val < 1e-6f) return;
    
    for (float& pixel : img.data) {
        pixel = (pixel - min_val) / (max_val - min_val) * 255.0f;
    }
}

} // namespace ImageUtil

namespace KernelFactory {

Kernel createSobelX() {
    Kernel k(3, "Sobel_X");
    k.data = {-1, 0, 1,
              -2, 0, 2,
              -1, 0, 1};
    return k;
}

Kernel createSobelY() {
    Kernel k(3, "Sobel_Y");
    k.data = {-1, -2, -1,
               0,  0,  0,
               1,  2,  1};
    return k;
}

Kernel createGaussian3x3() {
    Kernel k(3, "Gaussian_3x3");
    k.data = {1, 2, 1,
              2, 4, 2,
              1, 2, 1};
    // 归一化
    for (float& val : k.data) val /= 16.0f;
    return k;
}

Kernel createGaussian5x5() {
    Kernel k(5, "Gaussian_5x5");
    k.data = {1,  4,  6,  4, 1,
              4, 16, 24, 16, 4,
              6, 24, 36, 24, 6,
              4, 16, 24, 16, 4,
              1,  4,  6,  4, 1};
    float sum = 256.0f;
    for (float& val : k.data) val /= sum;
    return k;
}

Kernel createGaussian7x7() {
    Kernel k(7, "Gaussian_7x7");
    // 7x7 高斯核 (sigma ≈ 1.0)
    k.data = {
        0, 0, 1,  2, 1, 0, 0,
        0, 3, 13, 22, 13, 3, 0,
        1, 13, 59, 97, 59, 13, 1,
        2, 22, 97, 159, 97, 22, 2,
        1, 13, 59, 97, 59, 13, 1,
        0, 3, 13, 22, 13, 3, 0,
        0, 0, 1,  2, 1, 0, 0
    };
    float sum = 0;
    for (float val : k.data) sum += val;
    for (float& val : k.data) val /= sum;
    return k;
}

Kernel createGaussian(int size, float sigma) {
    if (size % 2 == 0) size++;  // 确保奇数
    Kernel k(size, "Gaussian_" + std::to_string(size) + "x" + std::to_string(size));
    
    int half = size / 2;
    float sum = 0.0f;
    float sigma2 = 2.0f * sigma * sigma;
    
    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            float val = std::exp(-(x*x + y*y) / sigma2) / (M_PI * sigma2);
            k.at(x + half, y + half) = val;
            sum += val;
        }
    }
    
    // 归一化
    for (float& val : k.data) val /= sum;
    return k;
}

Kernel createLaplacian() {
    Kernel k(3, "Laplacian");
    k.data = {0,  1, 0,
              1, -4, 1,
              0,  1, 0};
    return k;
}

Kernel createLoG(int size, float sigma) {
    if (size % 2 == 0) size++;
    Kernel k(size, "LoG_" + std::to_string(size));
    
    int half = size / 2;
    float sigma2 = sigma * sigma;
    float sigma4 = sigma2 * sigma2;
    float sum = 0.0f;
    
    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            float r2 = x*x + y*y;
            float val = -(1.0f - r2 / (2.0f * sigma2)) * 
                        std::exp(-r2 / (2.0f * sigma2)) / (M_PI * sigma4);
            k.at(x + half, y + half) = val;
            sum += val;
        }
    }
    
    // 确保和为0
    float avg = sum / (size * size);
    for (float& val : k.data) val -= avg;
    
    return k;
}

Kernel createBoxBlur(int size) {
    Kernel k(size, "BoxBlur_" + std::to_string(size));
    float val = 1.0f / (size * size);
    std::fill(k.data.begin(), k.data.end(), val);
    return k;
}

Kernel createPrewittX() {
    Kernel k(3, "Prewitt_X");
    k.data = {-1, 0, 1,
              -1, 0, 1,
              -1, 0, 1};
    return k;
}

Kernel createPrewittY() {
    Kernel k(3, "Prewitt_Y");
    k.data = {-1, -1, -1,
               0,  0,  0,
               1,  1,  1};
    return k;
}

Kernel createScharrX() {
    Kernel k(3, "Scharr_X");
    k.data = {-3, 0, 3,
              -10, 0, 10,
              -3, 0, 3};
    return k;
}

Kernel createScharrY() {
    Kernel k(3, "Scharr_Y");
    k.data = {-3, -10, -3,
               0,   0,  0,
               3,  10,  3};
    return k;
}

Kernel createSharpen() {
    Kernel k(3, "Sharpen");
    k.data = { 0, -1,  0,
              -1,  5, -1,
               0, -1,  0};
    return k;
}

Kernel createUnsharpMask(int size, float amount) {
    Kernel gaussian = createGaussian(size, size / 6.0f);
    Kernel k(size, "UnsharpMask_" + std::to_string(size));
    
    int center = size / 2;
    for (int i = 0; i < size * size; ++i) {
        k.data[i] = -amount * gaussian.data[i];
    }
    k.at(center, center) += (1.0f + amount);
    
    return k;
}

Kernel createEmboss() {
    Kernel k(3, "Emboss");
    k.data = {-2, -1, 0,
              -1,  1, 1,
               0,  1, 2};
    return k;
}

Kernel createMotionBlur(int size, float angle) {
    Kernel k(size, "MotionBlur_" + std::to_string(size));
    std::fill(k.data.begin(), k.data.end(), 0.0f);
    
    int center = size / 2;
    float rad = angle * M_PI / 180.0f;
    float dx = std::cos(rad);
    float dy = std::sin(rad);
    
    int count = 0;
    for (int i = -center; i <= center; ++i) {
        int x = center + (int)std::round(i * dx);
        int y = center + (int)std::round(i * dy);
        if (x >= 0 && x < size && y >= 0 && y < size) {
            k.at(x, y) = 1.0f;
            count++;
        }
    }
    
    if (count > 0) {
        for (float& val : k.data) val /= count;
    }
    
    return k;
}

Kernel loadKernelFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开卷积核文件: " << filename << std::endl;
        return Kernel();
    }
    
    int size;
    file >> size;
    
    Kernel k(size);
    for (int i = 0; i < size * size; ++i) {
        file >> k.data[i];
    }
    
    file.close();
    return k;
}

std::vector<Kernel> getStandardKernelSet() {
    return {
        createSobelX(),
        createSobelY(),
        createGaussian3x3(),
        createGaussian5x5(),
        createLaplacian(),
        createPrewittX(),
        createPrewittY(),
        createScharrX(),
        createScharrY(),
        createSharpen(),
        createEmboss(),
        createBoxBlur(3),
        createBoxBlur(5)
    };
}

std::vector<Kernel> getLargeKernelSet() {
    std::vector<Kernel> kernels;
    
    // 小型核 (3x3, 5x5)
    kernels.push_back(createSobelX());
    kernels.push_back(createSobelY());
    kernels.push_back(createGaussian3x3());
    kernels.push_back(createGaussian5x5());
    kernels.push_back(createLaplacian());
    kernels.push_back(createPrewittX());
    kernels.push_back(createPrewittY());
    kernels.push_back(createScharrX());
    kernels.push_back(createScharrY());
    kernels.push_back(createSharpen());
    kernels.push_back(createEmboss());
    
    // 中型核 (7x7, 9x9, 11x11)
    kernels.push_back(createGaussian7x7());
    kernels.push_back(createGaussian(9, 1.5f));
    kernels.push_back(createGaussian(11, 2.0f));
    kernels.push_back(createBoxBlur(7));
    kernels.push_back(createBoxBlur(9));
    kernels.push_back(createBoxBlur(11));
    kernels.push_back(createLoG(7, 1.0f));
    kernels.push_back(createLoG(9, 1.4f));
    kernels.push_back(createMotionBlur(9, 0.0f));
    kernels.push_back(createMotionBlur(9, 45.0f));
    
    // 大型核 (15x15, 21x21, 31x31) - 用于增加计算量
    kernels.push_back(createGaussian(15, 3.0f));
    kernels.push_back(createGaussian(21, 4.0f));
    kernels.push_back(createGaussian(31, 6.0f));
    kernels.push_back(createBoxBlur(15));
    kernels.push_back(createBoxBlur(21));
    kernels.push_back(createBoxBlur(31));
    kernels.push_back(createUnsharpMask(15, 1.5f));
    kernels.push_back(createLoG(15, 2.0f));
    kernels.push_back(createMotionBlur(21, 0.0f));
    kernels.push_back(createMotionBlur(21, 90.0f));
    
    return kernels;
}

} // namespace KernelFactory
