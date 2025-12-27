#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H

#include <vector>
#include <string>
#include <cstdint>

// Simple image structure (grayscale)
struct Image {
    int width;
    int height;
    std::vector<float> data;  // Grayscale values [0, 255]

    Image() : width(0), height(0) {}
    Image(int w, int h) : width(w), height(h), data(w * h, 0.0f) {}

    float& at(int x, int y) { return data[y * width + x]; }
    const float& at(int x, int y) const { return data[y * width + x]; }
    
    // Raw data pointer (for optimization)
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
    
    // Row pointer (for optimization)
    float* row_ptr(int y) { return data.data() + y * width; }
    const float* row_ptr(int y) const { return data.data() + y * width; }
};

// Convolution kernel structure
struct Kernel {
    int size;  // Kernel size (size x size)
    std::vector<float> data;
    std::string name;

    Kernel() : size(0) {}
    Kernel(int s, const std::string& n = "") : size(s), data(s * s, 0.0f), name(n) {}

    float& at(int x, int y) { return data[y * size + x]; }
    const float& at(int x, int y) const { return data[y * size + x]; }
    
    // Raw data pointer
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

// Image I/O utilities
namespace ImageUtil {
    // Generate test image (random noise or gradient pattern)
    Image generateTestImage(int width, int height, const std::string& pattern = "random");

    // Save image as text file
    bool saveImageAsText(const Image& img, const std::string& filename);

    // Load image from text file
    Image loadImageFromText(const std::string& filename);

    // Normalize image data to [0, 255]
    void normalizeImage(Image& img);
    
    // Generate multiple test images
    std::vector<Image> generateTestImages(int count, int width, int height, const std::string& pattern = "random");
}

// Predefined kernels
namespace KernelFactory {
    // Sobel X edge detection
    Kernel createSobelX();
    
    // Sobel Y edge detection
    Kernel createSobelY();
    
    // Gaussian blur 3x3
    Kernel createGaussian3x3();
    
    // Gaussian blur 5x5
    Kernel createGaussian5x5();
    
    // Gaussian blur 7x7
    Kernel createGaussian7x7();
    
    // Gaussian blur NxN (variable size)
    Kernel createGaussian(int size, float sigma = 1.0f);
    
    // Laplacian sharpening
    Kernel createLaplacian();
    
    // Laplacian of Gaussian (LoG)
    Kernel createLoG(int size, float sigma = 1.4f);
    
    // Box blur
    Kernel createBoxBlur(int size);
    
    // Prewitt edge detection
    Kernel createPrewittX();
    Kernel createPrewittY();
    
    // Scharr edge detection
    Kernel createScharrX();
    Kernel createScharrY();
    
    // Sharpen kernel
    Kernel createSharpen();
    Kernel createUnsharpMask(int size, float amount = 1.0f);
    
    // Emboss effect
    Kernel createEmboss();
    
    // Motion blur
    Kernel createMotionBlur(int size, float angle = 0.0f);
    
    // Load kernel from file
    Kernel loadKernelFromFile(const std::string& filename);
    
    // Get standard test kernel set
    std::vector<Kernel> getStandardKernelSet();
    
    // Get large kernel set (for performance testing)
    std::vector<Kernel> getLargeKernelSet();
}

#endif // IMAGE_UTIL_H
