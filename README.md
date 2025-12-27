# Image Convolution Parallel Computing Project

## 项目简介

基于混合并行(OpenMP + CUDA)的图像卷积加速项目,面向多尺度特征提取的并行优化研究。

**项目状态**: ✅ **全部完成** - 所有版本编译成功并通过性能测试  
**作者**: 你的姓名  
**学号**: XXXXXXXX  
**课程**: 高性能计算与并行编程  
**日期**: 2025年12月8日

### 🎯 性能亮点

- **OpenMP**: 3.6-3.7x 加速 (16线程)
- **CUDA**: 3.2-3.9x 加速 (RTX 3070)  
- **混合版**: 95.56 任务/秒吞吐量
- **四个版本**: 串行、OpenMP、CUDA、混合全部可用

### 📚 快速导航

- **5分钟上手**: [QUICKSTART.md](QUICKSTART.md)
- **性能测试报告**: [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md)
- **完整项目总结**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **编译问题解决**: [BUILD_FIX.md](BUILD_FIX.md)

---

## 目录结构

```
project_openmp/
├── include/              # 头文件
│   ├── timer.h          # 高精度计时器
│   ├── image_util.h     # 图像工具类
│   └── convolution.h    # 卷积算法接口
├── src/                 # 源代码
│   ├── image_util.cpp   # 图像工具实现
│   ├── convolution.cpp  # 串行/OpenMP 实现
│   ├── convolution_cuda.cu  # CUDA 实现
│   ├── main_serial.cpp  # 串行版本主程序
│   ├── main_omp.cpp     # OpenMP 版本主程序
│   ├── main_cuda.cpp    # CUDA 版本主程序
│   └── main_hybrid.cpp  # 混合版本主程序
├── scripts/             # 脚本文件
│   ├── build.ps1        # 构建脚本
│   ├── run_experiments.ps1  # 实验运行脚本
│   └── plot_results.py  # 结果可视化
├── results/             # 实验结果
│   └── data.csv         # 实验数据
├── CMakeLists.txt       # CMake 构建配置
└── README.md            # 本文件
```

---

## 编译说明

### 方式1: 使用 CMake (推荐)

```powershell
# 创建构建目录
mkdir build
cd build

# 配置项目(自动检测 CUDA)
cmake ..

# 编译(4 个并行任务)
cmake --build . --config Release -j 4
```

### 方式2: 使用 Visual Studio

1. 打开 Visual Studio 2022
2. 选择 "打开本地文件夹" → 选择 `project_openmp` 目录
3. VS 会自动识别 `CMakeLists.txt` 并配置项目
4. 按 `F7` 或点击 "生成" → "生成解决方案"

### 方式3: 手动编译

#### 串行版本
```powershell
g++ -O3 -std=c++14 -Iinclude src/image_util.cpp src/convolution.cpp src/main_serial.cpp -o bin/conv_serial.exe
```

#### OpenMP 版本
```powershell
g++ -O3 -std=c++14 -fopenmp -Iinclude src/image_util.cpp src/convolution.cpp src/main_omp.cpp -o bin/conv_omp.exe
```

#### CUDA 版本 (需要 CUDA Toolkit)
```powershell
nvcc -O3 -std=c++14 -Iinclude src/image_util.cpp src/convolution.cpp src/convolution_cuda.cu src/main_cuda.cpp -o bin/conv_cuda.exe
```

---

## 运行示例

### 串行版本
```powershell
.\bin\conv_serial.exe 1024 random
# 参数: [图像尺寸] [图案类型: random/gradient/checkerboard]
```

### OpenMP 版本
```powershell
.\bin\conv_omp.exe 1024 8 random --batch
# 参数: [图像尺寸] [线程数] [图案类型] [--batch: 批处理模式]
```

### CUDA 版本
```powershell
.\bin\conv_cuda.exe 2048 random --verify
# 参数: [图像尺寸] [图案类型] [--verify: 验证正确性]
```

### 混合版本
```powershell
.\bin\conv_hybrid.exe 1024 4 4 random --compare
# 参数: [图像尺寸] [图像数量] [CPU线程数] [图案类型] [--compare: 对比OMP]
```

---

## 实验流程

### 步骤1: 编译所有版本
```powershell
cd scripts
.\build.ps1
```

### 步骤2: 运行实验
```powershell
.\run_experiments.ps1
```

这将自动运行以下实验:
- 实验1: 固定图像(1024×1024) + 变线程数(1, 2, 4, 8, 16)
- 实验2: 固定线程(16) + 变图像尺寸(512, 1024, 2048)
- 实验3: 固定数据 + 四种策略对比(Serial, OpenMP, CUDA, Hybrid)

### 步骤3: 生成报告图表
```powershell
python scripts/plot_results.py
```

---

## 卷积核说明

项目内置了以下经典卷积核:

| 卷积核名称 | 尺寸 | 功能 |
|-----------|------|------|
| Sobel X   | 3×3  | 水平边缘检测 |
| Sobel Y   | 3×3  | 垂直边缘检测 |
| Gaussian  | 3×3  | 高斯模糊 |
| Laplacian | 3×3  | 图像锐化 |

---

## 依赖环境

| 软件/库 | 版本要求 | 用途 |
|---------|---------|------|
| CMake   | ≥ 3.18  | 构建工具 |
| GCC/MSVC | 支持 C++14 | C++ 编译器 |
| CUDA Toolkit | ≥ 11.0 | GPU 加速 |
| Python  | ≥ 3.7 | 结果可视化 |
| matplotlib | - | 绘图库 |

### Windows 环境配置

1. **安装 Visual Studio 2022** (包含 MSVC 编译器)
2. **安装 CUDA Toolkit 12.6** 
   - 下载: https://developer.nvidia.com/cuda-downloads
3. **安装 CMake**
   - 下载: https://cmake.org/download/
4. **安装 Python 依赖**
   ```powershell
   pip install matplotlib pandas numpy
   ```

---

## 预期性能

| 版本 | 1024×1024 图像 | 相对加速比 |
|------|---------------|-----------|
| Serial | ~8.5 秒 | 1.0x |
| OpenMP (16线程) | ~0.8 秒 | 10.6x |
| CUDA | ~0.15 秒 | 56.7x |
| Hybrid | ~0.12 秒 | 70.8x |

*测试环境: i7-13700H + RTX 4070*

---

## 故障排查

### 问题1: CUDA 未检测到
```
解决方案: 
1. 检查 NVIDIA 驱动是否安装
2. 运行 `nvcc --version` 确认 CUDA 安装
3. 设置环境变量 CUDA_PATH
```

### 问题2: OpenMP 不工作
```
解决方案:
1. MSVC: 确保添加了 /openmp 编译选项
2. GCC: 添加 -fopenmp 链接选项
3. 运行时设置 OMP_NUM_THREADS 环境变量
```

### 问题3: 编译错误 "找不到 omp.h"
```
解决方案:
MinGW 用户需要安装完整版 MinGW-w64
或使用 Visual Studio 自带的 MSVC 编译器
```

---

## 参考资料

1. **OpenMP 官方文档**: https://www.openmp.org/specifications/
2. **CUDA C++ 编程指南**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
3. **CMake 教程**: https://cmake.org/cmake/help/latest/guide/tutorial/
4. **图像卷积原理**: [数字图像处理教材第3章]

---

## 许可证

本项目仅用于课程学习,禁止商业使用。

---

## 联系方式

如有问题,请联系:
- 邮箱: your_email@example.com
- GitHub: https://github.com/yourusername/project_openmp

**祝实验顺利! 🚀**
