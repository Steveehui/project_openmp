# 快速开始指南

本指南帮助你在 5 分钟内运行起第一个程序。

## 环境要求

- Windows 10/11
- Visual Studio 2022 或 MinGW-w64
- CUDA Toolkit 11.0+ (可选,用于 GPU 加速)
- CMake 3.18+
- Python 3.7+ (可选,用于结果可视化)

## 快速编译与运行

### 方式1: 使用构建脚本(推荐)

```powershell
# 1. 编译所有版本
.\scripts\build.ps1

# 2. 测试串行版本
.\bin\conv_serial.exe 512

# 3. 测试 OpenMP 版本
.\bin\conv_omp.exe 512 8

# 4. (如果有 GPU) 测试 CUDA 版本
.\bin\conv_cuda.exe 512
```

### 方式2: 手动编译单个文件

如果只想快速测试串行版本:

```powershell
# 使用 MSVC (需要在 VS Developer Command Prompt 中运行)
cl /O2 /EHsc /std:c++14 /Iinclude src\image_util.cpp src\convolution.cpp src\main_serial.cpp /Fe:test_serial.exe

# 或使用 MinGW
g++ -O3 -std=c++14 -Iinclude src\image_util.cpp src\convolution.cpp src\main_serial.cpp -o test_serial.exe
```

## 常见问题

### Q1: 找不到 cmake 命令
**A:** 确保 CMake 已安装并添加到系统 PATH

### Q2: OpenMP 不工作
**A:** 
- MSVC: 检查是否添加了 `/openmp` 编译选项
- MinGW: 确保使用 `-fopenmp` 选项

### Q3: CUDA 版本编译失败
**A:** 
1. 检查 NVIDIA 驱动是否安装
2. 运行 `nvcc --version` 确认 CUDA 可用
3. 如不需要 GPU,可只编译 CPU 版本

## 验证安装

运行以下命令验证编译成功:

```powershell
# 检查可执行文件
ls .\bin\

# 应该看到:
# conv_serial.exe
# conv_omp.exe
# conv_cuda.exe (如果启用了 CUDA)
# conv_hybrid.exe (如果启用了 CUDA)
```

## 下一步

- 阅读 [README.md](README.md) 了解详细用法
- 运行 `.\scripts\run_experiments.ps1` 进行完整实验
- 查看 `src/` 目录学习代码实现

## 获取帮助

如遇到问题,请检查:
1. 编译器是否正确安装
2. CMake 版本是否满足要求
3. 是否在正确的目录下运行命令
