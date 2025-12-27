# 图像卷积并行计算项目（扩展说明，中文）
1
## 1. 并行算法设计思路（更细节）
- **串行基线**：
  - 直接空间卷积，边界不填充；以此作为正确性与加速比基准。
  - 在 `convolution.cpp` 的 `convolve_serial`、`convolve_serial_blocked`、`convolve_serial_row_cache` 中提供若干缓存友好变体，为并行实现提供参考时间与验证输出。

- **OpenMP（CPU）**：
  - *OMP_Basic*：外层 `y/x` 双循环并行，调度 static，开销低；适合均匀工作量。
  - *OMP_Blocked*：先分块再并行（默认 64×64），减少 cache 失效与跨线程冲突；在大图/大核下显著稳定性能。
  - *OMP_SIMD*：在累加内层加 `#pragma omp simd`，使线程并行叠加向量化；对大核乘加密集场景收益明显。
  - *批/任务并行*（`convolve_batch_omp*`）：将 (image, kernel) 组成 2D 任务表，使用 dynamic/guided 调度平衡不同核尺寸的负载；可配合 chunk 大小调节调度开销。
  - *NUMA/亲和性*：双路场景建议 `OMP_PLACES=cores OMP_PROC_BIND=close`，减少跨 NUMA 访存；对超大线程数可视情调低线程以避免带宽饱和。

- **CUDA（GPU，核心在 src/convolution_cuda.cu）**：
  - *CUDA_Naive*：一线程一输出像素，直接访存，最简单正确性基线。
  - *CUDA_Shared*：输入块+核放共享内存；根据 `kernel_size` 选择 16×16 或 8×8 block，计算 SMEM = kernel + tile，并确保 <48 KB 以提升占用率；适合中大核。
  - *CUDA_Const*：核元素 ≤1024 时使用常量内存广播，减少核访存；超过上限自动回退 Shared。
  - *CUDA_Streams*：对批任务使用多流（如 4/8/12），分批 H2D/D2H + kernel 并行，提升吞吐；流数可按 PCIe/任务规模调节。
  - *CUDA_Policy*（启发式）：
    - 小核 ks≤7 → Const；
    - 大核 ks≥31 或大图≥4K → 预留 FFT/Winograd，当前回退 Shared；
    - 稀疏度>60% → 预留块稀疏，当前回退 Shared。
  - *CUDA_Best*：在 batch 前预上传核，选择共享/流的组合，减少重复拷贝与 kernel 配置；适合多图多核时的高吞吐。
  - *内存/SMEM/同步策略*：
    - Kernel 放常量或全局，输入输出放全局，必要时分批上传；
    - 依据 SMEM 占用自动降级 block 尺寸以提升 occupancy；
    - 测时/验证前 `cudaDeviceSynchronize`，流模式在逻辑同步点等待。

- **Hybrid（CPU+GPU）**：
  - 入口在 `Convolution::convolve_hybrid`：CPU 负责调度，GPU 执行批任务；当前对大图大核采用“全 GPU”策略，减少 PCIe 往返。
  - 预留自适应分工接口 `convolve_adaptive`：可按图像尺寸阈值（默认 512）选择 CPU/ GPU 路径，后续可按负载与 GPU 占用做动态决策。

- **多 GPU（脚本层面粗粒度）**：
  - `scripts/run_dual_gpu.sh` 将图片批次拆为两份，分别以 `CUDA_VISIBLE_DEVICES=0/1` 启动独立进程；full 基准合并 CSV，CUDA-only 基准仅日志。
  - 目前未在单进程内做设备间分工或通信，适合批量独立任务的吞吐场景；若需更细粒度，可扩展代码对 `cudaSetDevice` 与任务分配做内建调度。

## 2. 工作负载 / 基准
- *benchmark_full*：依次运行 Serial、OpenMP 各策略、CUDA 各策略、Hybrid，使用大核集合（3×3 至 31×31），输出 `benchmark_full_results.csv`。
- *benchmark_cuda*：CUDA 专项对比，包含 CUDA_Const 与 CUDA_Policy。
- 卷积核：`KernelFactory::getLargeKernelSet()` 提供标准+大尺寸核以同时压测计算与带宽。

## 3. 环境（本机实测）
- **硬件**
  - CPU：2× Intel Xeon Platinum 8481C，96 物理核 / 192 线程，总 L3 约 210 MiB，支持 AVX-512 / AMX。
  - GPU：2× NVIDIA GeForce RTX 4090 D，24,564 MiB，计算能力 8.9。
- **软件**
  - OS：Ubuntu 22.04.5 LTS (Jammy)。
  - CUDA Toolkit：11.5（nvcc 11.5.119）。
  - 编译器：GCC 11.4.0，C++14，OpenMP 通过 `-fopenmp` 启用。
  - 构建：nvcc/g++ 命令行；CMake 可生成 CPU + CUDA 目标（conv_serial/omp/cuda/hybrid）。

## 4. 构建示例
- **GPU 全基准（nvcc）**
  ```bash
  cd /root/project_openmp
  nvcc -O3 -std=c++14 -DUSE_CUDA -Xcompiler "-fopenmp" -Iinclude \
    src/benchmark_full.cpp src/image_util.cpp src/convolution.cpp src/convolution_cuda.cu \
    -o bin/benchmark_full_cuda
  ```
- **CUDA 专项基准**
  ```bash
  nvcc -O3 -std=c++14 -DUSE_CUDA -Xcompiler "-fopenmp" -Iinclude \
    src/benchmark_cuda.cpp src/image_util.cpp src/convolution.cpp src/convolution_cuda.cu \
    -o bin/benchmark_cuda
  ```
- **CPU-only 快速构建**（无 CUDA）
  ```bash
  g++ -O3 -std=c++14 -fopenmp -Iinclude \
    src/benchmark_full.cpp src/image_util.cpp src/convolution.cpp \
    -o bin/benchmark_full
  ```

## 5. 运行示例
- **单 GPU / 全流程**
  ```bash
  ./bin/benchmark_full_cuda 2048 1 16 1   # [图像尺寸] [图像数] [线程数] [迭代次数]
  ```
- **CUDA 专项（含 CUDA_Const / CUDA_Policy）**
  ```bash
  ./bin/benchmark_cuda 2048 1 16 1
  ```
- **双 GPU 粗粒度拆分（按图片分配）**
  - 全基准并合并 CSV：
    ```bash
    ./scripts/run_dual_gpu.sh 4096 4 16 1   # 默认使用 benchmark_full_cuda
    ```
  - CUDA 专项双卡（不产 CSV，仅日志）：
    ```bash
    EXEC=./bin/benchmark_cuda ./scripts/run_dual_gpu.sh 2048 4 16 1
    ```
  - 输出：
    - 日志：`/tmp/benchmark_gpu0_<ts>.log`，`/tmp/benchmark_gpu1_<ts>.log`
    - CSV（全基准模式）：`results/benchmark_gpu0_<ts>.csv`，`results/benchmark_gpu1_<ts>.csv`，合并 `results/benchmark_dual_<ts>.csv`

## 6. 性能调优提示
- OMP_Blocked 的块尺寸可按目标 CPU 缓存调整；L3 小的机器可减小 block。
- CUDA 小核（≤7）优先用 CUDA_Const；大核或大图依赖共享内存 tiling；批量足够大时多流能提升吞吐。
- 双卡脚本当前仅按图像数平分，如需更均衡可手动拆分卷积核集合或扩展脚本逻辑。

## 7. 代码位置速览
- 算法核心：`src/convolution.cpp`，`src/convolution_cuda.cu`，头文件 `include/`。
- 基准程序：`src/benchmark_full.cpp`，`src/benchmark_cuda.cpp`。
- 脚本：`scripts/run_dual_gpu.sh`（双卡）、`scripts/plot_results.py`（可视化）、Windows 构建脚本位于 `scripts/*.ps1`、`*.bat`。
