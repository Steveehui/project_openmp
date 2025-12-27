# Image Convolution Parallel Computing (Extended Notes)
1
## 1. Parallel Algorithm Design
- **Serial baseline**: straightforward spatial convolution (no padding) as correctness and speedup reference.
- **OpenMP CPU**
  - *OMP_Basic*: pixel-wise outer loops parallelized with `omp parallel for`, fixed thread count.
  - *OMP_Blocked*: cache-aware tiling (typ. 64×64 tiles) to raise locality under parallelism.
  - *OMP_SIMD*: adds pragma SIMD on inner accumulation to leverage vector units (AVX2/AVX-512 on this host).
  - *Batch/Tasking* (used in other code paths): schedule image×kernel tasks for better load balance on mixed sizes.
- **CUDA GPU** (in `convolution_cuda.cu`)
  - *CUDA_Naive*: direct global-memory kernel; correctness baseline for GPU.
  - *CUDA_Shared*: tiles input + kernel into shared memory; auto-switches 16×16 vs 8×8 blocks when SMEM is tight.
  - *CUDA_Const*: for small kernels (≤32×32), places kernel in constant memory; falls back to shared for larger ones.
  - *CUDA_Streams*: pipelined batch execution over multiple streams to overlap transfers/compute.
  - *CUDA_Policy*: heuristic dispatcher — small kernels→const, large/sparse/FFT-like→shared fallback (placeholders for future specialized paths).
  - *CUDA_Best*: convenience path picking efficient shared/stream strategy with pre-uploaded kernels.
- **Hybrid CPU+GPU**
  - CPU schedules, GPU executes heavy tasks; currently GPU does all batch work for these sizes, but hook exists for adaptive split.
- **Multi-GPU (external script)**
  - `scripts/run_dual_gpu.sh` splits the image batch across two devices via `CUDA_VISIBLE_DEVICES` and launches two processes, then merges CSV (full benchmark) or just logs (CUDA-only benchmark). This is a coarse-grain splitter; there is no in-process multi-GPU kernel.

## 2. Workloads / Benchmarks
- *benchmark_full*: runs Serial, OpenMP variants, CUDA variants, Hybrid on a large kernel set (3×3–31×31) and saves `benchmark_full_results.csv`.
- *benchmark_cuda*: CUDA-focused comparison including CUDA_Const and CUDA_Policy.
- Kernels: standard + large set from `KernelFactory::getLargeKernelSet()` to stress memory and compute.
- Typical heavy run: 4096–8192 images, multiple kernels, configurable iterations.

## 3. Environment (this run)
- **Hardware**
  - CPU: 2× Intel Xeon Platinum 8481C, 96 cores / 192 threads total, AVX-512, AMX; L3 210 MiB.
  - GPU: 2× NVIDIA GeForce RTX 4090 D, 24,564 MiB each, compute capability 8.9.
- **Software**
  - OS: Ubuntu 22.04.5 LTS (Jammy).
  - CUDA Toolkit: 11.5 (nvcc 11.5.119).
  - Compiler: GCC 11.4.0, C++14; OpenMP enabled via `-fopenmp`.
  - Build system: ad-hoc nvcc/g++ commands; CMake file provided for CPU+CUDA targets (conv_serial/omp/cuda/hybrid).

## 4. Build Instructions
- **Full GPU-enabled benchmark (nvcc)**
  ```bash
  cd /root/project_openmp
  nvcc -O3 -std=c++14 -DUSE_CUDA -Xcompiler "-fopenmp" -Iinclude \
    src/benchmark_full.cpp src/image_util.cpp src/convolution.cpp src/convolution_cuda.cu \
    -o bin/benchmark_full_cuda
  ```
- **CUDA-focused benchmark**
  ```bash
  nvcc -O3 -std=c++14 -DUSE_CUDA -Xcompiler "-fopenmp" -Iinclude \
    src/benchmark_cuda.cpp src/image_util.cpp src/convolution.cpp src/convolution_cuda.cu \
    -o bin/benchmark_cuda
  ```
- **CPU-only quick build** (no CUDA)
  ```bash
  g++ -O3 -std=c++14 -fopenmp -Iinclude \
    src/benchmark_full.cpp src/image_util.cpp src/convolution.cpp \
    -o bin/benchmark_full
  ```

## 5. Run Instructions
- **Single GPU / full suite**
  ```bash
  ./bin/benchmark_full_cuda 2048 1 16 1   # [image_size] [num_images] [threads] [iterations]
  ```
- **CUDA-only comparison (includes CUDA_Const / CUDA_Policy)**
  ```bash
  ./bin/benchmark_cuda 2048 1 16 1
  ```
- **Dual-GPU coarse splitter (image-level split)**
  - Full benchmark with CSV merge:
    ```bash
    ./scripts/run_dual_gpu.sh 4096 4 16 1   # uses benchmark_full_cuda by default
    ```
  - CUDA-only benchmark on two GPUs (no CSV, logs only):
    ```bash
    EXEC=./bin/benchmark_cuda ./scripts/run_dual_gpu.sh 2048 4 16 1
    ```
  - Outputs:
    - Logs: `/tmp/benchmark_gpu0_<ts>.log`, `/tmp/benchmark_gpu1_<ts>.log`
    - CSV (full benchmark mode): `results/benchmark_gpu0_<ts>.csv`, `results/benchmark_gpu1_<ts>.csv`, merged `results/benchmark_dual_<ts>.csv`

## 6. Notes on Performance Tuning
- Use larger tiles (OMP_Blocked) when L3 is ample; adjust block size in `convolution.cpp` if targeting smaller caches.
- For CUDA, small kernels (≤7) benefit most from CUDA_Const; large kernels lean on shared-memory tiling; streams help when batches are sizable.
- Dual-GPU splitter currently divides by image count; for skewed workloads consider manual kernel-set partitioning or extending script logic.

## 7. Repository Pointers
- Core algorithms: `src/convolution.cpp`, `src/convolution_cuda.cu`, headers in `include/`.
- Benchmarks: `src/benchmark_full.cpp`, `src/benchmark_cuda.cpp`.
- Scripts: `scripts/run_dual_gpu.sh` (dual GPU), `scripts/plot_results.py` (visualization), PowerShell scripts for Windows builds.
