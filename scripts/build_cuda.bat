@echo off
REM CUDA 手动编译脚本 - 增强版

echo ========================================
echo   编译 CUDA 版本 (增强优化)
echo ========================================

REM 设置 Visual Studio 环境
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM 设置 CUDA 路径
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_PATH%\bin;%PATH%

REM 创建构建目录
if not exist build_cuda mkdir build_cuda
cd build_cuda

echo.
echo [1/5] 编译图像工具类...
cl /c /O2 /EHsc /std:c++14 /MD /utf-8 /openmp /I..\include ..\src\image_util.cpp
if errorlevel 1 goto :error

echo.
echo [2/5] 编译卷积实现 (OpenMP 优化)...
cl /c /O2 /EHsc /std:c++14 /MD /utf-8 /openmp /I..\include ..\src\convolution.cpp
if errorlevel 1 goto :error

echo.
echo [3/5] 编译 CUDA 卷积核 (sm_86 + 优化)...
nvcc -O3 -gencode arch=compute_86,code=sm_86 ^
     -Xcompiler "/MD /openmp /utf-8" ^
     --use_fast_math ^
     -I..\include ^
     -c ..\src\convolution_cuda.cu -o convolution_cuda.obj
if errorlevel 1 goto :error

echo.
echo [4/5] 链接 CUDA 可执行文件...
cl /Fe:conv_cuda.exe /O2 /EHsc /std:c++14 /MD /utf-8 /openmp /DUSE_CUDA ^
   /I..\include /I"%CUDA_PATH%\include" ^
   ..\src\main_cuda.cpp ^
   image_util.obj convolution.obj convolution_cuda.obj ^
   /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib
if errorlevel 1 goto :error

echo.
echo [5/5] 编译混合版本...
cl /c /O2 /EHsc /std:c++14 /MD /utf-8 /openmp /DUSE_CUDA ^
   /I..\include /I"%CUDA_PATH%\include" ^
   ..\src\main_hybrid.cpp
if errorlevel 1 goto :error

link /OUT:conv_hybrid.exe ^
     main_hybrid.obj image_util.obj convolution.obj convolution_cuda.obj ^
     /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib
if errorlevel 1 goto :error

REM 复制到 bin 目录
if not exist ..\bin mkdir ..\bin
copy conv_cuda.exe ..\bin\
copy conv_hybrid.exe ..\bin\

cd ..

echo.
echo ========================================
echo   CUDA 编译完成!
echo ========================================

dir bin\*.exe
goto :end

:error
echo.
echo ========================================
echo   编译失败! 请检查错误信息。
echo ========================================
cd ..
exit /b 1

:end
