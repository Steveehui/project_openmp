@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
cd /d %~dp0\..
cl /Fe:benchmark_full.exe /O2 /EHsc /std:c++14 /MD /utf-8 /openmp /DUSE_CUDA /Iinclude /I"%CUDA_PATH%\include" src\benchmark_full.cpp build_cuda\image_util.obj build_cuda\convolution.obj build_cuda\convolution_cuda.obj /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib
