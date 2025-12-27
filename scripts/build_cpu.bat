@echo off
REM CPU 版本编译脚本 (串行 + OpenMP)

echo ========================================
echo   编译 CPU 版本 (串行 + OpenMP)
echo ========================================

REM 设置 Visual Studio 环境
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM 创建构建目录
if not exist build_cpu mkdir build_cpu
cd build_cpu

echo.
echo [1/4] 编译图像工具类...
cl /c /O2 /EHsc /std:c++14 /MD /utf-8 /openmp /I..\include ..\src\image_util.cpp
if errorlevel 1 goto :error

echo.
echo [2/4] 编译卷积实现 (OpenMP 优化)...
cl /c /O2 /EHsc /std:c++14 /MD /utf-8 /openmp /I..\include ..\src\convolution.cpp
if errorlevel 1 goto :error

echo.
echo [3/4] 编译串行版本...
cl /Fe:conv_serial.exe /O2 /EHsc /std:c++14 /MD /utf-8 /openmp ^
   /I..\include ..\src\main_serial.cpp image_util.obj convolution.obj
if errorlevel 1 goto :error

echo.
echo [4/4] 编译 OpenMP 版本...
cl /Fe:conv_omp.exe /O2 /EHsc /std:c++14 /MD /utf-8 /openmp ^
   /I..\include ..\src\main_omp.cpp image_util.obj convolution.obj
if errorlevel 1 goto :error

REM 复制到 bin 目录
if not exist ..\bin mkdir ..\bin
copy conv_serial.exe ..\bin\
copy conv_omp.exe ..\bin\

cd ..

echo.
echo ========================================
echo   CPU 版本编译完成!
echo ========================================

dir bin\conv_serial.exe bin\conv_omp.exe
goto :end

:error
echo.
echo ========================================
echo   编译失败! 请检查错误信息。
echo ========================================
cd ..
exit /b 1

:end
