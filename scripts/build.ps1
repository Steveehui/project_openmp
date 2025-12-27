# PowerShell 构建脚本
# 用途: 自动编译所有版本的可执行文件

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  图像卷积项目 - 自动构建脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 检查 CMake 是否安装
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未检测到 CMake,请先安装!" -ForegroundColor Red
    exit 1
}

# 项目根目录
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
Write-Host "`n项目目录: $PROJECT_ROOT" -ForegroundColor Green

# 创建构建目录
$BUILD_DIR = Join-Path $PROJECT_ROOT "build"
if (Test-Path $BUILD_DIR) {
    Write-Host "清理旧的构建目录..." -ForegroundColor Yellow
    Remove-Item $BUILD_DIR -Recurse -Force
}
New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null

# 进入构建目录
Set-Location $BUILD_DIR

# 配置项目
Write-Host "`n[1/3] 配置 CMake 项目..." -ForegroundColor Cyan
cmake .. -DCMAKE_BUILD_TYPE=Release -A x64

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake 配置失败!" -ForegroundColor Red
    exit 1
}

# 编译项目
Write-Host "`n[2/3] 编译项目..." -ForegroundColor Cyan
cmake --build . --config Release -j 4

if ($LASTEXITCODE -ne 0) {
    Write-Host "编译失败!" -ForegroundColor Red
    exit 1
}

# 检查生成的可执行文件并复制到 bin 目录
Write-Host "`n[3/3] 检查编译结果..." -ForegroundColor Cyan
$BIN_DIR = Join-Path $PROJECT_ROOT "bin"
$BUILD_OUTPUT = Join-Path $BUILD_DIR "Release"

# 创建 bin 目录
if (-not (Test-Path $BIN_DIR)) {
    New-Item -ItemType Directory -Path $BIN_DIR | Out-Null
}

# 复制可执行文件
if (Test-Path $BUILD_OUTPUT) {
    Copy-Item (Join-Path $BUILD_OUTPUT "*.exe") $BIN_DIR -Force
}

$EXECUTABLES = @("conv_serial.exe", "conv_omp.exe", "conv_cuda.exe", "conv_hybrid.exe")

foreach ($exe in $EXECUTABLES) {
    $exe_path = Join-Path $BIN_DIR $exe
    if (Test-Path $exe_path) {
        Write-Host "  [OK] $exe" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $exe (可能未启用 CUDA)" -ForegroundColor Yellow
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  构建完成!" -ForegroundColor Green
Write-Host "  可执行文件位于: $BIN_DIR" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# 返回项目根目录
Set-Location $PROJECT_ROOT
