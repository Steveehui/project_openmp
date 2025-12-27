# PowerShell 实验运行脚本
# 用途: 自动运行所有实验并记录结果

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  图像卷积项目 - 实验运行脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 项目目录
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$BIN_DIR = Join-Path $PROJECT_ROOT "bin"
$RESULTS_DIR = Join-Path $PROJECT_ROOT "results"

# 创建结果目录
if (-not (Test-Path $RESULTS_DIR)) {
    New-Item -ItemType Directory -Path $RESULTS_DIR | Out-Null
}

# CSV 输出文件
$CSV_FILE = Join-Path $RESULTS_DIR "data.csv"

# 初始化 CSV
"实验组,版本,图像尺寸,线程数,运行时间(秒),加速比" | Out-File -FilePath $CSV_FILE -Encoding UTF8

# 辅助函数: 解析运行时间
function Get-RunTime {
    param($output)
    if ($output -match "总运行时间:\s+([\d.]+)\s+秒") {
        return [double]$matches[1]
    }
    return 0.0
}

# ===========================
# 实验1: 变线程数(强扩展性)
# ===========================
Write-Host "`n实验1: 固定图像(1024x1024) + 变线程数" -ForegroundColor Yellow
$IMG_SIZE = 1024
$THREADS = @(1, 2, 4, 8, 16)

# 串行基准
Write-Host "  运行串行版本..." -ForegroundColor Cyan
$serial_exe = Join-Path $BIN_DIR "conv_serial.exe"
$output = & $serial_exe $IMG_SIZE random 2>&1 | Out-String
$serial_time = Get-RunTime $output
Write-Host "    时间: $serial_time 秒" -ForegroundColor Green
"实验1,Serial,$IMG_SIZE,1,$serial_time,1.00" | Out-File -FilePath $CSV_FILE -Append -Encoding UTF8

# OpenMP 变线程
$omp_exe = Join-Path $BIN_DIR "conv_omp.exe"
if (Test-Path $omp_exe) {
    foreach ($t in $THREADS) {
        Write-Host "  运行 OpenMP ($t 线程)..." -ForegroundColor Cyan
        $output = & $omp_exe $IMG_SIZE $t random 2>&1 | Out-String
        $time = Get-RunTime $output
        $speedup = if ($time -gt 0) { [math]::Round($serial_time / $time, 2) } else { 0 }
        Write-Host "    时间: $time 秒, 加速比: ${speedup}x" -ForegroundColor Green
        "实验1,OpenMP,$IMG_SIZE,$t,$time,$speedup" | Out-File -FilePath $CSV_FILE -Append -Encoding UTF8
    }
}

# ===========================
# 实验2: 变图像尺寸(弱扩展性)
# ===========================
Write-Host "`n实验2: 固定线程(8) + 变图像尺寸" -ForegroundColor Yellow
$SIZES = @(512, 1024, 2048)
$FIXED_THREADS = 8

foreach ($size in $SIZES) {
    Write-Host "  测试尺寸: ${size}x${size}" -ForegroundColor Cyan
    
    # 串行
    Write-Host "    [Serial]" -ForegroundColor Gray
    $output = & $serial_exe $size random 2>&1 | Out-String
    $time = Get-RunTime $output
    Write-Host "      时间: $time 秒" -ForegroundColor Green
    "实验2,Serial,$size,1,$time,1.00" | Out-File -FilePath $CSV_FILE -Append -Encoding UTF8
    
    # OpenMP
    if (Test-Path $omp_exe) {
        Write-Host "    [OpenMP]" -ForegroundColor Gray
        $output = & $omp_exe $size $FIXED_THREADS random 2>&1 | Out-String
        $omp_time = Get-RunTime $output
        $speedup = if ($omp_time -gt 0) { [math]::Round($time / $omp_time, 2) } else { 0 }
        Write-Host "      时间: $omp_time 秒, 加速比: ${speedup}x" -ForegroundColor Green
        "实验2,OpenMP,$size,$FIXED_THREADS,$omp_time,$speedup" | Out-File -FilePath $CSV_FILE -Append -Encoding UTF8
    }
}

# ===========================
# 实验3: 四种策略对比
# ===========================
Write-Host "`n实验3: 四种策略对比 (1024x1024)" -ForegroundColor Yellow
$TEST_SIZE = 1024

# Serial
Write-Host "  [1/4] Serial..." -ForegroundColor Cyan
$output = & $serial_exe $TEST_SIZE random 2>&1 | Out-String
$serial_time = Get-RunTime $output
Write-Host "    时间: $serial_time 秒" -ForegroundColor Green
"实验3,Serial,$TEST_SIZE,1,$serial_time,1.00" | Out-File -FilePath $CSV_FILE -Append -Encoding UTF8

# OpenMP
if (Test-Path $omp_exe) {
    Write-Host "  [2/4] OpenMP..." -ForegroundColor Cyan
    $output = & $omp_exe $TEST_SIZE 16 random --batch 2>&1 | Out-String
    $time = Get-RunTime $output
    $speedup = if ($time -gt 0) { [math]::Round($serial_time / $time, 2) } else { 0 }
    Write-Host "    时间: $time 秒, 加速比: ${speedup}x" -ForegroundColor Green
    "实验3,OpenMP,$TEST_SIZE,16,$time,$speedup" | Out-File -FilePath $CSV_FILE -Append -Encoding UTF8
}

# CUDA
$cuda_exe = Join-Path $BIN_DIR "conv_cuda.exe"
if (Test-Path $cuda_exe) {
    Write-Host "  [3/4] CUDA..." -ForegroundColor Cyan
    $output = & $cuda_exe $TEST_SIZE random 2>&1 | Out-String
    $time = Get-RunTime $output
    $speedup = if ($time -gt 0) { [math]::Round($serial_time / $time, 2) } else { 0 }
    Write-Host "    时间: $time 秒, 加速比: ${speedup}x" -ForegroundColor Green
    "实验3,CUDA,$TEST_SIZE,1,$time,$speedup" | Out-File -FilePath $CSV_FILE -Append -Encoding UTF8
}

# Hybrid
$hybrid_exe = Join-Path $BIN_DIR "conv_hybrid.exe"
if (Test-Path $hybrid_exe) {
    Write-Host "  [4/4] Hybrid..." -ForegroundColor Cyan
    $output = & $hybrid_exe $TEST_SIZE 1 4 random 2>&1 | Out-String
    $time = Get-RunTime $output
    $speedup = if ($time -gt 0) { [math]::Round($serial_time / $time, 2) } else { 0 }
    Write-Host "    时间: $time 秒, 加速比: ${speedup}x" -ForegroundColor Green
    "实验3,Hybrid,$TEST_SIZE,4,$time,$speedup" | Out-File -FilePath $CSV_FILE -Append -Encoding UTF8
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  所有实验完成!" -ForegroundColor Green
Write-Host "  结果已保存到: $CSV_FILE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n下一步: 运行 'python scripts\plot_results.py' 生成图表" -ForegroundColor Yellow
