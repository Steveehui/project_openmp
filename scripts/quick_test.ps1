# PowerShell 快速性能测试脚本
# 用途: 对比串行和 OpenMP 版本的性能

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  图像卷积 - 性能对比测试" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$BIN_DIR = Join-Path $PROJECT_ROOT "bin"

# 测试参数
$IMAGE_SIZES = @(512, 1024, 2048)
$THREAD_COUNTS = @(1, 2, 4, 8, 16)

Write-Host "`n测试配置:" -ForegroundColor Yellow
Write-Host "  图像尺寸: $($IMAGE_SIZES -join ', ')"
Write-Host "  线程数: $($THREAD_COUNTS -join ', ')"
Write-Host ""

# 辅助函数: 解析运行时间
function Get-RunTime {
    param($output)
    if ($output -match "总运行时间:\s+([\d.]+)\s+秒") {
        return [double]$matches[1]
    }
    return 0.0
}

# 测试1: 固定图像尺寸,变线程数
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "测试1: 强扩展性测试 (1024x1024)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

$TEST_SIZE = 1024

# 串行基准
Write-Host "`n串行版本..." -ForegroundColor Green
$serial_exe = Join-Path $BIN_DIR "conv_serial.exe"
$output = & $serial_exe $TEST_SIZE random 2>&1 | Out-String
$serial_time = Get-RunTime $output
Write-Host "  运行时间: $($serial_time)s" -ForegroundColor White

# OpenMP 变线程数
$omp_exe = Join-Path $BIN_DIR "conv_omp.exe"
Write-Host "`nOpenMP 版本:" -ForegroundColor Green
Write-Host "线程数`t时间(s)`t加速比`t效率(%)"
Write-Host "------`t-------`t------`t-------"

foreach ($threads in $THREAD_COUNTS) {
    $output = & $omp_exe $TEST_SIZE $threads random 2>&1 | Out-String
    $time = Get-RunTime $output
    $speedup = if ($time -gt 0) { [math]::Round($serial_time / $time, 2) } else { 0 }
    $efficiency = if ($threads -gt 0) { [math]::Round($speedup / $threads * 100, 1) } else { 0 }
    
    Write-Host "$threads`t$time`t${speedup}x`t$efficiency%"
}

# 测试2: 固定线程数,变图像尺寸
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "测试2: 弱扩展性测试 (16线程)" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

$FIXED_THREADS = 16

Write-Host "`n尺寸`t`t串行(s)`tOpenMP(s)`t加速比"
Write-Host "----`t`t-------`t--------`t------"

foreach ($size in $IMAGE_SIZES) {
    # 串行
    $output = & $serial_exe $size random 2>&1 | Out-String
    $serial_t = Get-RunTime $output
    
    # OpenMP
    $output = & $omp_exe $size $FIXED_THREADS random 2>&1 | Out-String
    $omp_t = Get-RunTime $output
    
    $speedup = if ($omp_t -gt 0) { [math]::Round($serial_t / $omp_t, 2) } else { 0 }
    
    Write-Host "${size}x${size}`t$serial_t`t$omp_t`t${speedup}x"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "测试完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
