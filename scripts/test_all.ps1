# 完整性能对比测试脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  图像卷积 - 完整性能对比测试" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$imageSize = 2048
if ($args.Length -gt 0) {
    $imageSize = $args[0]
}

Write-Host "测试图像尺寸: ${imageSize}x${imageSize}" -ForegroundColor Yellow
Write-Host ""

# 测试串行版本
Write-Host ">>> [1/4] 串行版本" -ForegroundColor Green
.\bin\conv_serial.exe $imageSize random
Write-Host ""

# 测试 OpenMP 版本
Write-Host ">>> [2/4] OpenMP 并行版本" -ForegroundColor Green
.\bin\conv_omp.exe $imageSize random
Write-Host ""

# 测试 CUDA 版本
Write-Host ">>> [3/4] CUDA GPU 版本" -ForegroundColor Green
.\bin\conv_cuda.exe $imageSize random
Write-Host ""

# 测试混合版本
Write-Host ">>> [4/4] 混合版本 (OpenMP + CUDA)" -ForegroundColor Green
.\bin\conv_hybrid.exe $imageSize 4 16 random
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  测试完成!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "提示:" -ForegroundColor Yellow
Write-Host "  - 查看详细性能分析: cat PERFORMANCE_RESULTS.md" -ForegroundColor White
Write-Host "  - 测试其他尺寸: .\scripts\test_all.ps1 1024" -ForegroundColor White
Write-Host "  - 单独运行: .\bin\conv_<版本>.exe <尺寸> <图案>" -ForegroundColor White
