param(
    [string]$ExePath = "..\bin\benchmark_full.exe",
    [string]$Args = "",
    [string]$Devices = "0,1"
)

$exeFull = Resolve-Path $ExePath
$devList = $Devices.Split(',')

Write-Host "Running on devices: $Devices" -ForegroundColor Cyan

foreach ($d in $devList) {
    $env:CUDA_VISIBLE_DEVICES = $d
    Write-Host "\n=== Device $d ===" -ForegroundColor Green
    & $exeFull $Args
}

# reset
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
