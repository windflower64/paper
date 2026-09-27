$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $workspace 'D-FINE\experiments\phase_stql_qcer\b4_sam_qcer.yml'
$init = Join-Path $workspace 'weights\stql_qcer_common_init_v1.pth'
$output = Join-Path $workspace 'outputs\STQL_QCER_PREFLIGHT_B4_B16A2\seed0'

foreach ($path in @($python, $config, $init)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}

$operatingSystem = Get-CimInstance Win32_OperatingSystem
$freePhysicalGiB = [math]::Round($operatingSystem.FreePhysicalMemory / 1MB, 2)
$freeVirtualGiB = [math]::Round($operatingSystem.FreeVirtualMemory / 1MB, 2)
$gpu = & nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

Write-Host "GPU: $gpu"
Write-Host "Free physical memory: $freePhysicalGiB GiB"
Write-Host "Free committed/virtual memory: $freeVirtualGiB GiB"

if ($freeVirtualGiB -lt 8) {
    throw "Free committed/virtual memory is still below 8 GiB; stop before CUDA initialization."
}

Set-Location -LiteralPath $workspace
& $python -u 'D-FINE/tools/preflight_train_stql_qcer.py' `
    --repo 'D-FINE' `
    --config $config `
    --init $init `
    --output-dir $output `
    --optimizer-steps 5 `
    --seed 0

if ($LASTEXITCODE -ne 0) {
    throw "STQL/QCER preflight failed with exit code $LASTEXITCODE"
}

$report = Join-Path $output 'short_train_report.json'
if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
    throw "Preflight exited without its acceptance report: $report"
}

Write-Host "PASS report: $report"
