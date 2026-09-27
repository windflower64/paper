$ErrorActionPreference = 'Stop'

$repo = 'E:\two_paper\D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_m\m_lite0_shared_k1_reader_b32_30e_testdev_local.yml'
$checkpoint = 'E:\two_paper\weights\m_sdtec1r2_visible_gq1_thermal_e56_init.pth'
$reportDir = 'E:\two_paper\reports\70_deployment\M_LITE0'
$preflightReport = Join-Path $reportDir 'm_lite0_preflight_seed0.json'
$outputDir = 'E:\two_paper\outputs\M_LITE0_SHARED_K1_READER_B32_30E_TESTDEV\seed0'

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
Write-Output "[$(Get-Date -Format o)] M-Lite0 queue started"
Write-Output "config_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $config).Hash)"
Write-Output "decoder_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $repo 'src\zoo\dfine\dfine_decoder.py')).Hash)"
Write-Output "checkpoint_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath $checkpoint).Hash)"

while ($true) {
    $activeTraining = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -match 'train\.py' -and
        $_.CommandLine -match 'two_paper\\D-FINE'
    }
    if (-not $activeTraining) {
        break
    }
    Write-Output "[$(Get-Date -Format o)] waiting for active training PID(s): $($activeTraining.ProcessId -join ',')"
    Start-Sleep -Seconds 30
}

Set-Location -LiteralPath $repo
Write-Output "[$(Get-Date -Format o)] starting CUDA batch-32 preflight"
& $python 'tools\preflight_m_sdtec1r2_training.py' `
    --repo $repo `
    --config $config `
    --checkpoint $checkpoint `
    --output $preflightReport `
    --seed 0
if ($LASTEXITCODE -ne 0) {
    throw "M-Lite0 preflight failed with exit code $LASTEXITCODE"
}

Write-Output "[$(Get-Date -Format o)] preflight passed; starting 30-epoch reader training"
& $python -u 'train.py' `
    -c $config `
    -t $checkpoint `
    --seed 0 `
    --use-amp `
    --output-dir $outputDir
if ($LASTEXITCODE -ne 0) {
    throw "M-Lite0 training failed with exit code $LASTEXITCODE"
}
Write-Output "[$(Get-Date -Format o)] M-Lite0 training completed"
