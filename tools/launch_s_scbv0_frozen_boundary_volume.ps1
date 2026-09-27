$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$trainer = 'tools\train_s_scbv0_frozen_boundary_volume.py'
$outputDir = 'E:\two_paper\outputs\S_SCBV0_ALIGNED_FROZEN_A00_SEED0'
$reportDir = 'E:\two_paper\reports\26_semantic_boundary_volume\S_SCBV0'
$queueLog = Join-Path $reportDir 'queue.log'

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Conda解释器不存在：$pythonExe"
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $trainer) -PathType Leaf)) {
    throw "SCBV0训练器不存在：$trainer"
}
$gpuName = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($gpuName -join ''))) {
    throw '未检测到可用NVIDIA GPU，拒绝启动SCBV0'
}

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
if (Test-Path -LiteralPath (Join-Path $outputDir 'summary.json') -PathType Leaf) {
    throw "SCBV0已有完整summary，拒绝覆盖：$outputDir"
}
if (Test-Path -LiteralPath (Join-Path $outputDir 'last.pth') -PathType Leaf) {
    throw "SCBV0目录存在未完成训练权重，需先人工审计，拒绝自动覆盖：$outputDir"
}
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Add-Content -LiteralPath $queueLog `
    -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START SCBV0 epochs=12 batch=16 seed=0 gpu=$($gpuName -join ',')" `
    -Encoding UTF8

$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @(
        '-u', $trainer,
        '--epochs', '12',
        '--batch-size', '16',
        '--seed', '0',
        '--output-dir', $outputDir
    ) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $outputDir 'train_console.log') `
    -RedirectStandardError (Join-Path $outputDir 'train_stderr.log') `
    -Wait `
    -PassThru
if ($process.ExitCode -ne 0) {
    Add-Content -LiteralPath $queueLog `
        -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') FAILED exit_code=$($process.ExitCode)" `
        -Encoding UTF8
    throw "SCBV0失败，exit code=$($process.ExitCode)"
}
if (-not (Test-Path -LiteralPath (Join-Path $outputDir 'summary.json') -PathType Leaf)) {
    throw 'SCBV0进程结束但缺少summary'
}
Add-Content -LiteralPath $queueLog `
    -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') COMPLETE summary=$(Join-Path $outputDir 'summary.json')" `
    -Encoding UTF8

