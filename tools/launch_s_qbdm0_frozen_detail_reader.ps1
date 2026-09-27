$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$trainer = 'tools\train_s_qbdm0_frozen_detail_reader.py'
$outputDir = 'E:\two_paper\outputs\S_QBDM0_ALIGNED_FROZEN_A00_SEED0'
$reportDir = 'E:\two_paper\reports\25_query_guided_detail_memory\S_QBDM0'
$queueLog = Join-Path $reportDir 'queue.log'

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
if (Test-Path -LiteralPath (Join-Path $outputDir 'summary.json') -PathType Leaf) {
    throw "QBDM0已有完整summary，拒绝覆盖：$outputDir"
}
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
Add-Content -LiteralPath $queueLog `
    -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') START QBDM0 epochs=12 batch=16 seed=0" `
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
    throw "QBDM0失败，exit code=$($process.ExitCode)"
}
if (-not (Test-Path -LiteralPath (Join-Path $outputDir 'summary.json') -PathType Leaf)) {
    throw "QBDM0进程结束但缺少summary"
}
Add-Content -LiteralPath $queueLog `
    -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') COMPLETE summary=$(Join-Path $outputDir 'summary.json')" `
    -Encoding UTF8
