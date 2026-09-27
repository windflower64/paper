$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$trainer = 'tools\train_s_hrbr1_refinebox.py'
$summarizer = 'tools\summarize_s_hrbr5_d0_detail_causality.py'
$reportDir = 'E:\two_paper\reports\24_high_resolution_box_refinement\S_HRBR5_D0_DETAIL_CAUSALITY'
$queueLog = Join-Path $reportDir 'queue.log'

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    Add-Content -LiteralPath $queueLog `
        -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" `
        -Encoding UTF8
}

function Invoke-Arm {
    param(
        [string]$Mode,
        [string]$OutputName
    )
    $outputDir = Join-Path $workspace "outputs\$OutputName"
    $summaryPath = Join-Path $outputDir 'summary.json'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        throw "实验已存在完整 summary，拒绝覆盖：$summaryPath"
    }
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $stdoutPath = Join-Path $outputDir 'train_console.log'
    $stderrPath = Join-Path $outputDir 'train_stderr.log'
    Write-QueueLog "START mode=$Mode output=$outputDir"
    $process = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList @(
            '-u', $trainer,
            '--epochs', '12',
            '--batch-size', '16',
            '--seed', '0',
            '--feature-mode', $Mode,
            '--output-dir', $outputDir
        ) `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        Write-QueueLog "FAILED mode=$Mode exit_code=$($process.ExitCode) stderr=$stderrPath"
        throw "HRBR5-D0 $Mode 训练失败，exit code=$($process.ExitCode)"
    }
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "训练结束但缺少 summary：$summaryPath"
    }
    Write-QueueLog "DONE mode=$Mode summary=$summaryPath"
}

foreach ($required in @(
    $pythonExe,
    (Join-Path $repoRoot $trainer),
    (Join-Path $repoRoot $summarizer),
    (Join-Path $reportDir 'preflight.json')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "缺少必要文件：$required"
    }
}

Write-QueueLog 'QUEUE_STARTED protocol=HRBR5-D0 epochs=12 batch=16 seed=0 arms=full,lowpass,shifted_detail'
Invoke-Arm -Mode 'full' -OutputName 'S_HRBR5_D0_FULL_SEED0'
Invoke-Arm -Mode 'lowpass' -OutputName 'S_HRBR5_D0_LOWPASS_SEED0'
Invoke-Arm -Mode 'shifted_detail' -OutputName 'S_HRBR5_D0_SHIFTED_DETAIL_SEED0'

$summaryOutput = Join-Path $reportDir 'causality_summary.json'
$summaryStdout = Join-Path $reportDir 'summarize_console.log'
$summaryStderr = Join-Path $reportDir 'summarize_stderr.log'
$summaryProcess = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @(
        '-u', $summarizer,
        '--full', 'E:\two_paper\outputs\S_HRBR5_D0_FULL_SEED0\summary.json',
        '--lowpass', 'E:\two_paper\outputs\S_HRBR5_D0_LOWPASS_SEED0\summary.json',
        '--shifted-detail', 'E:\two_paper\outputs\S_HRBR5_D0_SHIFTED_DETAIL_SEED0\summary.json',
        '--output', $summaryOutput
    ) `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $summaryStdout `
    -RedirectStandardError $summaryStderr `
    -Wait `
    -PassThru
if ($summaryProcess.ExitCode -ne 0) {
    Write-QueueLog "FAILED summarize exit_code=$($summaryProcess.ExitCode) stderr=$summaryStderr"
    throw "HRBR5-D0 汇总失败，exit code=$($summaryProcess.ExitCode)"
}
Write-QueueLog "QUEUE_COMPLETE summary=$summaryOutput"
