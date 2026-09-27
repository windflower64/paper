$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$trainer = 'tools\train_s_hrbr1_refinebox.py'
$queueDir = 'E:\two_paper\reports\24_high_resolution_box_refinement\S_HRBR1_MULTISEED'
$queueLog = Join-Path $queueDir 'queue.log'

New-Item -ItemType Directory -Force -Path $queueDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    Add-Content -LiteralPath $queueLog `
        -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" `
        -Encoding UTF8
}

function Invoke-Seed {
    param([int]$Seed)
    $outputDir = Join-Path $workspace "outputs\S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED$Seed"
    $summaryPath = Join-Path $outputDir 'summary.json'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        throw "Seed $Seed already has a completed summary: $summaryPath"
    }
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $stdoutPath = Join-Path $outputDir 'train_console.log'
    $stderrPath = Join-Path $outputDir 'train_stderr.log'
    Write-QueueLog "START seed=$Seed output=$outputDir"
    $process = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList @(
            '-u', $trainer,
            '--epochs', '12',
            '--batch-size', '16',
            '--seed', "$Seed",
            '--output-dir', $outputDir
        ) `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        Write-QueueLog "FAILED seed=$Seed exit_code=$($process.ExitCode) stderr=$stderrPath"
        throw "HRBR1 seed $Seed failed with exit code $($process.ExitCode)"
    }
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Seed $Seed ended without summary: $summaryPath"
    }
    Write-QueueLog "DONE seed=$Seed summary=$summaryPath"
}

foreach ($required in @(
    $pythonExe,
    (Join-Path $repoRoot $trainer),
    'E:\two_paper\outputs\S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED0\summary.json'
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}

Write-QueueLog 'QUEUE_STARTED protocol=HRBR1 epochs=12 batch=16 seeds=1,2'
Invoke-Seed -Seed 1
Invoke-Seed -Seed 2

$summaryOutput = Join-Path $queueDir 'multiseed_summary.json'
$summaryStdout = Join-Path $queueDir 'summarize_console.log'
$summaryStderr = Join-Path $queueDir 'summarize_stderr.log'
$summaryProcess = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList @(
        '-u', 'tools\summarize_s_hrbr1_multiseed.py',
        '--summaries',
        'E:\two_paper\outputs\S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED0\summary.json',
        'E:\two_paper\outputs\S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED1\summary.json',
        'E:\two_paper\outputs\S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED2\summary.json',
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
    throw "HRBR1 multiseed summary failed with exit code $($summaryProcess.ExitCode)"
}
Write-QueueLog "QUEUE_COMPLETE summary=$summaryOutput"

