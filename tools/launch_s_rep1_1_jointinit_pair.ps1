$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$controlConfig = 'experiments\phase_s\s_rep1_1_jointinit_control_b32_60e_local.yml'
$progressiveConfig = 'experiments\phase_s\s_rep1_1_jointinit_progressive_w025_b32_60e_local.yml'
$controlRun = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_CONTROL_B32_60E\seed0'
$progressiveRun = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_PROGRESSIVE_W025_B32_60E\seed0'
$reportDir = 'E:\two_paper\reports\21_public_reproduction\S_REP1_1_JOINTINIT'
$queueDir = 'E:\two_paper\runs\21_public_reproduction\_queue'
$queueLog = Join-Path $queueDir 's_rep1_1_jointinit_pair_queue.log'

New-Item -ItemType Directory -Force -Path $queueDir, $reportDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    Add-Content -LiteralPath $queueLog `
        -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" `
        -Encoding UTF8
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments, [string]$StdoutPath)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StdoutPath) | Out-Null
    $stderrPath = "$StdoutPath.stderr.log"
    Write-QueueLog "START $Name"
    $process = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $stderrPath `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        Write-QueueLog "FAILED $Name exit_code=$($process.ExitCode) stderr=$stderrPath"
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    Write-QueueLog "DONE $Name"
}

foreach ($path in @($pythonExe)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}
foreach ($runDir in @($controlRun, $progressiveRun)) {
    if (Test-Path -LiteralPath (Join-Path $runDir 'log.txt') -PathType Leaf) {
        throw "JointInit output already contains a log: $runDir"
    }
}

Write-QueueLog 'QUEUE_STARTED method=REP1.1-JointInit paired=true epochs=60 batch=32 seed=0'

Invoke-PythonStep -Name 'full_batch32_amp_preflight' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_batch32_preflight.log') `
    -Arguments @(
        '-u', 'tools\preflight_s_rep1_jointinit_batch32.py',
        '--config', $progressiveConfig,
        '--expected-batch', '32', '--epoch', '0',
        '--output', (Join-Path $reportDir 'preflight_full_batch32.json')
    )

Invoke-PythonStep -Name 'train_control_from_common_initialization' `
    -StdoutPath (Join-Path $controlRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $controlConfig,
        '--seed', '0', '--use-amp'
    )

Invoke-PythonStep -Name 'train_progressive_spar_from_common_initialization' `
    -StdoutPath (Join-Path $progressiveRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $progressiveConfig,
        '--seed', '0', '--use-amp'
    )

$bestCheckpoint = Join-Path $progressiveRun 'best_stg1.pth'
$lastCheckpoint = Join-Path $progressiveRun 'last.pth'
foreach ($path in @($bestCheckpoint, $lastCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Training checkpoint is missing: $path"
    }
}

Invoke-PythonStep -Name 'progressive_best_mask_alignment' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_best_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $progressiveConfig,
        '--checkpoint', $bestCheckpoint, '--batches', '20',
        '--output', (Join-Path $reportDir 'mask_alignment_best.json')
    )

Invoke-PythonStep -Name 'progressive_last_mask_alignment' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_last_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $progressiveConfig,
        '--checkpoint', $lastCheckpoint, '--batches', '20',
        '--output', (Join-Path $reportDir 'mask_alignment_last.json')
    )

Invoke-PythonStep -Name 'summarize_jointinit_pair' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_jointinit.py',
        '--control-log', (Join-Path $controlRun 'log.txt'),
        '--progressive-log', (Join-Path $progressiveRun 'log.txt'),
        '--best-alignment', (Join-Path $reportDir 'mask_alignment_best.json'),
        '--last-alignment', (Join-Path $reportDir 'mask_alignment_last.json'),
        '--output', (Join-Path $reportDir 'final_summary.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'
