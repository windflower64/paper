$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$checkpoint = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\best_stg1.pth'
$runRoot = 'E:\two_paper\runs\21_public_reproduction'
$reportRoot = 'E:\two_paper\reports\21_public_reproduction\S_REP1_1_GENTLE'
$queueRoot = Join-Path $runRoot '_queue'
$queueLog = Join-Path $queueRoot 's_rep1_1_gentle_queue.log'

$controlConfig = 'experiments\phase_s\s_rep1_1_a00ft6_lr02_control_local.yml'
$sparConfig = 'experiments\phase_s\s_rep1_1_spar_w025_ft6_lr02_local.yml'
$controlDir = Join-Path $runRoot 'S_REP1_1_A00FT6_LR02_CONTROL\seed0'
$sparDir = Join-Path $runRoot 'S_REP1_1_SPAR_W025_FT6_LR02\seed0'

New-Item -ItemType Directory -Force -Path $queueRoot, $reportRoot | Out-Null

function Write-QueueLog {
    param([string]$Message)
    Add-Content -LiteralPath $queueLog `
        -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" `
        -Encoding UTF8
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$StdoutPath
    )
    $directory = Split-Path -Parent $StdoutPath
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
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
        Write-QueueLog "FAILED $Name exit_code=$($process.ExitCode) stdout=$StdoutPath stderr=$stderrPath"
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    Write-QueueLog "DONE $Name stdout=$StdoutPath stderr=$stderrPath"
}

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Python executable is missing: $pythonExe"
}
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "A00 checkpoint is missing: $checkpoint"
}

Write-QueueLog 'QUEUE_STARTED'

Invoke-PythonStep -Name 'preflight_batch16' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_preflight_batch16.log') `
    -Arguments @(
        '-u', 'tools\preflight_s_rep1_spar.py',
        '--config', $sparConfig,
        '--baseline-config', 'experiments\phase_s\visible_60e_base_local.yml',
        '--gradient-batch', '16',
        '--output', (Join-Path $reportRoot 'preflight_batch16.json')
    )

Invoke-PythonStep -Name 'low_lr_control' `
    -StdoutPath (Join-Path $controlDir 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $controlConfig,
        '-t', $checkpoint, '--seed', '0', '--use-amp'
    )

Invoke-PythonStep -Name 'gentle_spar' `
    -StdoutPath (Join-Path $sparDir 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $sparConfig,
        '-t', $checkpoint, '--seed', '0', '--use-amp'
    )

$bestCheckpoint = Join-Path $sparDir 'best_stg1.pth'
$lastCheckpoint = Join-Path $sparDir 'last.pth'
foreach ($path in @($bestCheckpoint, $lastCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing SPAR checkpoint: $path"
    }
}

Invoke-PythonStep -Name 'best_mask_alignment' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_best_mask_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $sparConfig, '--checkpoint', $bestCheckpoint,
        '--batches', '20',
        '--output', (Join-Path $reportRoot 'mask_alignment_best.json')
    )

Invoke-PythonStep -Name 'last_mask_alignment' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_last_mask_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $sparConfig, '--checkpoint', $lastCheckpoint,
        '--batches', '20',
        '--output', (Join-Path $reportRoot 'mask_alignment_last.json')
    )

Invoke-PythonStep -Name 'summarize_pair' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_summarize_pair.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_pair.py',
        '--control-log', (Join-Path $controlDir 'log.txt'),
        '--spar-log', (Join-Path $sparDir 'log.txt'),
        '--output', (Join-Path $reportRoot 'paired_result.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'
