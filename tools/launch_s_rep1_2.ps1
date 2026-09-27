$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$a00Checkpoint = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\best_stg1.pth'
$config = 'experiments\phase_s\s_rep1_2_spar_w025_maskres_ft6_lr02_local.yml'
$runDir = 'E:\two_paper\runs\21_public_reproduction\S_REP1_2_SPAR_W025_MASKRES_FT6_LR02\seed0'
$controlLog = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_A00FT6_LR02_CONTROL\seed0\log.txt'
$rep11Log = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_RESTART30_SPAR_W025_LR02\seed0\log.txt'
$reportDir = 'E:\two_paper\reports\21_public_reproduction\S_REP1_2'
$queueDir = 'E:\two_paper\runs\21_public_reproduction\_queue'
$queueLog = Join-Path $queueDir 's_rep1_2_queue.log'

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

foreach ($path in @($pythonExe, $a00Checkpoint, $controlLog, $rep11Log)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}
if (Test-Path -LiteralPath (Join-Path $runDir 'log.txt') -PathType Leaf) {
    throw "REP1.2 output already contains a training log: $runDir"
}

Write-QueueLog 'QUEUE_STARTED method=REP1.2 change=mask_resolution_only epochs=6 batch=16'

Invoke-PythonStep -Name 'preflight_batch16' `
    -StdoutPath (Join-Path $queueDir 's_rep1_2_preflight.log') `
    -Arguments @(
        '-u', 'tools\preflight_s_rep1_spar.py', '--config', $config,
        '--baseline-config', 'experiments\phase_s\visible_60e_base_local.yml',
        '--gradient-batch', '16',
        '--output', (Join-Path $reportDir 'preflight_batch16.json')
    )

Invoke-PythonStep -Name 'train_rep1_2_from_a00' `
    -StdoutPath (Join-Path $runDir 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $config, '-t', $a00Checkpoint,
        '--seed', '0', '--use-amp'
    )

$bestCheckpoint = Join-Path $runDir 'best_stg1.pth'
$lastCheckpoint = Join-Path $runDir 'last.pth'
foreach ($path in @($bestCheckpoint, $lastCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Training checkpoint is missing: $path"
    }
}

Invoke-PythonStep -Name 'best_mask_alignment' `
    -StdoutPath (Join-Path $queueDir 's_rep1_2_best_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py', '--config', $config,
        '--checkpoint', $bestCheckpoint, '--batches', '20',
        '--output', (Join-Path $reportDir 'mask_alignment_best.json')
    )

Invoke-PythonStep -Name 'last_mask_alignment' `
    -StdoutPath (Join-Path $queueDir 's_rep1_2_last_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py', '--config', $config,
        '--checkpoint', $lastCheckpoint, '--batches', '20',
        '--output', (Join-Path $reportDir 'mask_alignment_last.json')
    )

Invoke-PythonStep -Name 'summarize' `
    -StdoutPath (Join-Path $queueDir 's_rep1_2_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_2.py',
        '--control-log', $controlLog,
        '--rep11-log', $rep11Log,
        '--rep12-log', (Join-Path $runDir 'log.txt'),
        '--output', (Join-Path $reportDir 'final_summary.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'
