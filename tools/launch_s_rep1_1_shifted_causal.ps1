$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$checkpoint = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\best_stg1.pth'
$runRoot = 'E:\two_paper\runs\21_public_reproduction'
$reportRoot = 'E:\two_paper\reports\21_public_reproduction\S_REP1_1_GENTLE'
$queueRoot = Join-Path $runRoot '_queue'
$queueLog = Join-Path $queueRoot 's_rep1_1_shifted_causal_queue.log'
$shiftedConfig = 'experiments\phase_s\s_rep1_1_spar_shift50_w025_ft6_lr02_local.yml'
$alignedConfig = 'experiments\phase_s\s_rep1_1_spar_w025_ft6_lr02_local.yml'
$shiftedDir = Join-Path $runRoot 'S_REP1_1_SPAR_SHIFT50_W025_FT6_LR02\seed0'

New-Item -ItemType Directory -Force -Path $queueRoot, $reportRoot | Out-Null

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
        Write-QueueLog "FAILED $Name exit_code=$($process.ExitCode) stdout=$StdoutPath stderr=$stderrPath"
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    Write-QueueLog "DONE $Name"
}

Write-QueueLog 'QUEUE_STARTED'

Invoke-PythonStep -Name 'shifted_preflight_batch16' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_shifted_preflight.log') `
    -Arguments @(
        '-u', 'tools\preflight_s_rep1_spar.py',
        '--config', $shiftedConfig,
        '--baseline-config', 'experiments\phase_s\visible_60e_base_local.yml',
        '--gradient-batch', '16',
        '--output', (Join-Path $reportRoot 'preflight_shift50_batch16.json')
    )

Invoke-PythonStep -Name 'shifted_mask_training' `
    -StdoutPath (Join-Path $shiftedDir 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $shiftedConfig,
        '-t', $checkpoint, '--seed', '0', '--use-amp'
    )

$bestCheckpoint = Join-Path $shiftedDir 'best_stg1.pth'
$lastCheckpoint = Join-Path $shiftedDir 'last.pth'
foreach ($path in @($bestCheckpoint, $lastCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing shifted-mask checkpoint: $path"
    }
}

# Evaluate both checkpoints against unmodified masks. The aligned config is
# intentional: it prevents the training-time shift from leaking into analysis.
Invoke-PythonStep -Name 'shifted_best_alignment' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_shifted_best_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $alignedConfig, '--checkpoint', $bestCheckpoint,
        '--batches', '20',
        '--output', (Join-Path $reportRoot 'shifted_training_alignment_best.json')
    )

Invoke-PythonStep -Name 'shifted_last_alignment' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_shifted_last_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $alignedConfig, '--checkpoint', $lastCheckpoint,
        '--batches', '20',
        '--output', (Join-Path $reportRoot 'shifted_training_alignment_last.json')
    )

Invoke-PythonStep -Name 'causal_summary' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_causal_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_1_causal.py',
        '--control-log', (Join-Path $runRoot 'S_REP1_1_A00FT6_LR02_CONTROL\seed0\log.txt'),
        '--aligned-log', (Join-Path $runRoot 'S_REP1_1_SPAR_W025_FT6_LR02\seed0\log.txt'),
        '--shifted-log', (Join-Path $shiftedDir 'log.txt'),
        '--output', (Join-Path $reportRoot 'causal_result.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'
