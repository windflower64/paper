$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$runRoot = 'E:\two_paper\runs\21_public_reproduction'
$reportRoot = 'E:\two_paper\reports\21_public_reproduction\S_REP1_1_EXT12'
$queueRoot = Join-Path $runRoot '_queue'
$queueLog = Join-Path $queueRoot 's_rep1_1_resume6to12_queue.log'

$controlBaseDir = Join-Path $runRoot 'S_REP1_1_A00FT6_LR02_CONTROL\seed0'
$sparBaseDir = Join-Path $runRoot 'S_REP1_1_SPAR_W025_FT6_LR02\seed0'
$controlDir = Join-Path $runRoot 'S_REP1_1_A00FT12_LR02_CONTROL\seed0'
$sparDir = Join-Path $runRoot 'S_REP1_1_SPAR_W025_FT12_LR02\seed0'
$controlCheckpoint = Join-Path $controlBaseDir 'last.pth'
$sparCheckpoint = Join-Path $sparBaseDir 'last.pth'
$controlConfig = 'experiments\phase_s\s_rep1_1_a00ft12_lr02_control_resume_local.yml'
$sparConfig = 'experiments\phase_s\s_rep1_1_spar_w025_ft12_lr02_resume_local.yml'

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

foreach ($checkpoint in @($controlCheckpoint, $sparCheckpoint)) {
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Missing resume checkpoint: $checkpoint"
    }
}

Write-QueueLog 'QUEUE_STARTED'

Invoke-PythonStep -Name 'control_resume_epoch6_to_11' `
    -StdoutPath (Join-Path $controlDir 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $controlConfig,
        '-r', $controlCheckpoint, '--seed', '0', '--use-amp'
    )

Invoke-PythonStep -Name 'spar_resume_epoch6_to_11' `
    -StdoutPath (Join-Path $sparDir 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $sparConfig,
        '-r', $sparCheckpoint, '--seed', '0', '--use-amp'
    )

$bestCheckpoint = Join-Path $sparDir 'best_stg1.pth'
$lastCheckpoint = Join-Path $sparDir 'last.pth'
foreach ($checkpoint in @($bestCheckpoint, $lastCheckpoint)) {
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Missing extended SPAR checkpoint: $checkpoint"
    }
}

Invoke-PythonStep -Name 'extended_best_alignment' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_ext12_best_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $sparConfig, '--checkpoint', $bestCheckpoint,
        '--batches', '20',
        '--output', (Join-Path $reportRoot 'mask_alignment_best.json')
    )

Invoke-PythonStep -Name 'extended_last_alignment' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_ext12_last_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $sparConfig, '--checkpoint', $lastCheckpoint,
        '--batches', '20',
        '--output', (Join-Path $reportRoot 'mask_alignment_last.json')
    )

Invoke-PythonStep -Name 'summarize_extension' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_1_ext12_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_1_extension.py',
        '--control-base-log', (Join-Path $controlBaseDir 'log.txt'),
        '--control-resume-log', (Join-Path $controlDir 'log.txt'),
        '--spar-base-log', (Join-Path $sparBaseDir 'log.txt'),
        '--spar-resume-log', (Join-Path $sparDir 'log.txt'),
        '--output', (Join-Path $reportRoot 'paired_result_0_to_11.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'
