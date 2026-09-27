$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$sourcePid = 10068
$checkpoint = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\best_stg1.pth'
$runRoot = 'E:\two_paper\runs\21_public_reproduction'
$queueRoot = Join-Path $runRoot '_queue'
$queueLog = Join-Path $queueRoot 's_rep1_pair_queue.log'

New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Add-Content -LiteralPath $queueLog -Value $line -Encoding UTF8
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$LogPath
    )
    $logDirectory = Split-Path -Parent $LogPath
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    Write-QueueLog "START $Name"
    $stderrPath = "$LogPath.stderr.log"
    $process = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $LogPath `
        -RedirectStandardError $stderrPath `
        -Wait `
        -PassThru
    $exitCode = $process.ExitCode
    if ($exitCode -ne 0) {
        Write-QueueLog "FAILED $Name exit_code=$exitCode stdout=$LogPath stderr=$stderrPath"
        throw "$Name failed with exit code $exitCode"
    }
    Write-QueueLog "DONE $Name stdout=$LogPath stderr=$stderrPath"
}

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Python executable is missing: $pythonExe"
}
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "A00 checkpoint is missing: $checkpoint"
}

Write-QueueLog "QUEUE_STARTED wait_pid=$sourcePid"
while ($null -ne (Get-Process -Id $sourcePid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 30
}
Write-QueueLog "SOURCE_TRAINING_EXITED pid=$sourcePid"
Start-Sleep -Seconds 10

$preflightLog = Join-Path $queueRoot 'preflight_batch16.log'
Invoke-PythonStep -Name 'preflight_batch16' -LogPath $preflightLog -Arguments @(
    '-u',
    'tools\preflight_s_rep1_spar.py',
    '--config', 'experiments\phase_s\s_rep1_spar_sam_feature_regularization_local.yml',
    '--baseline-config', 'experiments\phase_s\visible_60e_base_local.yml',
    '--gradient-batch', '16',
    '--output', 'E:\two_paper\reports\21_public_reproduction\S_REP1_SPAR\preflight_batch16.json'
)

$controlLog = Join-Path $runRoot 'S_REP1_A00FT10_CONTROL\seed0\train_console.log'
Invoke-PythonStep -Name 'a00ft10_control' -LogPath $controlLog -Arguments @(
    '-u',
    'train.py',
    '-c', 'experiments\phase_s\s_rep1_a00ft10_control_local.yml',
    '-t', $checkpoint,
    '--seed', '0',
    '--use-amp'
)

$sparLog = Join-Path $runRoot 'S_REP1_SPAR_A00FT10\seed0\train_console.log'
Invoke-PythonStep -Name 'spar_a00ft10' -LogPath $sparLog -Arguments @(
    '-u',
    'train.py',
    '-c', 'experiments\phase_s\s_rep1_spar_a00ft10_local.yml',
    '-t', $checkpoint,
    '--seed', '0',
    '--use-amp'
)

Write-QueueLog 'QUEUE_COMPLETE'
