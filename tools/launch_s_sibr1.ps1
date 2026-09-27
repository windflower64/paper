$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$startCheckpoint = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\best_stg1.pth'
$controlCheckpoint = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_A00FT6_LR02_CONTROL\seed0\best_stg1.pth'
$controlLog = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_A00FT6_LR02_CONTROL\seed0\log.txt'
$config = 'experiments\phase_s\s_sibr1_sam_incoherent_boundary_retention_ft6_lr02_local.yml'
$runDir = 'E:\two_paper\runs\21_public_reproduction\S_SIBR1_INCOHERENT_BOUNDARY_RETENTION_FT6_LR02\seed0'
$reportDir = 'E:\two_paper\reports\21_public_reproduction\S_SIBR1'
$queueDir = 'E:\two_paper\runs\21_public_reproduction\_queue'
$queueLog = Join-Path $queueDir 's_sibr1_queue.log'

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

foreach ($path in @($pythonExe, $startCheckpoint, $controlCheckpoint, $controlLog)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}
if (Test-Path -LiteralPath (Join-Path $runDir 'log.txt') -PathType Leaf) {
    throw "SIBR1 run already contains a training log: $runDir"
}

Write-QueueLog 'QUEUE_STARTED method=SIBR1 actual_transition=S8_to_S16 batch=16 epochs=6'

Invoke-PythonStep -Name 'preflight_batch16' `
    -StdoutPath (Join-Path $queueDir 's_sibr1_preflight.log') `
    -Arguments @(
        '-u', 'tools\preflight_s_sbrd1.py', '--config', $config,
        '--checkpoint', $startCheckpoint, '--gradient-batch', '16',
        '--output', (Join-Path $reportDir 'preflight_batch16.json')
    )

Invoke-PythonStep -Name 'a00_start_retention' `
    -StdoutPath (Join-Path $queueDir 's_sibr1_start_retention.log') `
    -Arguments @(
        '-u', 'tools\validate_s_sibr1_retention.py', '--config', $config,
        '--checkpoint', $startCheckpoint, '--batches', '10',
        '--output', (Join-Path $reportDir 'retention_a00_start.json')
    )

Invoke-PythonStep -Name 'control_best_retention' `
    -StdoutPath (Join-Path $queueDir 's_sibr1_control_retention.log') `
    -Arguments @(
        '-u', 'tools\validate_s_sibr1_retention.py', '--config', $config,
        '--checkpoint', $controlCheckpoint, '--batches', '10',
        '--output', (Join-Path $reportDir 'retention_control_best.json')
    )

Invoke-PythonStep -Name 'train_sibr1' `
    -StdoutPath (Join-Path $runDir 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $config, '-t', $startCheckpoint,
        '--seed', '0', '--use-amp'
    )

$bestCheckpoint = Join-Path $runDir 'best_stg1.pth'
$lastCheckpoint = Join-Path $runDir 'last.pth'
foreach ($path in @($bestCheckpoint, $lastCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Training checkpoint is missing: $path"
    }
}

Invoke-PythonStep -Name 'best_retention' `
    -StdoutPath (Join-Path $queueDir 's_sibr1_best_retention.log') `
    -Arguments @(
        '-u', 'tools\validate_s_sibr1_retention.py', '--config', $config,
        '--checkpoint', $bestCheckpoint, '--batches', '10',
        '--output', (Join-Path $reportDir 'retention_best.json')
    )

Invoke-PythonStep -Name 'last_retention' `
    -StdoutPath (Join-Path $queueDir 's_sibr1_last_retention.log') `
    -Arguments @(
        '-u', 'tools\validate_s_sibr1_retention.py', '--config', $config,
        '--checkpoint', $lastCheckpoint, '--batches', '10',
        '--output', (Join-Path $reportDir 'retention_last.json')
    )

Invoke-PythonStep -Name 'summarize' `
    -StdoutPath (Join-Path $queueDir 's_sibr1_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_sibr1.py', '--control-log', $controlLog,
        '--sibr-log', (Join-Path $runDir 'log.txt'),
        '--start-retention', (Join-Path $reportDir 'retention_a00_start.json'),
        '--control-retention', (Join-Path $reportDir 'retention_control_best.json'),
        '--best-retention', (Join-Path $reportDir 'retention_best.json'),
        '--last-retention', (Join-Path $reportDir 'retention_last.json'),
        '--output', (Join-Path $reportDir 'final_summary.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'
