$ErrorActionPreference = 'Stop'

# Keep this controller ASCII-only for Windows PowerShell 5.1.
$processEnvironment = [Environment]::GetEnvironmentVariables()
$processPath = [string]$processEnvironment['Path']
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $processPath, 'Process')

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$config = 'experiments\phase_s\s_tflr1_joint_localization_b16_60e_local.yml'
$checkpoint = 'E:\two_paper\weights\dfine_n_coco.pth'
$a00Log = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\log.txt'
$runDir = 'E:\two_paper\runs\27_tflr\S_TFLR1_JOINT_LOCALIZATION_B16_60E\seed0'
$reportDir = 'E:\two_paper\reports\27_tflr\S_TFLR1'
$preflight = Join-Path $reportDir 'preflight.json'
$localCausality = 'E:\two_paper\reports\27_tflr\S_TFLR1\fsd_target_local_causality.json'
$queueDir = 'E:\two_paper\runs\27_tflr\_queue'
$queueLog = Join-Path $queueDir 's_tflr1_queue.log'

New-Item -ItemType Directory -Force -Path $runDir, $reportDir, $queueDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    Add-Content -LiteralPath $queueLog `
        -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" -Encoding UTF8
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments, [string]$StdoutPath)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StdoutPath) | Out-Null
    $stderrPath = "$StdoutPath.stderr.log"
    Write-QueueLog "START $Name"
    $process = Start-Process -FilePath $pythonExe -ArgumentList $Arguments `
        -WorkingDirectory $repoRoot -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutPath -RedirectStandardError $stderrPath `
        -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        Write-QueueLog "FAILED $Name exit_code=$($process.ExitCode) stderr=$stderrPath"
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    Write-QueueLog "DONE $Name"
}

function Get-ValidationRowCount {
    param([string]$LogPath)
    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) { return 0 }
    return @(
        Get-Content -LiteralPath $LogPath -Encoding UTF8 |
            ForEach-Object { try { $_ | ConvertFrom-Json } catch { } } |
            Where-Object { $null -ne $_.test_coco_eval_bbox }
    ).Count
}

foreach ($required in @($pythonExe, $checkpoint, $a00Log, $preflight, $localCausality)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}

$artifactDir = Join-Path $runDir 'artifacts'
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
foreach ($relative in @(
    $config,
    'src\zoo\dfine\dfine.py',
    'src\nn\backbone\hgnetv2.py',
    'evaluate_custom_sizes_and_importance.py',
    'tools\preflight_s_tflr1.py',
    'tools\summarize_s_tflr1.py',
    'tools\launch_s_tflr1.ps1'
)) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $relative) -Destination $artifactDir -Force
}
Copy-Item -LiteralPath $preflight, $localCausality -Destination $artifactDir -Force

$logPath = Join-Path $runDir 'log.txt'
$rows = Get-ValidationRowCount $logPath
$trainArgs = @('-u', 'train.py', '-c', $config, '--seed', '0', '--use-amp')
if ($rows -gt 0) {
    $lastCheckpoint = Join-Path $runDir 'last.pth'
    if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
        throw "TFLR1 has a partial log but no last.pth: $runDir"
    }
    $trainArgs += @('-r', $lastCheckpoint)
    Write-QueueLog "RESUME train_tflr1 rows=$rows"
} else {
    $trainArgs += @('-t', $checkpoint)
}

if ($rows -lt 60) {
    Invoke-PythonStep -Name 'train_tflr1_60e' -Arguments $trainArgs `
        -StdoutPath (Join-Path $runDir 'train_console.log')
}
$rows = Get-ValidationRowCount $logPath
$bestCheckpoint = Join-Path $runDir 'best_stg1.pth'
if ($rows -lt 60 -or -not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
    throw "TFLR1 did not complete 60 validation rows: rows=$rows"
}

$causalDir = Join-Path $runDir 'final_fixed_best_validation'
foreach ($mode in @('full', 'zero', 'shifted', 'swap_direction')) {
    Invoke-PythonStep -Name "validate_tflr1_$mode" `
        -StdoutPath (Join-Path $queueDir "validate_tflr1_$mode.log") `
        -Arguments @(
            '-u', 'evaluate_custom_sizes_and_importance.py', '--repo', $repoRoot,
            '--config', $config, '--checkpoint', $bestCheckpoint,
            '--output-dir', $causalDir, '--weight-source', 'ema',
            '--skip-importance', '--tflr-mode', $mode
        )
}

Invoke-PythonStep -Name 'summarize_tflr1' `
    -StdoutPath (Join-Path $queueDir 'summarize_tflr1.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_tflr1.py',
        '--train-log', $logPath, '--a00-log', $a00Log,
        '--causal-dir', $causalDir, '--preflight', $preflight,
        '--local-causality', $localCausality,
        '--output', (Join-Path $reportDir 'final_summary.json')
    )

Set-Content -LiteralPath (Join-Path $reportDir 'QUEUE_COMPLETED.txt') `
    -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') TFLR1 completed" -Encoding UTF8
Write-QueueLog 'QUEUE_COMPLETED'
