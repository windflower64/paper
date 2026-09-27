$ErrorActionPreference = 'Stop'

# Managed Windows shells may expose both PATH and Path. Start-Process rejects
# that duplicate dictionary, so normalize only this queue helper process.
$processEnvironment = [Environment]::GetEnvironmentVariables()
$processPath = [string]$processEnvironment['Path']
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $processPath, 'Process')

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$cocoCheckpoint = 'E:\two_paper\weights\dfine_n_coco.pth'
$a00Log = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\log.txt'
$fsdConfig = 'experiments\phase_s\s_fsd0_direct_s8_s16_b16_60e_local.yml'
$controlConfig = 'experiments\phase_s\s_fsd0_std_reinit_s8_s16_b16_60e_local.yml'
$fsdRun = 'E:\two_paper\runs\26_fsd_down\S_FSD0_DIRECT_S8_S16_B16_60E\seed0'
$controlRun = 'E:\two_paper\runs\26_fsd_down\S_FSD0_STD_REINIT_S8_S16_B16_60E\seed0'
$reportDir = 'E:\two_paper\reports\26_fsd_down\S_FSD0'
$queueDir = 'E:\two_paper\runs\26_fsd_down\_queue'
$queueLog = Join-Path $queueDir 's_fsd0_pair_queue.log'

New-Item -ItemType Directory -Force -Path $queueDir, $reportDir | Out-Null

function Write-QueueLog {
    param([string]$Message)
    Add-Content -LiteralPath $queueLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" -Encoding UTF8
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

function Invoke-Training {
    param([string]$Name, [string]$Config, [string]$RunDir)
    $logPath = Join-Path $RunDir 'log.txt'
    $bestPath = Join-Path $RunDir 'best_stg1.pth'
    $lastPath = Join-Path $RunDir 'last.pth'
    $rows = Get-ValidationRowCount $logPath
    if ($rows -ge 60 -and (Test-Path -LiteralPath $bestPath -PathType Leaf)) {
        Write-QueueLog "SKIP $Name completed_rows=$rows"
        return
    }
    $arguments = @('-u', 'train.py', '-c', $Config, '--seed', '0', '--use-amp')
    if ($rows -gt 0) {
        if (-not (Test-Path -LiteralPath $lastPath -PathType Leaf)) {
            throw "$Name has a partial log but no last.pth: $RunDir"
        }
        $arguments += @('-r', $lastPath)
        Write-QueueLog "RESUME $Name rows=$rows checkpoint=$lastPath"
    } else {
        $arguments += @('-t', $cocoCheckpoint)
    }
    Invoke-PythonStep -Name $Name -Arguments $arguments `
        -StdoutPath (Join-Path $RunDir 'train_console.log')
    $rows = Get-ValidationRowCount $logPath
    if ($rows -lt 60 -or -not (Test-Path -LiteralPath $bestPath -PathType Leaf)) {
        throw "$Name did not complete 60 validation rows: rows=$rows"
    }
}

foreach ($path in @($pythonExe, $cocoCheckpoint, $a00Log)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}

foreach ($runDir in @($fsdRun, $controlRun)) {
    $artifactDir = Join-Path $runDir 'artifacts'
    New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
    foreach ($relative in @(
        $fsdConfig,
        $controlConfig,
        'experiments\phase_s\visible_60e_base_local.yml',
        'src\nn\backbone\hgnetv2.py',
        'tools\preflight_s_fsd0.py',
        'tools\summarize_s_fsd0.py',
        'tools\launch_s_fsd0_pair.ps1'
    )) {
        Copy-Item -LiteralPath (Join-Path $repoRoot $relative) -Destination $artifactDir -Force
    }
    Copy-Item -LiteralPath (Join-Path $reportDir 'preflight.json') `
        -Destination $artifactDir -Force -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath $artifactDir -File |
        Get-FileHash -Algorithm SHA256 |
        Select-Object Path, Hash |
        ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $artifactDir 'SHA256SUMS.json') -Encoding UTF8
}

$preflight = Join-Path $reportDir 'preflight.json'
if (-not (Test-Path -LiteralPath $preflight -PathType Leaf)) {
    Invoke-PythonStep -Name 'fsd0_preflight_batch16' `
        -StdoutPath (Join-Path $queueDir 's_fsd0_preflight.log') `
        -Arguments @(
            '-u', 'tools\preflight_s_fsd0.py', '--config', $fsdConfig,
            '--control-config', $controlConfig, '--checkpoint', $cocoCheckpoint,
            '--expected-batch', '16', '--output', $preflight
        )
}

Write-QueueLog 'QUEUE_STARTED method=FSD0 pair=STD-REINIT epochs=60 batch=16 seed=0'
Invoke-Training -Name 'train_fsd0_60e' -Config $fsdConfig -RunDir $fsdRun

$causalDir = Join-Path $fsdRun 'final_fixed_best_validation'
$bestCheckpoint = Join-Path $fsdRun 'best_stg1.pth'
foreach ($mode in @('full', 'll_only', 'shift_hf', 'phase_permute', 'spatial_only')) {
    Invoke-PythonStep -Name "validate_fsd0_$mode" `
        -StdoutPath (Join-Path $queueDir "s_fsd0_validate_$mode.log") `
        -Arguments @(
            '-u', 'evaluate_custom_sizes_and_importance.py', '--repo', $repoRoot,
            '--config', $fsdConfig, '--checkpoint', $bestCheckpoint,
            '--output-dir', $causalDir, '--weight-source', 'ema',
            '--skip-importance', '--fsd-mode', $mode
        )
}

Invoke-Training -Name 'train_std_reinit_control_60e' -Config $controlConfig -RunDir $controlRun

Invoke-PythonStep -Name 'summarize_fsd0_pair' `
    -StdoutPath (Join-Path $queueDir 's_fsd0_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_fsd0.py',
        '--fsd-log', (Join-Path $fsdRun 'log.txt'),
        '--control-log', (Join-Path $controlRun 'log.txt'),
        '--causal-dir', $causalDir, '--a00-log', $a00Log,
        '--output', (Join-Path $reportDir 'final_summary.json')
    )

Set-Content -LiteralPath (Join-Path $reportDir 'QUEUE_COMPLETED.txt') `
    -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') FSD0 paired queue completed" -Encoding UTF8
Write-QueueLog 'QUEUE_COMPLETED'
