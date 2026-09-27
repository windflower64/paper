$ErrorActionPreference = 'Stop'

# Windows PowerShell 5.1 may decode a BOM-less UTF-8 script as ANSI.
# Keep this runtime controller ASCII-only; human-facing reports remain Chinese.
$processEnvironment = [Environment]::GetEnvironmentVariables()
$processPath = [string]$processEnvironment['Path']
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $processPath, 'Process')

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$fsd0Completion = 'E:\two_paper\reports\26_fsd_down\S_FSD0\QUEUE_COMPLETED.txt'
$fsd0Summary = 'E:\two_paper\reports\26_fsd_down\S_FSD0\final_summary.json'
$fsd0Log = 'E:\two_paper\runs\26_fsd_down\S_FSD0_DIRECT_S8_S16_B16_60E\seed0\log.txt'
$controlLog = 'E:\two_paper\runs\26_fsd_down\S_FSD0_STD_REINIT_S8_S16_B16_60E\seed0\log.txt'
$a00Log = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\log.txt'
$config = 'experiments\phase_s\s_fsd1_functional_init_s8_s16_b16_60e_local.yml'
$originalCheckpoint = 'E:\two_paper\weights\dfine_n_coco.pth'
$functionalCheckpoint = 'E:\two_paper\weights\s_fsd1_functional_init_coco_tuning.pth'
$reportDir = 'E:\two_paper\reports\26_fsd_down\S_FSD1_FUNCTIONAL_INIT'
$functionalReport = Join-Path $reportDir 'functional_init.json'
$decisionReport = Join-Path $reportDir 'branch_decision.json'
$runDir = 'E:\two_paper\runs\26_fsd_down\S_FSD1_FUNCTIONAL_INIT_S8_S16_B16_60E\seed0'
$queueDir = 'E:\two_paper\runs\26_fsd_down\_queue'
$queueLog = Join-Path $queueDir 's_fsd1_after_control_queue.log'

New-Item -ItemType Directory -Force -Path $reportDir, $queueDir | Out-Null

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

Write-QueueLog 'WATCHER_STARTED waiting_for=FSD0_paired_control'
while (-not (Test-Path -LiteralPath $fsd0Completion -PathType Leaf)) {
    Start-Sleep -Seconds 30
}
if (-not (Test-Path -LiteralPath $fsd0Summary -PathType Leaf)) {
    throw "FSD0 completion exists but summary is missing: $fsd0Summary"
}

$summary = Get-Content -LiteralPath $fsd0Summary -Raw -Encoding UTF8 | ConvertFrom-Json
$machine = $summary.machine_decision
if ($null -eq $machine) {
    throw 'FSD0 summary does not contain machine_decision'
}
$deltaControl = [double]$machine.fsd_minus_control_ap
$fsd0AP = [double]$machine.fsd_best_ap
$controlAP = [double]$machine.control_best_ap
$a00AP = [double]$machine.a00_best_ap
$eligible = ($deltaControl -ge 0.002) -and ($fsd0AP -lt $a00AP)
$decision = [ordered]@{
    decision_time = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    fsd0_best_ap = $fsd0AP
    control_best_ap = $controlAP
    a00_best_ap = $a00AP
    fsd0_minus_control_ap = $deltaControl
    gate = 'FSD0 minus control >= 0.002 AP and FSD0 < A00'
    allow_fsd1 = $eligible
}
$decision | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $decisionReport -Encoding UTF8

if (-not $eligible) {
    Write-QueueLog "BRANCH_CLOSED eligible=false delta_control=$deltaControl fsd0_ap=$fsd0AP a00_ap=$a00AP"
    Set-Content -LiteralPath (Join-Path $reportDir 'FSD1_NOT_STARTED.txt') `
        -Value 'FSD1 was not started because the preregistered branch gate failed.' `
        -Encoding UTF8
    exit 0
}

try {
    Invoke-PythonStep -Name 'prepare_fsd1_functional_init' `
        -StdoutPath (Join-Path $queueDir 's_fsd1_functional_init.log') `
        -Arguments @(
            '-u', 'tools\prepare_s_fsd1_functional_init.py',
            '--config', $config, '--checkpoint', $originalCheckpoint,
            '--output-checkpoint', $functionalCheckpoint,
            '--report', $functionalReport, '--epochs', '3', '--batch-size', '16'
        )
} catch {
    Write-QueueLog 'FUNCTIONAL_MATCH_GATE_FAILED no_detector_training_started'
    Set-Content -LiteralPath (Join-Path $reportDir 'FUNCTIONAL_GATE_FAILED.txt') `
        -Value $_.Exception.Message -Encoding UTF8
    exit 0
}

$rows = Get-ValidationRowCount (Join-Path $runDir 'log.txt')
$trainArgs = @('-u', 'train.py', '-c', $config, '--seed', '0', '--use-amp')
if ($rows -gt 0) {
    $lastCheckpoint = Join-Path $runDir 'last.pth'
    if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) {
        throw "FSD1 has a partial log but no last.pth: $runDir"
    }
    $trainArgs += @('-r', $lastCheckpoint)
    Write-QueueLog "RESUME train_fsd1 rows=$rows"
} else {
    $trainArgs += @('-t', $functionalCheckpoint)
}

if ($rows -lt 60) {
    Invoke-PythonStep -Name 'train_fsd1_60e' -Arguments $trainArgs `
        -StdoutPath (Join-Path $runDir 'train_console.log')
}
$rows = Get-ValidationRowCount (Join-Path $runDir 'log.txt')
$bestCheckpoint = Join-Path $runDir 'best_stg1.pth'
if ($rows -lt 60 -or -not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
    throw "FSD1 did not complete 60 validation rows: rows=$rows"
}

$artifactDir = Join-Path $runDir 'artifacts'
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
foreach ($relative in @(
    $config,
    'src\nn\backbone\hgnetv2.py',
    'tools\prepare_s_fsd1_functional_init.py',
    'tools\summarize_s_fsd1.py',
    'tools\launch_s_fsd1_after_control.ps1'
)) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $relative) -Destination $artifactDir -Force
}
Copy-Item -LiteralPath $functionalReport -Destination $artifactDir -Force

$causalDir = Join-Path $runDir 'final_fixed_best_validation'
foreach ($mode in @('full', 'll_only', 'shift_hf', 'phase_permute', 'spatial_only')) {
    Invoke-PythonStep -Name "validate_fsd1_$mode" `
        -StdoutPath (Join-Path $queueDir "s_fsd1_validate_$mode.log") `
        -Arguments @(
            '-u', 'evaluate_custom_sizes_and_importance.py', '--repo', $repoRoot,
            '--config', $config, '--checkpoint', $bestCheckpoint,
            '--output-dir', $causalDir, '--weight-source', 'ema',
            '--skip-importance', '--fsd-mode', $mode
        )
}

Invoke-PythonStep -Name 'summarize_fsd1' `
    -StdoutPath (Join-Path $queueDir 's_fsd1_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_fsd1.py',
        '--fsd1-log', (Join-Path $runDir 'log.txt'),
        '--fsd0-log', $fsd0Log, '--control-log', $controlLog,
        '--a00-log', $a00Log, '--causal-dir', $causalDir,
        '--functional-report', $functionalReport,
        '--output', (Join-Path $reportDir 'final_summary.json')
    )

Set-Content -LiteralPath (Join-Path $reportDir 'QUEUE_COMPLETED.txt') `
    -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') FSD1 completed" -Encoding UTF8
Write-QueueLog 'QUEUE_COMPLETED'
