$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$checkpoint = 'E:\two_paper\outputs\C_ONLY_GQ1_B8A4_20E_TESTDEV\seed0\best_stg1.pth'
$config = 'experiments\phase_s\s_sqmi2_s4_mask_warmup_b16_6e_testdev_local.yml'
$runDir = 'E:\two_paper\outputs\S_SQMI2_S4_MASK_WARMUP_B16_6E_TESTDEV\seed0'
$reportDir = 'E:\two_paper\reports\146_sqmi2_s4'
$queueDir = Join-Path $reportDir 'queue'
$statusFile = Join-Path $queueDir 'status.json'
$queueLog = Join-Path $queueDir 'queue.log'

New-Item -ItemType Directory -Force -Path $queueDir | Out-Null

function Write-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    $status = [ordered]@{
        updated_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        state = $State
        stage = $Stage
        message = $Message
        run_dir = $runDir
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
    Add-Content -LiteralPath $queueLog -Value "$($status.updated_at) [$State] $Stage $Message" -Encoding UTF8
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments, [string]$StdoutPath)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StdoutPath) | Out-Null
    $stderrPath = "$StdoutPath.stderr.log"
    Write-Status -State 'running' -Stage $Name -Message 'started'
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
        Write-Status -State 'failed' -Stage $Name -Message "exit_code=$($process.ExitCode); stderr=$stderrPath"
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    Write-Status -State 'running' -Stage $Name -Message 'completed'
}

foreach ($path in @($pythonExe, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}
if (Test-Path -LiteralPath (Join-Path $runDir 'log.txt') -PathType Leaf) {
    throw "Refusing to overwrite existing S-QMI2 run: $runDir"
}

try {
    Invoke-Step `
        -Name 'strict_preflight' `
        -StdoutPath (Join-Path $queueDir 'preflight_console.log') `
        -Arguments @('-u', 'tools\preflight_s_sqmi2_s4.py')

    Invoke-Step `
        -Name 'mask_warmup' `
        -StdoutPath (Join-Path $runDir 'train_console.log') `
        -Arguments @(
            '-u', 'train.py', '-c', $config, '-t', $checkpoint,
            '--seed', '0', '--use-amp'
        )

    Invoke-Step `
        -Name 'quality_selection' `
        -StdoutPath (Join-Path $queueDir 'quality_console.log') `
        -Arguments @('-u', 'tools\select_sqmi2_mask_checkpoint.py')

    $quality = Get-Content -LiteralPath (Join-Path $reportDir 'quality_selection.json') -Encoding UTF8 | ConvertFrom-Json
    if ($quality.status -eq 'pass') {
        Write-Status -State 'complete' -Stage 'quality_pass' -Message "selected_epoch=$($quality.selected_epoch); quality-gate design is allowed"
    }
    else {
        Write-Status -State 'complete' -Stage 'quality_stop' -Message "selected_epoch=$($quality.selected_epoch); thresholds not met; no detection training"
    }
}
catch {
    Write-Status -State 'failed' -Stage 'queue' -Message $_.Exception.Message
    throw
}
