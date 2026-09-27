$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$workspace = 'E:\two_paper'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$checkpoint = 'E:\two_paper\outputs\C_ONLY_GQ1_B8A4_20E_TESTDEV\seed0\best_stg1.pth'
$auxConfig = 'experiments\phase_s\s_sqmi1_aux_c_gq1_b16a2_12e_testdev_local.yml'
$initConfig = 'experiments\phase_s\s_sqmi1_init_c_gq1_b16a2_12e_testdev_local.yml'
$auxRun = 'E:\two_paper\outputs\S_SQMI1_AUX_C_GQ1_B16A2_12E_TESTDEV\seed0'
$initRun = 'E:\two_paper\outputs\S_SQMI1_INIT_C_GQ1_B16A2_12E_TESTDEV\seed0'
$queueDir = 'E:\two_paper\reports\144_sqmi1_preflight\queue'
$queueLog = Join-Path $queueDir 'queue.log'
$statusFile = Join-Path $queueDir 'status.json'

New-Item -ItemType Directory -Force -Path $queueDir | Out-Null

function Write-Status {
    param([string]$State, [string]$Stage, [string]$Message)
    $status = [ordered]@{
        updated_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        state = $State
        stage = $Stage
        message = $Message
        aux_run = $auxRun
        init_run = $initRun
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
    Add-Content -LiteralPath $queueLog -Value "$($status.updated_at) [$State] $Stage $Message" -Encoding UTF8
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments, [string]$StdoutPath)
    $parent = Split-Path -Parent $StdoutPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
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
foreach ($run in @($auxRun, $initRun)) {
    if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
        throw "Refusing to overwrite an existing formal run: $run"
    }
}

try {
    Write-Status -State 'running' -Stage 'preflight' -Message 'paired protocol started'
    Invoke-Step `
        -Name 'preflight' `
        -StdoutPath (Join-Path $queueDir 'preflight_console.log') `
        -Arguments @('-u', 'tools\preflight_s_sqmi1.py')

    Invoke-Step `
        -Name 'train_aux' `
        -StdoutPath (Join-Path $auxRun 'train_console.log') `
        -Arguments @(
            '-u', 'train.py', '-c', $auxConfig, '-t', $checkpoint,
            '--seed', '0', '--use-amp'
        )

    if (-not (Test-Path -LiteralPath (Join-Path $auxRun 'best_stg1.pth') -PathType Leaf)) {
        throw 'AUX training completed without best_stg1.pth'
    }

    Invoke-Step `
        -Name 'train_init' `
        -StdoutPath (Join-Path $initRun 'train_console.log') `
        -Arguments @(
            '-u', 'train.py', '-c', $initConfig, '-t', $checkpoint,
            '--seed', '0', '--use-amp'
        )

    if (-not (Test-Path -LiteralPath (Join-Path $initRun 'best_stg1.pth') -PathType Leaf)) {
        throw 'INIT training completed without best_stg1.pth'
    }
    Write-Status -State 'complete' -Stage 'pair' -Message 'AUX and INIT both completed'
}
catch {
    Write-Status -State 'failed' -Stage 'queue' -Message $_.Exception.Message
    throw
}
