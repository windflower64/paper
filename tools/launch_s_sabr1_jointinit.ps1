$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$cocoCheckpoint = 'E:\two_paper\weights\dfine_n_coco.pth'
$pilotConfig = 'experiments\phase_s\s_sabr1_jointinit_cocotune_pilot2_local.yml'
$fullConfig = 'experiments\phase_s\s_sabr1_jointinit_cocotune_w025_b32_60e_local.yml'
$pilotRun = 'E:\two_paper\runs\21_public_reproduction\S_SABR1_JOINTINIT_COCOTUNE_PILOT2\seed0'
$fullRun = 'E:\two_paper\runs\21_public_reproduction\S_SABR1_JOINTINIT_COCOTUNE_W025_B32_60E\seed0'
$reportDir = 'E:\two_paper\reports\21_public_reproduction\S_SABR1_JOINTINIT'
$queueDir = 'E:\two_paper\runs\21_public_reproduction\_queue'
$queueLog = Join-Path $queueDir 's_sabr1_jointinit_queue.log'

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

foreach ($path in @($pythonExe, $cocoCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}
foreach ($preflightName in @('preflight_epoch0.json', 'preflight_epoch10.json', 'preflight_full_batch32.json')) {
    $preflightPath = Join-Path $reportDir $preflightName
    if (-not (Test-Path -LiteralPath $preflightPath -PathType Leaf)) {
        throw "Required preflight is missing: $preflightPath"
    }
    $preflight = Get-Content -LiteralPath $preflightPath -Encoding UTF8 | ConvertFrom-Json
    if ($preflight.status -ne 'PASS') {
        throw "Preflight did not pass: $preflightPath"
    }
}
foreach ($runDir in @($pilotRun, $fullRun)) {
    if (Test-Path -LiteralPath (Join-Path $runDir 'log.txt') -PathType Leaf) {
        throw "SABR output already contains a log: $runDir"
    }
}

Write-QueueLog 'QUEUE_STARTED method=SABR1-JointInit epochs=60 batch=32 seed=0'

Invoke-PythonStep -Name 'sabr1_pilot2' `
    -StdoutPath (Join-Path $pilotRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $pilotConfig,
        '-t', $cocoCheckpoint, '--seed', '0', '--use-amp'
    )

$tuningPattern = [regex]::Escape("Tuning checkpoint from $cocoCheckpoint")
$pilotConsole = Join-Path $pilotRun 'train_console.log'
$pilotLog = Join-Path $pilotRun 'log.txt'
if (-not (Select-String -LiteralPath $pilotConsole -Pattern $tuningPattern -Quiet)) {
    throw 'Pilot did not confirm the exact COCO tuning checkpoint'
}
$pilotRows = @(
    Get-Content -LiteralPath $pilotLog -Encoding UTF8 |
        ForEach-Object { try { $_ | ConvertFrom-Json } catch { } }
)
if ($pilotRows.Count -ne 2) {
    throw "Pilot must produce exactly two validation rows, got $($pilotRows.Count)"
}
$pilotEpoch1Ap = [double]$pilotRows[1].test_coco_eval_bbox[0]
if ($pilotEpoch1Ap -lt 0.20) {
    Write-QueueLog "FAILED pilot_initialization_gate epoch1_ap=$pilotEpoch1Ap threshold=0.20"
    throw "SABR pilot failed initialization gate: epoch1 AP=$pilotEpoch1Ap"
}
Write-QueueLog "PASS pilot_initialization_gate epoch1_ap=$pilotEpoch1Ap threshold=0.20"

Invoke-PythonStep -Name 'train_sabr1_jointinit_60e' `
    -StdoutPath (Join-Path $fullRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $fullConfig,
        '-t', $cocoCheckpoint, '--seed', '0', '--use-amp'
    )

$fullConsole = Join-Path $fullRun 'train_console.log'
if (-not (Select-String -LiteralPath $fullConsole -Pattern $tuningPattern -Quiet)) {
    throw 'Formal training did not confirm the exact COCO tuning checkpoint'
}
foreach ($name in @('best_stg1.pth', 'last.pth', 'log.txt')) {
    $path = Join-Path $fullRun $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Formal SABR output is missing: $path"
    }
}
Write-QueueLog 'QUEUE_COMPLETE'

