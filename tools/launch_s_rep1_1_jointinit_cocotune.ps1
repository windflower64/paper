$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$cocoCheckpoint = 'E:\two_paper\weights\dfine_n_coco.pth'
$a00Log = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\log.txt'
$baselineConfig = 'experiments\phase_s\visible_60e_base_local.yml'
$pilotConfig = 'experiments\phase_s\s_rep1_1_jointinit_cocotune_pilot2_local.yml'
$fullConfig = 'experiments\phase_s\s_rep1_1_jointinit_cocotune_progressive_w025_b32_60e_local.yml'
$pilotRun = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_COCOTUNE_GMAX100_PILOT2\seed0'
$fullRun = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_COCOTUNE_GMAX100_PROGRESSIVE_W025_B32_60E\seed0'
$reportDir = 'E:\two_paper\reports\21_public_reproduction\S_REP1_1_JOINTINIT_COCOTUNE_GMAX100'
$queueDir = 'E:\two_paper\runs\21_public_reproduction\_queue'
$queueLog = Join-Path $queueDir 's_rep1_1_jointinit_cocotune_gmax100_queue.log'

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

foreach ($path in @($pythonExe, $cocoCheckpoint, $a00Log)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}
foreach ($runDir in @($pilotRun, $fullRun)) {
    if (Test-Path -LiteralPath (Join-Path $runDir 'log.txt') -PathType Leaf) {
        throw "Corrected JointInit output already contains a log: $runDir"
    }
}

Write-QueueLog 'QUEUE_STARTED method=REP1.1-JointInit-CocoTune-GMax100 epochs=60 batch=32 seed=0'

Invoke-PythonStep -Name 'coco_static_preflight_epoch0' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_cocotune_preflight_e0.log') `
    -Arguments @(
        '-u', 'tools\preflight_s_rep1_spar.py',
        '--config', $fullConfig, '--baseline-config', $baselineConfig,
        '--checkpoint', $cocoCheckpoint,
        '--expected-batch', '32', '--gradient-batch', '2', '--epoch', '0',
        '--output', (Join-Path $reportDir 'preflight_epoch0.json')
    )

Invoke-PythonStep -Name 'coco_static_preflight_epoch10' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_cocotune_preflight_e10.log') `
    -Arguments @(
        '-u', 'tools\preflight_s_rep1_spar.py',
        '--config', $fullConfig, '--baseline-config', $baselineConfig,
        '--checkpoint', $cocoCheckpoint,
        '--expected-batch', '32', '--gradient-batch', '2', '--epoch', '10',
        '--max-shared-gradient-ratio', '0.10',
        '--min-shared-gradient-cosine', '-0.20',
        '--output', (Join-Path $reportDir 'preflight_epoch10.json')
    )

Invoke-PythonStep -Name 'coco_full_batch32_amp_preflight' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_cocotune_batch32.log') `
    -Arguments @(
        '-u', 'tools\preflight_s_rep1_jointinit_batch32.py',
        '--config', $fullConfig, '--checkpoint', $cocoCheckpoint,
        '--expected-batch', '32', '--epoch', '0', '--amp-init-scale', '1024',
        '--output', (Join-Path $reportDir 'preflight_full_batch32.json')
    )

Invoke-PythonStep -Name 'coco_tuning_pilot2' `
    -StdoutPath (Join-Path $pilotRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $pilotConfig,
        '-t', $cocoCheckpoint, '--seed', '0', '--use-amp'
    )

$pilotConsole = Join-Path $pilotRun 'train_console.log'
$pilotLog = Join-Path $pilotRun 'log.txt'
$tuningPattern = [regex]::Escape("Tuning checkpoint from $cocoCheckpoint")
if (-not (Select-String -LiteralPath $pilotConsole -Pattern $tuningPattern -Quiet)) {
    throw 'Pilot console did not confirm the exact COCO tuning checkpoint'
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
    throw "COCO pilot failed its initialization gate: epoch1 AP=$pilotEpoch1Ap"
}
Write-QueueLog "PASS pilot_initialization_gate epoch1_ap=$pilotEpoch1Ap threshold=0.20"

Invoke-PythonStep -Name 'train_jointinit_cocotune_60e' `
    -StdoutPath (Join-Path $fullRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $fullConfig,
        '-t', $cocoCheckpoint, '--seed', '0', '--use-amp'
    )

$fullConsole = Join-Path $fullRun 'train_console.log'
if (-not (Select-String -LiteralPath $fullConsole -Pattern $tuningPattern -Quiet)) {
    throw 'Formal console did not confirm the exact COCO tuning checkpoint'
}

$bestCheckpoint = Join-Path $fullRun 'best_stg1.pth'
$lastCheckpoint = Join-Path $fullRun 'last.pth'
foreach ($path in @($bestCheckpoint, $lastCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Training checkpoint is missing: $path"
    }
}

Invoke-PythonStep -Name 'best_mask_alignment' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_cocotune_best_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py', '--config', $fullConfig,
        '--checkpoint', $bestCheckpoint, '--batches', '20',
        '--output', (Join-Path $reportDir 'mask_alignment_best.json')
    )

Invoke-PythonStep -Name 'last_mask_alignment' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_cocotune_last_alignment.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py', '--config', $fullConfig,
        '--checkpoint', $lastCheckpoint, '--batches', '20',
        '--output', (Join-Path $reportDir 'mask_alignment_last.json')
    )

Invoke-PythonStep -Name 'summarize_against_a00_first60' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_cocotune_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_jointinit.py',
        '--control-log', $a00Log,
        '--progressive-log', (Join-Path $fullRun 'log.txt'),
        '--best-alignment', (Join-Path $reportDir 'mask_alignment_best.json'),
        '--last-alignment', (Join-Path $reportDir 'mask_alignment_last.json'),
        '--output', (Join-Path $reportDir 'final_summary.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'
