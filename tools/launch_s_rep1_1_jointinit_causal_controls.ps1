$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$cocoCheckpoint = 'E:\two_paper\weights\dfine_n_coco.pth'
$a00Log = 'E:\two_paper\outputs\dfine_n_visible_640x512_full_seed0\log.txt'
$alignedConfig = 'experiments\phase_s\s_rep1_1_jointinit_cocotune_progressive_w025_b32_60e_local.yml'
$boxConfig = 'experiments\phase_s\s_rep1_1_jointinit_cocotune_boxmask_w025_b32_60e_local.yml'
$shiftConfig = 'experiments\phase_s\s_rep1_1_jointinit_cocotune_shift50_w025_b32_60e_local.yml'
$alignedRun = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_COCOTUNE_GMAX100_PROGRESSIVE_W025_B32_60E\seed0'
$boxRun = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_COCOTUNE_BOXMASK_W025_B32_60E\seed0'
$shiftRun = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_COCOTUNE_SHIFT50_W025_B32_60E\seed0'
$reportDir = 'E:\two_paper\reports\21_public_reproduction\S_REP1_1_JOINTINIT_CAUSAL'
$queueDir = 'E:\two_paper\runs\21_public_reproduction\_queue'
$queueLog = Join-Path $queueDir 's_rep1_1_jointinit_causal_controls_queue.log'

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

foreach ($path in @(
    $pythonExe,
    $cocoCheckpoint,
    $a00Log,
    (Join-Path $alignedRun 'log.txt'),
    (Join-Path $alignedRun 'best_stg1.pth'),
    (Join-Path $reportDir 'preflight_box_epoch10.json'),
    (Join-Path $reportDir 'preflight_shift_epoch10.json'),
    (Join-Path $reportDir 'preflight_box_batch32.json'),
    (Join-Path $reportDir 'preflight_shift_batch32.json')
)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file is missing: $path"
    }
}
foreach ($runDir in @($boxRun, $shiftRun)) {
    if (Test-Path -LiteralPath (Join-Path $runDir 'log.txt') -PathType Leaf) {
        throw "Causal-control output already contains a training log: $runDir"
    }
}
foreach ($reportName in @(
    'preflight_box_epoch10.json',
    'preflight_shift_epoch10.json',
    'preflight_box_batch32.json',
    'preflight_shift_batch32.json'
)) {
    $report = Get-Content -LiteralPath (Join-Path $reportDir $reportName) -Encoding UTF8 | ConvertFrom-Json
    if ($report.status -ne 'PASS') {
        throw "Preflight did not pass: $reportName"
    }
}

$tuningPattern = [regex]::Escape("Tuning checkpoint from $cocoCheckpoint")
Write-QueueLog 'QUEUE_STARTED method=JointInit-Causal-Controls order=aligned_validation,box60,shift60 batch=32 seed=0'

Invoke-PythonStep -Name 'aligned_target_preference' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_aligned_target_preference.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $alignedConfig,
        '--checkpoint', (Join-Path $alignedRun 'best_stg1.pth'),
        '--batches', '20',
        '--output', (Join-Path $reportDir 'target_preference_aligned_best.json')
    )

Invoke-PythonStep -Name 'train_boxmask_jointinit_60e' `
    -StdoutPath (Join-Path $boxRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $boxConfig,
        '-t', $cocoCheckpoint, '--seed', '0', '--use-amp'
    )
if (-not (Select-String -LiteralPath (Join-Path $boxRun 'train_console.log') -Pattern $tuningPattern -Quiet)) {
    throw 'BoxMask console did not confirm the exact COCO tuning checkpoint'
}
if ((Get-Content -LiteralPath (Join-Path $boxRun 'log.txt') -Encoding UTF8).Count -ne 60) {
    throw 'BoxMask training did not produce exactly 60 validation rows'
}

Invoke-PythonStep -Name 'boxmask_target_preference' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_box_target_preference.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $alignedConfig,
        '--checkpoint', (Join-Path $boxRun 'best_stg1.pth'),
        '--batches', '20',
        '--output', (Join-Path $reportDir 'target_preference_box_best.json')
    )

Invoke-PythonStep -Name 'train_shift50_jointinit_60e' `
    -StdoutPath (Join-Path $shiftRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $shiftConfig,
        '-t', $cocoCheckpoint, '--seed', '0', '--use-amp'
    )
if (-not (Select-String -LiteralPath (Join-Path $shiftRun 'train_console.log') -Pattern $tuningPattern -Quiet)) {
    throw 'ShiftMask console did not confirm the exact COCO tuning checkpoint'
}
if ((Get-Content -LiteralPath (Join-Path $shiftRun 'log.txt') -Encoding UTF8).Count -ne 60) {
    throw 'ShiftMask training did not produce exactly 60 validation rows'
}

Invoke-PythonStep -Name 'shift50_target_preference' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_shift_target_preference.log') `
    -Arguments @(
        '-u', 'tools\validate_s_rep1_mask_alignment.py',
        '--config', $alignedConfig,
        '--checkpoint', (Join-Path $shiftRun 'best_stg1.pth'),
        '--batches', '20',
        '--output', (Join-Path $reportDir 'target_preference_shift_best.json')
    )

Invoke-PythonStep -Name 'summarize_jointinit_causal_controls' `
    -StdoutPath (Join-Path $queueDir 's_rep1_1_jointinit_causal_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_jointinit_causal_controls.py',
        '--a00-log', $a00Log,
        '--aligned-log', (Join-Path $alignedRun 'log.txt'),
        '--box-log', (Join-Path $boxRun 'log.txt'),
        '--shift-log', (Join-Path $shiftRun 'log.txt'),
        '--aligned-validation', (Join-Path $reportDir 'target_preference_aligned_best.json'),
        '--box-validation', (Join-Path $reportDir 'target_preference_box_best.json'),
        '--shift-validation', (Join-Path $reportDir 'target_preference_shift_best.json'),
        '--output', (Join-Path $reportDir 'final_summary.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'

