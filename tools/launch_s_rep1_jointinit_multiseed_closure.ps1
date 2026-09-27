$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$cocoCheckpoint = 'E:\two_paper\weights\dfine_n_coco.pth'
$baselineConfig = 'experiments\phase_s\visible_60e_base_amp1024_local.yml'
$jointConfig = 'experiments\phase_s\s_rep1_1_jointinit_cocotune_progressive_w025_b32_60e_local.yml'
$baselineRoot = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_PAIRED_A00'
$jointRoot = 'E:\two_paper\runs\21_public_reproduction\S_REP1_1_JOINTINIT_COCOTUNE_GMAX100_PROGRESSIVE_W025_B32_60E'
$seed0JointReport = 'E:\two_paper\reports\21_public_reproduction\S_REP1_1_JOINTINIT_COCOTUNE_GMAX100'
$reportRoot = 'E:\two_paper\reports\21_public_reproduction\S_REP1_1_JOINTINIT_MULTISEED'
$queueRoot = 'E:\two_paper\runs\21_public_reproduction\_queue'
$queueLog = Join-Path $queueRoot 's_rep1_jointinit_multiseed_closure_queue.log'

New-Item -ItemType Directory -Force -Path $baselineRoot, $reportRoot, $queueRoot | Out-Null

function Write-QueueLog {
    param([string]$Message)
    Add-Content -LiteralPath $queueLog `
        -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" `
        -Encoding UTF8
}

function Assert-EmptyRunDirectory {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        $items = @(Get-ChildItem -LiteralPath $Path -Force)
        if ($items.Count -gt 0) {
            throw "Refusing to overwrite non-empty run directory: $Path"
        }
    }
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

$seed0JointRun = Join-Path $jointRoot 'seed0'
$seed0Best = Join-Path $seed0JointRun 'best_stg1.pth'
$seed0Last = Join-Path $seed0JointRun 'last.pth'
$seed0BestAlignment = Join-Path $seed0JointReport 'mask_alignment_best.json'
$seed0LastAlignment = Join-Path $seed0JointReport 'mask_alignment_last.json'
foreach ($path in @($seed0Best, $seed0Last, $seed0BestAlignment, $seed0LastAlignment)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Seed0 closure asset is missing: $path"
    }
}

foreach ($seed in 0, 1, 2) {
    Assert-EmptyRunDirectory -Path (Join-Path $baselineRoot "seed$seed")
}
foreach ($seed in 1, 2) {
    Assert-EmptyRunDirectory -Path (Join-Path $jointRoot "seed$seed")
}

Write-QueueLog "QUEUE_STARTED pid=$PID protocol=paired_A00_vs_JointInit seeds=0,1,2 batch=32 epochs=60 amp_init_scale=1024"

# Seed0 JointInit already exists, but its historical A00 used the default AMP
# initial scale.  Re-run only the paired A00 seed0 and regenerate the summary.
$seed0BaselineRun = Join-Path $baselineRoot 'seed0'
Invoke-PythonStep -Name 'train_paired_a00_seed0' `
    -StdoutPath (Join-Path $seed0BaselineRun 'train_console.log') `
    -Arguments @(
        '-u', 'train.py', '-c', $baselineConfig,
        '-t', $cocoCheckpoint, '--seed', '0', '--use-amp',
        '--output-dir', $seed0BaselineRun
    )

Invoke-PythonStep -Name 'summarize_seed0' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_jointinit_multiseed_seed0_summary.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_jointinit.py',
        '--control-log', (Join-Path $seed0BaselineRun 'log.txt'),
        '--progressive-log', (Join-Path $seed0JointRun 'log.txt'),
        '--best-alignment', $seed0BestAlignment,
        '--last-alignment', $seed0LastAlignment,
        '--output', (Join-Path $reportRoot 'seed0_summary.json')
    )

foreach ($seed in 1, 2) {
    $baselineRun = Join-Path $baselineRoot "seed$seed"
    $jointRun = Join-Path $jointRoot "seed$seed"
    $seedReport = Join-Path $reportRoot "seed$seed"
    New-Item -ItemType Directory -Force -Path $seedReport | Out-Null

    Invoke-PythonStep -Name "train_paired_a00_seed$seed" `
        -StdoutPath (Join-Path $baselineRun 'train_console.log') `
        -Arguments @(
            '-u', 'train.py', '-c', $baselineConfig,
            '-t', $cocoCheckpoint, '--seed', "$seed", '--use-amp',
            '--output-dir', $baselineRun
        )

    Invoke-PythonStep -Name "train_jointinit_seed$seed" `
        -StdoutPath (Join-Path $jointRun 'train_console.log') `
        -Arguments @(
            '-u', 'train.py', '-c', $jointConfig,
            '-t', $cocoCheckpoint, '--seed', "$seed", '--use-amp',
            '--output-dir', $jointRun
        )

    $jointBest = Join-Path $jointRun 'best_stg1.pth'
    $jointLast = Join-Path $jointRun 'last.pth'
    foreach ($path in @($jointBest, $jointLast)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "JointInit checkpoint is missing: $path"
        }
    }

    $bestAlignment = Join-Path $seedReport 'mask_alignment_best.json'
    $lastAlignment = Join-Path $seedReport 'mask_alignment_last.json'
    Invoke-PythonStep -Name "validate_jointinit_seed${seed}_best_alignment" `
        -StdoutPath (Join-Path $queueRoot "s_rep1_jointinit_seed${seed}_best_alignment.log") `
        -Arguments @(
            '-u', 'tools\validate_s_rep1_mask_alignment.py',
            '--config', $jointConfig, '--checkpoint', $jointBest,
            '--batches', '20', '--output', $bestAlignment
        )
    Invoke-PythonStep -Name "validate_jointinit_seed${seed}_last_alignment" `
        -StdoutPath (Join-Path $queueRoot "s_rep1_jointinit_seed${seed}_last_alignment.log") `
        -Arguments @(
            '-u', 'tools\validate_s_rep1_mask_alignment.py',
            '--config', $jointConfig, '--checkpoint', $jointLast,
            '--batches', '20', '--output', $lastAlignment
        )

    Invoke-PythonStep -Name "summarize_seed$seed" `
        -StdoutPath (Join-Path $queueRoot "s_rep1_jointinit_multiseed_seed${seed}_summary.log") `
        -Arguments @(
            '-u', 'tools\summarize_s_rep1_jointinit.py',
            '--control-log', (Join-Path $baselineRun 'log.txt'),
            '--progressive-log', (Join-Path $jointRun 'log.txt'),
            '--best-alignment', $bestAlignment,
            '--last-alignment', $lastAlignment,
            '--output', (Join-Path $reportRoot "seed${seed}_summary.json")
        )
}

Invoke-PythonStep -Name 'aggregate_three_seed_closure' `
    -StdoutPath (Join-Path $queueRoot 's_rep1_jointinit_multiseed_aggregate.log') `
    -Arguments @(
        '-u', 'tools\summarize_s_rep1_jointinit_multiseed.py',
        '--seed-summary', (Join-Path $reportRoot 'seed0_summary.json'),
        '--seed-summary', (Join-Path $reportRoot 'seed1_summary.json'),
        '--seed-summary', (Join-Path $reportRoot 'seed2_summary.json'),
        '--output', (Join-Path $reportRoot 'multiseed_summary.json')
    )

Write-QueueLog 'QUEUE_COMPLETE'
