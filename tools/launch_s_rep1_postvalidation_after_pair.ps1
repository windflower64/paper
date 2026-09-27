$ErrorActionPreference = 'Stop'

$repoRoot = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$pairQueuePid = 19236
$runRoot = 'E:\two_paper\runs\21_public_reproduction'
$reportRoot = 'E:\two_paper\reports\21_public_reproduction\S_REP1_SPAR'
$queueLog = Join-Path $runRoot '_queue\s_rep1_pair_queue.log'
$validationLog = Join-Path $runRoot '_queue\s_rep1_postvalidation_queue.log'

function Write-ValidationLog {
    param([string]$Message)
    Add-Content -LiteralPath $validationLog `
        -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message" `
        -Encoding UTF8
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    $stdout = Join-Path $runRoot "_queue\$Name.log"
    $stderr = "$stdout.stderr.log"
    Write-ValidationLog "START $Name"
    $process = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        Write-ValidationLog "FAILED $Name exit_code=$($process.ExitCode)"
        throw "$Name failed with exit code $($process.ExitCode)"
    }
    Write-ValidationLog "DONE $Name"
}

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
Write-ValidationLog "WAIT_PAIR_QUEUE pid=$pairQueuePid"
while ($null -ne (Get-Process -Id $pairQueuePid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 30
}

$queueState = Get-Content -LiteralPath $queueLog -Raw
if ($queueState -notmatch 'QUEUE_COMPLETE') {
    Write-ValidationLog 'PAIR_QUEUE_INCOMPLETE'
    throw 'Pair queue exited without QUEUE_COMPLETE; post-validation was not run.'
}

$sparDir = Join-Path $runRoot 'S_REP1_SPAR_A00FT10\seed0'
$bestCheckpoint = Join-Path $sparDir 'best_stg1.pth'
$lastCheckpoint = Join-Path $sparDir 'last.pth'
foreach ($checkpoint in @($bestCheckpoint, $lastCheckpoint)) {
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Missing SPAR checkpoint: $checkpoint"
    }
}

Invoke-PythonStep -Name 'spar_best_mask_alignment' -Arguments @(
    '-u', 'tools\validate_s_rep1_mask_alignment.py',
    '--config', 'experiments\phase_s\s_rep1_spar_a00ft10_local.yml',
    '--checkpoint', $bestCheckpoint,
    '--batches', '20',
    '--output', (Join-Path $reportRoot 'mask_alignment_best.json')
)

Invoke-PythonStep -Name 'spar_last_mask_alignment' -Arguments @(
    '-u', 'tools\validate_s_rep1_mask_alignment.py',
    '--config', 'experiments\phase_s\s_rep1_spar_a00ft10_local.yml',
    '--checkpoint', $lastCheckpoint,
    '--batches', '20',
    '--output', (Join-Path $reportRoot 'mask_alignment_last.json')
)

Invoke-PythonStep -Name 'summarize_s_rep1_pair' -Arguments @(
    '-u', 'tools\summarize_s_rep1_pair.py',
    '--control-log', (Join-Path $runRoot 'S_REP1_A00FT10_CONTROL\seed0\log.txt'),
    '--spar-log', (Join-Path $runRoot 'S_REP1_SPAR_A00FT10\seed0\log.txt'),
    '--output', (Join-Path $reportRoot 'paired_result.json')
)

Write-ValidationLog 'POSTVALIDATION_COMPLETE'
