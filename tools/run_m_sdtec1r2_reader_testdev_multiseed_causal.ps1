$ErrorActionPreference = "Stop"
$pythonExe = "D:\conda\envs\hello_word\python.exe"
$repoRoot = "E:\two_paper\D-FINE"
$trainingRoot = "E:\two_paper\outputs\M_SDTEC1R2_READER_TESTDEV_MULTISEED"
$reportRoot = "E:\two_paper\reports\60_multimodal\M_SDTEC1R2_READER_TESTDEV_MULTISEED"
$causalRoot = Join-Path $reportRoot "causal"
$queueStatus = Join-Path $reportRoot "queue_status.json"
$causalRunner = Join-Path $repoRoot "tools\run_m_sdtec1_causal_validations.ps1"
$causalSummarizer = Join-Path $repoRoot "tools\summarize_m_sdtec1_causal.py"
$multiSeedSummarizer = Join-Path $repoRoot "tools\summarize_m_sdtec1r2_reader_testdev_multiseed.py"
$multiSeedOutput = Join-Path $reportRoot "multiseed_summary.json"

foreach ($path in @(
    $pythonExe, $queueStatus, $causalRunner, $causalSummarizer, $multiSeedSummarizer
)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}
$queue = Get-Content -LiteralPath $queueStatus -Encoding UTF8 | ConvertFrom-Json
if ($queue.status -ne "QUEUE_COMPLETE") {
    throw "Training queue is not complete"
}
New-Item -ItemType Directory -Path $causalRoot -Force | Out-Null

foreach ($seed in @(0, 1, 2)) {
    $config = Join-Path $repoRoot (
        "experiments\phase_m\m_sdtec1r2_reader_testdev_multiseed_seed{0}_local.yml" -f $seed
    )
    $checkpoint = Join-Path $trainingRoot ("seed{0}\best_stg1.pth" -f $seed)
    $seedCausalRoot = Join-Path $causalRoot ("seed{0}" -f $seed)
    $summary = Join-Path $seedCausalRoot "summary.json"
    foreach ($path in @($config, $checkpoint)) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Required seed artifact does not exist: $path"
        }
    }
    if (Test-Path -LiteralPath $seedCausalRoot) {
        throw "Refusing to overwrite existing causal results: $seedCausalRoot"
    }

    & $causalRunner `
        -Checkpoint $checkpoint `
        -Config $config `
        -OutputRoot $seedCausalRoot `
        -Modes @("normal", "zero", "feature_permute")
    if ($LASTEXITCODE -ne 0) {
        throw "Causal validation failed for seed $seed"
    }
    & $pythonExe $causalSummarizer --root $seedCausalRoot --output $summary
    if ($LASTEXITCODE -ne 0) {
        throw "Causal summary failed for seed $seed"
    }
}

& $pythonExe $multiSeedSummarizer `
    --training-root $trainingRoot `
    --causal-root $causalRoot `
    --output $multiSeedOutput
if ($LASTEXITCODE -ne 0) {
    throw "Multi-seed summary failed"
}
Write-Host "MULTISEED_CAUSAL_AND_SUMMARY_COMPLETE"
