$ErrorActionPreference = "Stop"
$pythonExe = "D:\conda\envs\hello_word\python.exe"
$repoRoot = "E:\two_paper\D-FINE"
$outputsRoot = "E:\two_paper\outputs"
$reportRoot = "E:\two_paper\reports\60_multimodal\M_SDTEC1R2_MINIMAL_ABLATIONS"
$causalRoot = Join-Path $reportRoot "causal"
$queueStatus = Join-Path $reportRoot "queue_status.json"
$causalRunner = Join-Path $repoRoot "tools\run_m_sdtec1_causal_validations.ps1"
$causalSummarizer = Join-Path $repoRoot "tools\summarize_m_sdtec1_causal.py"
$ablationSummarizer = Join-Path $repoRoot "tools\summarize_m_sdtec1r2_minimal_ablations.py"
$mainSummary = "E:\two_paper\reports\60_multimodal\M_SDTEC1R2_READER_TESTDEV_MULTISEED\multiseed_summary.json"
$output = Join-Path $reportRoot "minimal_ablations_summary.json"
$outputNames = @{
    k1 = "M_SDTEC1R2_ABLATION_K1_TESTDEV"
    noreliability = "M_SDTEC1R2_ABLATION_NORELIABILITY_TESTDEV"
    sameposition = "M_SDTEC1R2_ABLATION_SAMEPOSITION_TESTDEV"
}

foreach ($path in @(
    $pythonExe, $queueStatus, $causalRunner, $causalSummarizer,
    $ablationSummarizer, $mainSummary
)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}
$queue = Get-Content -LiteralPath $queueStatus -Encoding UTF8 | ConvertFrom-Json
if ($queue.status -ne "QUEUE_COMPLETE") {
    throw "Minimal ablation training queue is not complete"
}
New-Item -ItemType Directory -Path $causalRoot -Force | Out-Null

foreach ($variant in @("k1", "noreliability", "sameposition")) {
    foreach ($seed in @(0, 1, 2)) {
        $config = Join-Path $repoRoot (
            "experiments\phase_m\m_sdtec1r2_ablation_{0}_seed{1}_local.yml" -f $variant, $seed
        )
        $checkpoint = Join-Path $outputsRoot (
            "{0}\seed{1}\best_stg1.pth" -f $outputNames[$variant], $seed
        )
        $seedRoot = Join-Path $causalRoot ("{0}\seed{1}" -f $variant, $seed)
        $summary = Join-Path $seedRoot "summary.json"
        foreach ($path in @($config, $checkpoint)) {
            if (-not (Test-Path -LiteralPath $path)) {
                throw "Required artifact does not exist: $path"
            }
        }
        if (Test-Path -LiteralPath $seedRoot) {
            throw "Refusing to overwrite causal results: $seedRoot"
        }
        & $causalRunner `
            -Checkpoint $checkpoint `
            -Config $config `
            -OutputRoot $seedRoot `
            -Modes @("normal", "zero", "feature_permute", "global_mismatch")
        if ($LASTEXITCODE -ne 0) {
            throw "Causal validation failed for ${variant}, seed $seed"
        }
        & $pythonExe $causalSummarizer --root $seedRoot --output $summary
        if ($LASTEXITCODE -ne 0) {
            throw "Causal summary failed for ${variant}, seed $seed"
        }
    }
}

& $pythonExe $ablationSummarizer `
    --outputs-root $outputsRoot `
    --causal-root $causalRoot `
    --main-summary $mainSummary `
    --output $output
if ($LASTEXITCODE -ne 0) {
    throw "Minimal ablation summary failed"
}
Write-Host "MINIMAL_ABLATION_CAUSAL_AND_SUMMARY_COMPLETE"

