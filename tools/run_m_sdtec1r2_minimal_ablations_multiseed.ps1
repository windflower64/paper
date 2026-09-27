param(
    [ValidateSet("k1", "noreliability", "sameposition")]
    [string[]]$Variants = @("k1", "noreliability", "sameposition"),

    [int[]]$Seeds = @(0, 1, 2)
)

$ErrorActionPreference = "Stop"
$pythonExe = "D:\conda\envs\hello_word\python.exe"
$repoRoot = "E:\two_paper\D-FINE"
$trainEntry = Join-Path $repoRoot "train.py"
$initialCheckpoint = "E:\two_paper\weights\m_sdtec1r2_visible_gq1_thermal_e56_init.pth"
$statusRoot = "E:\two_paper\reports\60_multimodal\M_SDTEC1R2_MINIMAL_ABLATIONS"
$outputRoots = @{
    k1 = "E:\two_paper\outputs\M_SDTEC1R2_ABLATION_K1_TESTDEV"
    noreliability = "E:\two_paper\outputs\M_SDTEC1R2_ABLATION_NORELIABILITY_TESTDEV"
    sameposition = "E:\two_paper\outputs\M_SDTEC1R2_ABLATION_SAMEPOSITION_TESTDEV"
}

foreach ($path in @($pythonExe, $trainEntry, $initialCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}
foreach ($seed in $Seeds) {
    if ($seed -notin @(0, 1, 2)) {
        throw "Only preregistered seeds 0, 1 and 2 are allowed: $seed"
    }
}
New-Item -ItemType Directory -Path $statusRoot -Force | Out-Null

# Every Python process in this long sequential queue reloads source from disk.
# Lock the implementation and inherited protocol files so an accidental edit
# cannot make later seeds use a different model.
$lockFiles = @(
    (Join-Path $repoRoot "src\zoo\dfine\dfine_decoder.py"),
    (Join-Path $repoRoot "src\zoo\dfine\dfine.py"),
    (Join-Path $repoRoot "src\solver\det_engine.py"),
    (Join-Path $repoRoot "experiments\phase_m\m_sdtec1r2_hybrid_reader_b32_30e_local.yml"),
    (Join-Path $repoRoot "experiments\phase_m\m_sdtec1r2_hybrid_reader_b32_30e_testdev_local.yml")
)
foreach ($variant in $Variants) {
    $lockFiles += Join-Path $repoRoot (
        "experiments\phase_m\m_sdtec1r2_ablation_{0}_reader_b32_30e_testdev_local.yml" -f $variant
    )
    foreach ($seed in $Seeds) {
        $lockFiles += Join-Path $repoRoot (
            "experiments\phase_m\m_sdtec1r2_ablation_{0}_seed{1}_local.yml" -f $variant, $seed
        )
    }
}
$lockFiles = @($lockFiles | Select-Object -Unique)
$lockedHashes = [ordered]@{}
foreach ($path in $lockFiles) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Lock file does not exist: $path"
    }
    $lockedHashes[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}
$manifest = [ordered]@{
    protocol = "three_variants_x_three_seeds_original_test_development"
    variants = $Variants
    seeds = $Seeds
    epochs = 30
    batch_size = 32
    initial_checkpoint = $initialCheckpoint
    initial_checkpoint_sha256 = (Get-FileHash -LiteralPath $initialCheckpoint -Algorithm SHA256).Hash
    locked_files_sha256 = $lockedHashes
    started_at = (Get-Date).ToString("o")
}
$manifest | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $statusRoot "queue_manifest.json") -Encoding UTF8

foreach ($variant in $Variants) {
    $variantStatusRoot = Join-Path $statusRoot $variant
    New-Item -ItemType Directory -Path $variantStatusRoot -Force | Out-Null
    foreach ($seed in $Seeds) {
        foreach ($path in $lockFiles) {
            $currentHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
            if ($currentHash -ne $lockedHashes[$path]) {
                throw "Protocol lock changed during queue: $path"
            }
        }

        $config = Join-Path $repoRoot (
            "experiments\phase_m\m_sdtec1r2_ablation_{0}_seed{1}_local.yml" -f $variant, $seed
        )
        $runDir = Join-Path $outputRoots[$variant] ("seed{0}" -f $seed)
        if (Test-Path -LiteralPath $runDir) {
            throw "Refusing to overwrite an existing run directory: $runDir"
        }
        New-Item -ItemType Directory -Path $runDir | Out-Null
        $consoleLog = Join-Path $runDir "console.log"
        Write-Host "Starting M1 ablation ${variant}, seed $seed"
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $pythonExe $trainEntry `
            -c $config `
            -t $initialCheckpoint `
            --use-amp `
            --seed $seed 2>&1 | Tee-Object -FilePath $consoleLog
        $pythonExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedErrorActionPreference
        if ($pythonExitCode -ne 0) {
            throw "Training failed for ${variant}, seed $seed, exit code $pythonExitCode"
        }

        $trainingLog = Join-Path $runDir "log.txt"
        $bestCheckpoint = Join-Path $runDir "best_stg1.pth"
        foreach ($path in @($trainingLog, $bestCheckpoint)) {
            if (-not (Test-Path -LiteralPath $path)) {
                throw "Training artifact missing: $path"
            }
        }
        $records = @(
            Get-Content -LiteralPath $trainingLog -Encoding UTF8 |
                Where-Object { $_.Trim() } |
                ForEach-Object { $_ | ConvertFrom-Json }
        )
        if ($records.Count -ne 30 -or [int]$records[-1].epoch -ne 29) {
            throw "${variant}, seed $seed did not produce a complete 30-epoch log"
        }
        $bestRecord = $records |
            Sort-Object { [double]$_.test_coco_eval_bbox[0] } -Descending |
            Select-Object -First 1
        $seedStatus = [ordered]@{
            variant = $variant
            seed = $seed
            epochs = $records.Count
            best_epoch = [int]$bestRecord.epoch
            best_AP = [double]$bestRecord.test_coco_eval_bbox[0]
            best_AP75 = [double]$bestRecord.test_coco_eval_bbox[2]
            best_APS = [double]$bestRecord.test_coco_eval_bbox[3]
            output_dir = $runDir
            status = "COMPLETE"
        }
        $seedStatus | ConvertTo-Json -Depth 4 |
            Set-Content -LiteralPath (Join-Path $variantStatusRoot ("seed{0}_status.json" -f $seed)) -Encoding UTF8
        Write-Host "Completed ${variant}, seed ${seed}: epoch=$($seedStatus.best_epoch) AP=$($seedStatus.best_AP)"
    }
}

$queueStatus = [ordered]@{
    variants = $Variants
    seeds = $Seeds
    completed_at = (Get-Date).ToString("o")
    status = "QUEUE_COMPLETE"
}
$queueStatus | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $statusRoot "queue_status.json") -Encoding UTF8
Write-Host "MINIMAL_ABLATION_QUEUE_COMPLETE"

