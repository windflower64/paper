param(
    [int[]]$Seeds = @(0, 1, 2)
)

$ErrorActionPreference = "Stop"
$pythonExe = "D:\conda\envs\hello_word\python.exe"
$repoRoot = "E:\two_paper\D-FINE"
$trainEntry = Join-Path $repoRoot "train.py"
$initialCheckpoint = "E:\two_paper\weights\m_sdtec1r2_visible_gq1_thermal_e56_init.pth"
$outputRoot = "E:\two_paper\outputs\M_SDTEC1R2_READER_TESTDEV_MULTISEED"
$statusRoot = "E:\two_paper\reports\60_multimodal\M_SDTEC1R2_READER_TESTDEV_MULTISEED"

foreach ($path in @($pythonExe, $trainEntry, $initialCheckpoint)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}
New-Item -ItemType Directory -Path $statusRoot -Force | Out-Null

foreach ($seed in $Seeds) {
    if ($seed -notin @(0, 1, 2)) {
        throw "Only the preregistered seeds 0, 1 and 2 are allowed: $seed"
    }
    $config = Join-Path $repoRoot (
        "experiments\phase_m\m_sdtec1r2_reader_testdev_multiseed_seed{0}_local.yml" -f $seed
    )
    $runDir = Join-Path $outputRoot ("seed{0}" -f $seed)
    if (-not (Test-Path -LiteralPath $config)) {
        throw "Seed config does not exist: $config"
    }
    if (Test-Path -LiteralPath $runDir) {
        throw "Refusing to overwrite an existing seed directory: $runDir"
    }

    New-Item -ItemType Directory -Path $runDir | Out-Null
    $consoleLog = Join-Path $runDir "console.log"
    Write-Host "Starting strict M-SDTEC1-R2 reader seed $seed"
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
        throw "Training failed for seed $seed with exit code $pythonExitCode"
    }

    $trainingLog = Join-Path $runDir "log.txt"
    $bestCheckpoint = Join-Path $runDir "best_stg1.pth"
    if (-not (Test-Path -LiteralPath $trainingLog)) {
        throw "Training log missing for seed $seed"
    }
    if (-not (Test-Path -LiteralPath $bestCheckpoint)) {
        throw "Best checkpoint missing for seed $seed"
    }
    $records = @(
        Get-Content -LiteralPath $trainingLog -Encoding UTF8 |
            Where-Object { $_.Trim() } |
            ForEach-Object { $_ | ConvertFrom-Json }
    )
    if ($records.Count -ne 30 -or [int]$records[-1].epoch -ne 29) {
        throw "Seed $seed did not produce a complete 30-epoch log"
    }
    $bestRecord = $records |
        Sort-Object { [double]$_.test_coco_eval_bbox[0] } -Descending |
        Select-Object -First 1
    $seedStatus = [ordered]@{
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
        Set-Content -LiteralPath (Join-Path $statusRoot ("seed{0}_status.json" -f $seed)) -Encoding UTF8
    Write-Host "Completed seed ${seed}: best epoch=$($seedStatus.best_epoch) AP=$($seedStatus.best_AP)"
}

$queueStatus = [ordered]@{
    seeds = $Seeds
    completed_at = (Get-Date).ToString("o")
    status = "QUEUE_COMPLETE"
}
$queueStatus | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $statusRoot "queue_status.json") -Encoding UTF8
Write-Host "QUEUE_COMPLETE"
