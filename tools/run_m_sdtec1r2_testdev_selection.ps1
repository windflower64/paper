param(
    [string]$OutputRoot = "E:\two_paper\reports\60_multimodal\M_SDTEC1R2_TESTDEV_SELECTION"
)

$ErrorActionPreference = "Stop"
$pythonExe = "D:\conda\envs\hello_word\python.exe"
$repoRoot = "E:\two_paper\D-FINE"
$trainEntry = Join-Path $repoRoot "train.py"
$rgbConfig = Join-Path $repoRoot "experiments\phase_m\m_control_gq1_testdev_local.yml"
$rgbtConfig = Join-Path $repoRoot "experiments\phase_m\m_sdtec1r2_hybrid_reader_b32_30e_testdev_local.yml"
$readerRoot = "E:\two_paper\outputs\M_SDTEC1R2_HYBRID_READER_B32_30E\seed0"

$candidates = @(
    @{ Name = "rgb_gq1_baseline"; Config = $rgbConfig; Checkpoint = "E:\two_paper\runs\30_channel\C_PAT_GQ1_S32_R4\seed0\best_stg1.pth" },
    @{ Name = "reader_best_e2"; Config = $rgbtConfig; Checkpoint = (Join-Path $readerRoot "best_stg1.pth") },
    @{ Name = "reader_e4"; Config = $rgbtConfig; Checkpoint = (Join-Path $readerRoot "checkpoint0004.pth") },
    @{ Name = "reader_e9"; Config = $rgbtConfig; Checkpoint = (Join-Path $readerRoot "checkpoint0009.pth") },
    @{ Name = "reader_e14"; Config = $rgbtConfig; Checkpoint = (Join-Path $readerRoot "checkpoint0014.pth") },
    @{ Name = "reader_e19"; Config = $rgbtConfig; Checkpoint = (Join-Path $readerRoot "checkpoint0019.pth") },
    @{ Name = "reader_e24"; Config = $rgbtConfig; Checkpoint = (Join-Path $readerRoot "checkpoint0024.pth") },
    @{ Name = "reader_e29"; Config = $rgbtConfig; Checkpoint = (Join-Path $readerRoot "checkpoint0029.pth") }
)

foreach ($path in @($pythonExe, $trainEntry, $rgbConfig, $rgbtConfig)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate.Checkpoint)) {
        throw "Checkpoint does not exist: $($candidate.Checkpoint)"
    }
    $candidateOutput = Join-Path $OutputRoot $candidate.Name
    $evalPath = Join-Path $candidateOutput "eval.pth"
    if (Test-Path -LiteralPath $evalPath) {
        Write-Host "Skipping completed test-as-development evaluation: $($candidate.Name)"
        continue
    }
    New-Item -ItemType Directory -Path $candidateOutput -Force | Out-Null
    $consoleLog = Join-Path $candidateOutput "console.txt"
    Write-Host "Running test-as-development evaluation: $($candidate.Name)"
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $pythonExe $trainEntry `
        -c $candidate.Config `
        -r $candidate.Checkpoint `
        --test-only `
        --use-amp `
        --output-dir $candidateOutput 2>&1 |
        Tee-Object -FilePath $consoleLog
    $pythonExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($pythonExitCode -ne 0) {
        throw "Evaluation failed: $($candidate.Name)"
    }
}

& $pythonExe (Join-Path $repoRoot "tools\summarize_m_sdtec1r2_testdev.py") `
    --root $OutputRoot
if ($LASTEXITCODE -ne 0) {
    throw "Test-as-development summary failed"
}
