param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [string]$Config = "E:\two_paper\D-FINE\experiments\phase_m\m_sdtec1r2_hybrid_reader_b32_30e_local.yml",

    [string]$OutputRoot = "E:\two_paper\reports\60_multimodal\M_SDTEC1R2_CAUSAL",

    [string[]]$Modes = @(
        "normal", "zero", "batch_shuffle", "feature_permute", "global_mismatch"
    ),

    [int]$GlobalMismatchOffset = 670
)

$ErrorActionPreference = "Stop"
$pythonExe = "D:\conda\envs\hello_word\python.exe"
$repoRoot = "E:\two_paper\D-FINE"
$trainEntry = Join-Path $repoRoot "train.py"

foreach ($path in @($pythonExe, $Config, $Checkpoint, $trainEntry)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

foreach ($mode in $Modes) {
    $modeOutput = Join-Path $OutputRoot $mode
    New-Item -ItemType Directory -Path $modeOutput -Force | Out-Null
    $consoleLog = Join-Path $modeOutput "console.txt"
    Write-Host "Running M-SDTEC1 causal validation: $mode"
    # Windows PowerShell can promote harmless native stderr warnings to a
    # terminating NativeCommandError when ErrorActionPreference is Stop.
    # Judge the Python process by its exit code while still preserving stderr
    # in the audit log.
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    if ($mode -eq "global_mismatch") {
        & $pythonExe $trainEntry `
            -c $Config `
            -r $Checkpoint `
            --test-only `
            --use-amp `
            --output-dir $modeOutput `
            -u "DFINE.rgbt_thermal_intervention=normal" `
               "val_dataloader.dataset.infrared_index_offset=$GlobalMismatchOffset" 2>&1 |
            Tee-Object -FilePath $consoleLog
    }
    else {
        & $pythonExe $trainEntry `
            -c $Config `
            -r $Checkpoint `
            --test-only `
            --use-amp `
            --output-dir $modeOutput `
            -u "DFINE.rgbt_thermal_intervention=$mode" 2>&1 |
            Tee-Object -FilePath $consoleLog
    }
    $pythonExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($pythonExitCode -ne 0) {
        throw "Validation failed for mode: $mode"
    }
}
