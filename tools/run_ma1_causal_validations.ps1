param(
    [string]$Checkpoint = 'E:\two_paper\outputs\MA1_P1_SOFT_ALIGNED_B16_20E_TESTDEV_R1\seed0\checkpoint0017.pth',
    [string]$Config = 'E:\two_paper\D-FINE\experiments\phase_m\ma1_p1_soft_aligned_reader_b16_20e_testdev_stable_local.yml',
    [string]$OutputRoot = 'E:\two_paper\reports\80_cdm_joint\MA1_SOFT_ALIGNED\epoch17_causal',
    [int]$GlobalMismatchOffset = 670
)

$ErrorActionPreference = 'Stop'
$python = 'D:\conda\envs\hello_word\python.exe'
$repo = 'E:\two_paper\D-FINE'
$entry = Join-Path $repo 'train.py'
$modes = @(
    'normal',
    'zero_content_valid',
    'batch_shuffle',
    'feature_permute',
    'global_mismatch'
)

foreach ($path in @($python, $entry, $Config, $Checkpoint)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing MA1 causal validation file: $path"
    }
}

foreach ($mode in $modes) {
    $modeOutput = Join-Path $OutputRoot $mode
    $evalPath = Join-Path $modeOutput 'eval.pth'
    if (Test-Path -LiteralPath $evalPath -PathType Leaf) {
        Write-Host "Skipping completed MA1 causal mode: $mode"
        continue
    }
    New-Item -ItemType Directory -Path $modeOutput -Force | Out-Null
    $consoleLog = Join-Path $modeOutput 'console.txt'
    Write-Host "Running MA1 causal mode: $mode"
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    if ($mode -eq 'global_mismatch') {
        & $python $entry `
            -c $Config `
            -r $Checkpoint `
            --test-only `
            --use-amp `
            --output-dir $modeOutput `
            -u 'DFINE.rgbt_thermal_intervention=normal' `
               "val_dataloader.dataset.infrared_index_offset=$GlobalMismatchOffset" 2>&1 |
            Tee-Object -FilePath $consoleLog
    }
    else {
        & $python $entry `
            -c $Config `
            -r $Checkpoint `
            --test-only `
            --use-amp `
            --output-dir $modeOutput `
            -u "DFINE.rgbt_thermal_intervention=$mode" 2>&1 |
            Tee-Object -FilePath $consoleLog
    }
    $pythonExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($pythonExitCode -ne 0) {
        throw "MA1 causal validation failed: $mode"
    }
}
