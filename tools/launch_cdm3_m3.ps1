$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_cdm\cdm3_m3_candidate_calibration_b32_20e_testdev_local.yml'
$checkpoint = Join-Path $workspace 'weights\cdm3_m3_dhrqs1_thermal_candidate_identity_init.pth'
$preflight = Join-Path $workspace 'reports\80_cdm_joint\CDM3_M3_CANDIDATE_CALIBRATION\preflight.json'
$initEvaluation = Join-Path $workspace 'reports\80_cdm_joint\CDM3_M3_CANDIDATE_CALIBRATION\init_equivalence\eval.pth'
$run = Join-Path $workspace 'outputs\CDM3_M3_CANDIDATE_CALIBRATION_B32_20E_TESTDEV\seed0'

foreach ($file in @($python, $config, $checkpoint, $preflight, $initEvaluation)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing M3 launch file: $file"
    }
}
$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'M3 preflight did not pass; training is forbidden'
}
if ([int]$preflightData.batch_size -ne 32) {
    throw "M3 preflight batch is not 32: $($preflightData.batch_size)"
}
if ([int]$preflightData.trainable_tensor_count -ne 26) {
    throw "M3 trainable tensor count changed: $($preflightData.trainable_tensor_count)"
}
if ([int]$preflightData.frozen_candidate_head_tensors -ne 6) {
    throw 'M3 imported thermal candidate head is not fully frozen'
}
if (-not [bool]$preflightData.initial_normal_equals_zero_logits) {
    throw 'M3 initialization did not preserve exact C+D logits'
}
if (-not [bool]$preflightData.opened_normal_equals_zero_boxes) {
    throw 'M3 changed box geometry during preflight'
}
if ([double]$preflightData.feature_permutation_max_candidate_token_error -ne 0.0) {
    throw 'M3 candidate interface is not exactly spatial-permutation invariant'
}
if ($preflightData.steps[-1].trainable_with_nonzero_gradient -ne 26) {
    throw 'Not every trainable M3 tensor received a nonzero gradient'
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and
    $_.CommandLine -like '*cdm3_m3_candidate_calibration_b32_20e_testdev_local.yml*'
}
if ($duplicate) {
    throw "M3 is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "M3 output already has a training log; refusing overwrite: $run"
}

$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    $checkpoint,
    $preflight,
    $initEvaluation,
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_decoder.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_criterion.py'),
    (Join-Path $repo 'tools\prepare_cdm3_m3_init.py'),
    (Join-Path $repo 'tools\preflight_cdm3_m3.py'),
    (Join-Path $repo 'tools\audit_ir_candidate_supervision.py')
)
foreach ($artifact in $artifacts) {
    Copy-Item -LiteralPath $artifact -Destination $artifactDir -Force
}
$manifest = foreach ($artifact in Get-ChildItem -LiteralPath $artifactDir -File) {
    [pscustomobject]@{
        File = $artifact.Name
        Bytes = $artifact.Length
        SHA256 = (Get-FileHash -LiteralPath $artifact.FullName -Algorithm SHA256).Hash
    }
}
$manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $artifactDir 'SHA256_MANIFEST.json') -Encoding UTF8

$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$arguments = @(
    '-u', 'train.py',
    '-c', $config,
    '-t', $checkpoint,
    '--seed', '0',
    '--use-amp'
)
$process = Start-Process -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $run 'train_console.log') `
    -RedirectStandardError (Join-Path $run 'train_error.log') `
    -WindowStyle Hidden `
    -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $run 'train.pid') -Encoding ASCII

[pscustomobject]@{
    Experiment = 'CDM3-M3-CandidateCalibration-B32-20E-TestDev'
    PID = $process.Id
    Epochs = 20
    Batch = 32
    Log = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
