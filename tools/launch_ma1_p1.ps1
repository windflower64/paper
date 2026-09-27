$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_m\ma1_p1_soft_aligned_reader_b16_20e_testdev_stable_local.yml'
$checkpoint = Join-Path $workspace 'weights\ma1_p1_dhrqs1_thermal_soft_aligned_identity_init.pth'
$preflight = Join-Path $workspace 'reports\80_cdm_joint\MA1_SOFT_ALIGNED\preflight_p0_b16_stable.json'
$geometry = Join-Path $workspace 'reports\80_cdm_joint\FUSION_BOUNDARY_AUDIT\pair_geometry.json'
$initEvaluation = Join-Path $workspace 'reports\80_cdm_joint\MA1_SOFT_ALIGNED\init_equivalence\eval.pth'
$initSummary = Join-Path $workspace 'reports\80_cdm_joint\MA1_SOFT_ALIGNED\init_equivalence\summary.json'
$run = Join-Path $workspace 'outputs\MA1_P1_SOFT_ALIGNED_B16_20E_TESTDEV_R1\seed0'

foreach ($file in @(
    $python, $config, $checkpoint, $preflight, $geometry,
    $initEvaluation, $initSummary
)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing M-A1 launch file: $file"
    }
}

$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'M-A1/P0 did not pass; formal training is forbidden'
}
if ([int]$preflightData.batch_size -ne 16) {
    throw "M-A1 preflight batch is not 16: $($preflightData.batch_size)"
}
if ([int]$preflightData.trainable_tensor_count -ne 13) {
    throw "M-A1 trainable tensor count changed: $($preflightData.trainable_tensor_count)"
}
if (-not [bool]$preflightData.initial_normal_equals_zero_logits) {
    throw 'M-A1 initialization did not preserve exact C+D logits'
}
if (-not [bool]$preflightData.opened_normal_equals_zero_boxes) {
    throw 'M-A1 changed box geometry during preflight'
}
if (-not [bool]$preflightData.opened_zero_content_delta_exact_zero) {
    throw 'M-A1 zero-content delta is not exact zero'
}
if ([int]$preflightData.steps[-1].trainable_with_nonzero_gradient -ne 13) {
    throw 'Not every M-A1 trainable tensor received detection gradients'
}

$initData = Get-Content -LiteralPath $initSummary -Raw -Encoding UTF8 | ConvertFrom-Json
if ($initData.status -ne 'PASS') {
    throw 'M-A1 initialization test equivalence did not pass'
}
if ([double]$initData.absolute_error -ne 0.0) {
    throw "M-A1 initialization AP differs from strict C+D: $($initData.absolute_error)"
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and
    $_.CommandLine -like '*ma1_p1_soft_aligned_reader_b16_20e_testdev_stable_local.yml*'
}
if ($duplicate) {
    throw "M-A1 is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "M-A1 output already has a training log; refusing overwrite: $run"
}

$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    $checkpoint,
    $preflight,
    $geometry,
    $initEvaluation,
    $initSummary,
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_decoder.py'),
    (Join-Path $repo 'src\solver\det_engine.py'),
    (Join-Path $repo 'tools\prepare_ma1_p1_init.py'),
    (Join-Path $repo 'tools\preflight_ma1_p0.py'),
    (Join-Path $repo 'tools\audit_rgbt_pair_geometry.py')
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
    Experiment = 'MA1-P1-SoftAligned-B16-20E-TestDev-R1'
    PID = $process.Id
    Epochs = 20
    TrainBatch = 16
    ValidationBatch = 32
    TrainableParameters = [int]$preflightData.trainable_parameter_count
    StrictStartAP = [double]$initData.test_ap
    Log = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
