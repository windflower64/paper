$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_cdm\cdm2_m2_logit_calibration_b32_20e_testdev_local.yml'
$checkpoint = Join-Path $workspace 'weights\cdm2_m2_dhrqs1_k8_logit_identity_init.pth'
$preflight = Join-Path $workspace 'reports\80_cdm_joint\CDM2_M2_LOGIT_CALIBRATION\preflight.json'
$initEvaluation = Join-Path $workspace 'reports\80_cdm_joint\CDM2_M2_LOGIT_CALIBRATION\init_testdev\eval.pth'
$run = Join-Path $workspace 'outputs\CDM2_M2_LOGIT_CALIBRATION_B32_20E_TESTDEV\seed0'

foreach ($file in @($python, $config, $checkpoint, $preflight, $initEvaluation)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing M2 launch file: $file"
    }
}
$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'M2 preflight did not pass; training is forbidden'
}
if ([int]$preflightData.batch_size -ne 32) {
    throw "M2 preflight batch is not 32: $($preflightData.batch_size)"
}
if (-not [bool]$preflightData.initial_normal_equals_zero_logits) {
    throw 'M2 initialization did not preserve exact C+D logits'
}
if (-not [bool]$preflightData.opened_normal_equals_zero_boxes) {
    throw 'M2 changed box geometry during preflight'
}
if ([int]$preflightData.protected_hrqs_queries -ne 50) {
    throw 'M2 did not protect all HRQS queries'
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and
    $_.CommandLine -like '*cdm2_m2_logit_calibration_b32_20e_testdev_local.yml*'
}
if ($duplicate) {
    throw "M2 is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "M2 output already has a training log; refusing overwrite: $run"
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
    (Join-Path $repo 'tools\prepare_cdm2_m2_init.py'),
    (Join-Path $repo 'tools\preflight_cdm2_m2.py')
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
    Experiment = 'CDM2-M2-LogitCalibration-B32-20E-TestDev'
    PID = $process.Id
    Epochs = 20
    Batch = 32
    Log = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
