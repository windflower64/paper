$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_m\mb2_explicit_local_alignment_b16_30e_testdev_local.yml'
$checkpoint = Join-Path $workspace 'weights\ma1_p1_dhrqs1_thermal_soft_aligned_identity_init.pth'
$preflight = Join-Path $workspace 'reports\80_cdm_joint\MB1_SPATIAL_ALIGNMENT\mb2_preflight.json'
$p1Report = Join-Path $workspace 'reports\80_cdm_joint\MB1_SPATIAL_ALIGNMENT\p1_target_alignment.json'
$p2Report = Join-Path $workspace 'reports\80_cdm_joint\MB1_SPATIAL_ALIGNMENT\p2_bounded_fusion.json'
$run = Join-Path $workspace 'outputs\MB2_EXPLICIT_LOCAL_ALIGNMENT_B16_30E_TESTDEV\seed0'

foreach ($file in @($python, $config, $checkpoint, $preflight, $p1Report, $p2Report)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing M-B2 launch file: $file"
    }
}

$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'M-B2 preflight did not pass; formal training is forbidden'
}
if ([int]$preflightData.batch_size -ne 16) {
    throw "M-B2 preflight batch is not 16: $($preflightData.batch_size)"
}
if (-not [bool]$preflightData.initial_normal_equals_zero_logits) {
    throw 'M-B2 initialization did not preserve exact C+D logits'
}
if (-not [bool]$preflightData.opened_normal_equals_zero_boxes) {
    throw 'M-B2 altered RGB box geometry during preflight'
}
if (-not [bool]$preflightData.opened_zero_content_delta_exact_zero) {
    throw 'M-B2 zero-content fallback is not exact'
}
if ([int]$preflightData.steps[0].trainable_with_nonzero_gradient -lt 12) {
    throw 'Explicit alignment supervision did not open alignment gradients at step zero'
}
if ([int]$preflightData.steps[-1].trainable_with_nonzero_gradient -ne 13) {
    throw 'Not every M-B2 trainable tensor received gradients'
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(.exe)?$' -and
    $_.CommandLine -like '*mb2_explicit_local_alignment_b16_30e_testdev_local.yml*'
}
if ($duplicate) {
    throw "M-B2 is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "M-B2 output already has a training log; refusing overwrite: $run"
}

$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    $checkpoint,
    $preflight,
    $p1Report,
    $p2Report,
    (Join-Path $repo 'src\data\dataset\coco_dataset.py'),
    (Join-Path $repo 'src\data\transforms\_transforms.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_decoder.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_criterion.py'),
    (Join-Path $repo 'tools\preflight_ma1_p0.py')
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
    Experiment = 'M-B2-Explicit-Local-Alignment-B16-30E-TestDev'
    PID = $process.Id
    Epochs = 30
    TrainBatch = 16
    ValidationBatch = 32
    TrainableParameters = [int]$preflightData.trainable_parameter_count
    Log = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
