$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_m\m_sd2_joint_cdm_b32_30e_testdev_local.yml'
$checkpoint = Join-Path $workspace 'weights\m_sd2_joint_coco_thermal_identity_init.pth'
$preflight = Join-Path $workspace 'reports\80_cdm_joint\M_SD2_JOINT_CDM\preflight.json'
$run = Join-Path $workspace 'outputs\M_SD2_JOINT_CDM_B32_30E_TESTDEV\seed0'

foreach ($file in @($python, $config, $checkpoint, $preflight)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing M-SD2 launch file: $file"
    }
}

$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'M-SD2 preflight did not pass; training is forbidden'
}
if ([int]$preflightData.batch_size -ne 8 -or
    [int]$preflightData.gradient_accumulation_steps -ne 4 -or
    [int]$preflightData.effective_batch_size -ne 32) {
    throw 'M-SD2 preflight batch protocol is not physical 8 x accumulation 4 = effective 32'
}
if ([double]$preflightData.identity_logit_max_error -ne 0.0 -or
    [double]$preflightData.identity_box_max_error -ne 0.0) {
    throw 'M-SD2 zero-init identity check failed'
}
if ([double]$preflightData.spatial_permutation_logit_max_error -ne 0.0 -or
    [double]$preflightData.spatial_permutation_box_max_error -ne 0.0) {
    throw 'M-SD2 coordinate-free permutation check failed'
}
if ([int]$preflightData.hrqs_selected_min -ne 50 -or
    [int]$preflightData.hrqs_selected_max -ne 50) {
    throw 'M-SD2 HRQS query quota is invalid'
}
$lastStep = $preflightData.steps[-1]
if ([int]$lastStep.sd2.with_nonzero_gradient -ne [int]$lastStep.sd2.tensor_count) {
    throw 'M-SD2 did not establish full gradient connectivity after opening'
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and $_.CommandLine -like '*m_sd2_joint_cdm_b32_30e_testdev_local.yml*'
}
if ($duplicate) {
    throw "M-SD2 is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "M-SD2 output already has a training log; refusing overwrite: $run"
}

$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    $checkpoint,
    $preflight,
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\solver\det_engine.py'),
    (Join-Path $repo 'src\solver\det_solver.py'),
    (Join-Path $repo 'tools\prepare_m_sd2_joint_init.py'),
    (Join-Path $repo 'tools\preflight_m_sd2_joint.py')
)
foreach ($artifact in $artifacts) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        throw "Missing M-SD2 artifact: $artifact"
    }
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
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:CUDA_MODULE_LOADING = 'LAZY'
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
    Experiment = 'M-SD2-JOINT-CDM-B32-30E-TESTDEV'
    PID = $process.Id
    Epochs = 30
    PhysicalBatch = 8
    GradientAccumulation = 4
    EffectiveBatch = 32
    Log = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
