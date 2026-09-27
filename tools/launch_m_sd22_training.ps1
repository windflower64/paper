$ErrorActionPreference = 'Stop'

# Some managed Windows terminals expose PATH and Path simultaneously.  Normalize
# the process environment before Start-Process builds a child environment block.
$taskPathValue = $env:Path
[Environment]::SetEnvironmentVariable('PATH', $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable('Path', $taskPathValue, [EnvironmentVariableTarget]::Process)

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_m\m_sd22_thermal_contrast_cdm_b8a4_20e_testdev_local.yml'
$checkpoint = Join-Path $workspace 'weights\m_sd2_joint_coco_thermal_identity_init.pth'
$preflight = Join-Path $workspace 'reports\80_cdm_joint\M_SD22_THERMAL_CONTRAST\preflight.json'
$reportDir = Join-Path $workspace 'reports\80_cdm_joint\M_SD22_THERMAL_CONTRAST'
$run = Join-Path $workspace 'outputs\M_SD22_THERMAL_CONTRAST_CDM_B8A4_20E_TESTDEV\seed0'
$statusFile = Join-Path $reportDir 'training_status.json'

foreach ($file in @($python, $config, $checkpoint, $preflight)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing M-SD2.2 launch file: $file"
    }
}

$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'M-SD2.2 formal preflight did not pass; training is forbidden'
}
if (-not [bool]$preflightData.thermal_contrastive) {
    throw 'M-SD2.2 thermal-contrastive residual is not enabled'
}
if ([int]$preflightData.batch_size -ne 8 -or
    [int]$preflightData.gradient_accumulation_steps -ne 4 -or
    [int]$preflightData.effective_batch_size -ne 32) {
    throw 'M-SD2.2 batch protocol must be physical 8 x accumulation 4'
}
if ([double]$preflightData.identity_logit_max_error -ne 0.0 -or
    [double]$preflightData.identity_box_max_error -ne 0.0 -or
    [double]$preflightData.null_conditioned_feature_max_error -ne 0.0) {
    throw 'M-SD2.2 identity/null-response check failed'
}
if ([double]$preflightData.spatial_permutation_conditioned_feature_max_error -gt
    [double]$preflightData.spatial_permutation_feature_tolerance) {
    throw 'M-SD2.2 coordinate-free feature permutation check failed'
}
if ([double]$preflightData.opened_probe_rms_ratio_max -gt
    [double]$preflightData.configured_max_rms_ratio) {
    throw 'M-SD2.2 opened probe exceeded the configured RMS limit'
}
if ([int]$preflightData.hrqs_selected_min -ne 50 -or
    [int]$preflightData.hrqs_selected_max -ne 50) {
    throw 'M-SD2.2 did not retain exactly 50 HRQS candidates'
}
if ($preflightData.missing_optimizer_parameters.Count -ne 0) {
    throw 'M-SD2.2 optimizer omitted trainable parameters'
}
$lastStep = $preflightData.steps[-1]
if ([int]$lastStep.sd2.with_nonzero_gradient -ne [int]$lastStep.sd2.tensor_count -or
    [int]$lastStep.sd2.non_finite_gradient_tensors -ne 0) {
    throw 'M-SD2.2 did not establish complete finite gradients after opening'
}

$duplicate = Get-Process -Name python, pythonw -ErrorAction SilentlyContinue
if ($duplicate) {
    throw "A Python process is already running; refusing a competing training job. PID: $($duplicate.Id -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "M-SD2.2 output already has a training log; refusing overwrite: $run"
}

New-Item -ItemType Directory -Path $run -Force | Out-Null
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    (Join-Path $repo 'experiments\phase_m\m_sd21_token_cdm_b8a4_20e_testdev_local.yml'),
    (Join-Path $repo 'experiments\phase_m\m_sd21_pair_base_b8a4_20e_testdev_local.yml'),
    $checkpoint,
    $preflight,
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\solver\det_engine.py'),
    (Join-Path $repo 'src\solver\det_solver.py'),
    (Join-Path $repo 'tools\preflight_m_sd2_joint.py'),
    (Join-Path $repo 'tools\launch_m_sd22_training.ps1')
)
foreach ($artifact in $artifacts) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        throw "Missing M-SD2.2 artifact: $artifact"
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
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8'
$env:TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT = '100'

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
    status = 'running'
    experiment = 'M-SD2.2-THERMAL-CONTRAST-CDM-B8A4-20E-TESTDEV'
    pid = $process.Id
    started_at = (Get-Date).ToString('o')
    physical_batch = 8
    gradient_accumulation = 4
    effective_batch = 32
    epochs = 20
    seed = 0
    run = $run
} | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8

$process.WaitForExit()
$process.Refresh()
if ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) {
    throw "M-SD2.2 exited with code $($process.ExitCode)"
}

$log = Join-Path $run 'log.txt'
$last = Join-Path $run 'last.pth'
if (-not (Test-Path -LiteralPath $log -PathType Leaf) -or
    -not (Test-Path -LiteralPath $last -PathType Leaf)) {
    throw 'M-SD2.2 stopped without final training artifacts'
}
$lastRecord = Get-Content -LiteralPath $log -Tail 1 -Encoding UTF8 | ConvertFrom-Json
if ([int]$lastRecord.epoch -lt 19) {
    throw "M-SD2.2 stopped before epoch 19; last epoch: $($lastRecord.epoch)"
}

[pscustomobject]@{
    status = 'complete'
    experiment = 'M-SD2.2-THERMAL-CONTRAST-CDM-B8A4-20E-TESTDEV'
    pid = $process.Id
    started_at = (Get-Item -LiteralPath (Join-Path $run 'train.pid')).CreationTime.ToString('o')
    completed_at = (Get-Date).ToString('o')
    last_epoch = [int]$lastRecord.epoch
    run = $run
} | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
