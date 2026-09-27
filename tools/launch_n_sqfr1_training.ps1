$ErrorActionPreference = 'Stop'

$taskPathValue = $env:Path
[Environment]::SetEnvironmentVariable('PATH', $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable('Path', $taskPathValue, [EnvironmentVariableTarget]::Process)

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_n\n_sqfr1_c_gq1_b8a4_20e_testdev_local.yml'
$checkpoint = Join-Path $workspace 'weights\m_sd2_joint_coco_thermal_identity_init.pth'
$reportDir = Join-Path $workspace 'reports\95_n_sqfr1\N_SQFR1'
$preflight = Join-Path $reportDir 'preflight.json'
$run = Join-Path $workspace 'outputs\N_SQFR1_C_GQ1_B8A4_20E_TESTDEV\seed0'
$statusFile = Join-Path $reportDir 'training_status.json'

foreach ($file in @($python, $config, $checkpoint, $preflight)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing N-SQFR1 launch file: $file"
    }
}

$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'N-SQFR1 formal preflight did not pass; training is forbidden'
}
if ([int]$preflightData.physical_batch -ne 8 -or
    [int]$preflightData.gradient_accumulation_steps -ne 4 -or
    [int]$preflightData.effective_batch_size -ne 32) {
    throw 'N-SQFR1 protocol must use physical batch 8, accumulation 4, effective batch 32'
}
if ([double]$preflightData.initial_full_disabled_box_error -ne 0.0 -or
    [double]$preflightData.initial_zero_disabled_box_error -ne 0.0 -or
    [double]$preflightData.zero_disabled_box_error -ne 0.0) {
    throw 'N-SQFR1 identity or zero-S8 fallback check failed'
}
if ([int]$preflightData.inference_refined_queries -ne 64) {
    throw 'N-SQFR1 did not retain exactly 64 inference queries'
}

$duplicate = Get-Process -Name python, pythonw -ErrorAction SilentlyContinue
if ($duplicate) {
    throw "A Python process is already running; refusing competing training. PID: $($duplicate.Id -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "N-SQFR1 output already contains a training log; refusing overwrite: $run"
}

New-Item -ItemType Directory -Path $run -Force | Out-Null
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    (Join-Path $repo 'experiments\phase_m\c_only_gq1_b8a4_20e_testdev_local.yml'),
    $checkpoint,
    $preflight,
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_decoder.py'),
    (Join-Path $repo 'tests\test_n_sqfr1_refiner.py'),
    (Join-Path $repo 'tests\test_n_sqfr1_s8_routing.py'),
    (Join-Path $repo 'tests\test_n_sqfr1_protocol_config.py'),
    (Join-Path $repo 'tools\preflight_n_sqfr1.py'),
    (Join-Path $repo 'tools\launch_n_sqfr1_training.ps1')
)
foreach ($artifact in $artifacts) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        throw "Missing N-SQFR1 artifact: $artifact"
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
    experiment = 'N-SQFR1-C-GQ1-B8A4-20E-TESTDEV'
    pid = $process.Id
    started_at = (Get-Date).ToString('o')
    physical_batch = 8
    gradient_accumulation = 4
    effective_batch = 32
    epochs = 20
    seed = 0
    run = $run
} | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8

Write-Output "N-SQFR1 training started. PID=$($process.Id)"
Write-Output "Log: $(Join-Path $run 'train_console.log')"
