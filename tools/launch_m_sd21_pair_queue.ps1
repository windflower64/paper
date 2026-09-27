$ErrorActionPreference = 'Stop'

# The managed Windows terminal can expose both PATH and Path in the inherited
# environment block. Start-Process treats them as duplicate dictionary keys.
# Normalize them once so the queue and its Python children can start reliably.
$taskPathValue = $env:Path
[Environment]::SetEnvironmentVariable('PATH', $null, [EnvironmentVariableTarget]::Process)
[Environment]::SetEnvironmentVariable('Path', $taskPathValue, [EnvironmentVariableTarget]::Process)

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$checkpoint = Join-Path $workspace 'weights\m_sd2_joint_coco_thermal_identity_init.pth'
$preflight = Join-Path $workspace 'reports\80_cdm_joint\M_SD21_TOKEN_PAIR\preflight.json'
$controlConfig = Join-Path $repo 'experiments\phase_m\m_sd21_pair_control_cdm_b8a4_20e_testdev_local.yml'
$tokenConfig = Join-Path $repo 'experiments\phase_m\m_sd21_token_cdm_b8a4_20e_testdev_local.yml'
$controlRun = Join-Path $workspace 'outputs\M_SD21_PAIR_CONTROL_CDM_B8A4_20E_TESTDEV\seed0'
$tokenRun = Join-Path $workspace 'outputs\M_SD21_TOKEN_CDM_B8A4_20E_TESTDEV\seed0'
$pairReport = Join-Path $workspace 'reports\80_cdm_joint\M_SD21_TOKEN_PAIR'

foreach ($file in @($python, $checkpoint, $preflight, $controlConfig, $tokenConfig)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing M-SD2.1 pair file: $file"
    }
}

$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'M-SD2.1 formal preflight did not pass; pair training is forbidden'
}
if ($preflightData.conditioner_class -ne 'SpatiallyDecoupledThermalTokenConditioner') {
    throw "Unexpected conditioner: $($preflightData.conditioner_class)"
}
if ([int]$preflightData.batch_size -ne 8 -or
    [int]$preflightData.gradient_accumulation_steps -ne 4 -or
    [int]$preflightData.effective_batch_size -ne 32) {
    throw 'M-SD2.1 batch protocol must be physical 8 x accumulation 4'
}
if ([double]$preflightData.identity_logit_max_error -ne 0.0 -or
    [double]$preflightData.identity_box_max_error -ne 0.0) {
    throw 'M-SD2.1 is not an exact identity at initialization'
}
if ([double]$preflightData.spatial_permutation_conditioned_feature_max_error -gt 0.0002) {
    throw 'M-SD2.1 coordinate-free feature permutation check failed'
}
if ([int]$preflightData.hrqs_selected_min -ne 50 -or
    [int]$preflightData.hrqs_selected_max -ne 50) {
    throw 'M-SD2.1 did not retain exactly 50 HRQS candidates'
}
if ($preflightData.missing_optimizer_parameters.Count -ne 0) {
    throw 'M-SD2.1 optimizer omitted trainable parameters'
}
$lastStep = $preflightData.steps[-1]
if ([int]$lastStep.sd2.with_nonzero_gradient -le 0) {
    throw 'M-SD2.1 token conditioner has no nonzero gradient after opening'
}

$duplicate = Get-Process -Name python, pythonw -ErrorAction SilentlyContinue
if ($duplicate) {
    throw "A Python process is already running; refusing to start a competing training job. PID: $($duplicate.Id -join ', ')"
}

function Test-CompletedRun {
    param([Parameter(Mandatory = $true)][string]$Run)
    $log = Join-Path $Run 'log.txt'
    $last = Join-Path $Run 'last.pth'
    if (-not (Test-Path -LiteralPath $log -PathType Leaf) -or
        -not (Test-Path -LiteralPath $last -PathType Leaf)) {
        return $false
    }
    try {
        $record = Get-Content -LiteralPath $log -Tail 1 -Encoding UTF8 | ConvertFrom-Json
        return [int]$record.epoch -ge 19
    }
    catch {
        return $false
    }
}

$controlComplete = Test-CompletedRun -Run $controlRun
$tokenComplete = Test-CompletedRun -Run $tokenRun
foreach ($entry in @(
    [pscustomobject]@{ Run = $controlRun; Complete = $controlComplete },
    [pscustomobject]@{ Run = $tokenRun; Complete = $tokenComplete }
)) {
    if ((Test-Path -LiteralPath (Join-Path $entry.Run 'log.txt') -PathType Leaf) -and
        -not $entry.Complete) {
        throw "Pair output has an incomplete training log; refusing overwrite: $($entry.Run)"
    }
}

$artifactDir = Join-Path $pairReport 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $controlConfig,
    $tokenConfig,
    (Join-Path $repo 'experiments\phase_m\m_sd21_pair_base_b8a4_20e_testdev_local.yml'),
    $checkpoint,
    $preflight,
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\solver\det_engine.py'),
    (Join-Path $repo 'src\solver\det_solver.py'),
    (Join-Path $repo 'tools\preflight_m_sd2_joint.py'),
    (Join-Path $repo 'tools\launch_m_sd21_pair_queue.ps1')
)
foreach ($artifact in $artifacts) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        throw "Missing pair artifact: $artifact"
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

function Invoke-PairMember {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Config,
        [Parameter(Mandatory = $true)][string]$Run
    )
    New-Item -ItemType Directory -Path $Run -Force | Out-Null
    $arguments = @(
        '-u', 'train.py',
        '-c', $Config,
        '-t', $checkpoint,
        '--seed', '0',
        '--use-amp'
    )
    $process = Start-Process -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $repo `
        -RedirectStandardOutput (Join-Path $Run 'train_console.log') `
        -RedirectStandardError (Join-Path $Run 'train_error.log') `
        -WindowStyle Hidden `
        -PassThru
    $process.Id | Set-Content -LiteralPath (Join-Path $Run 'train.pid') -Encoding ASCII
    [pscustomobject]@{
        status = 'running'
        member = $Name
        pid = $process.Id
        started_at = (Get-Date).ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pairReport 'queue_status.json') -Encoding UTF8
    $process.WaitForExit()
    $process.Refresh()
    if ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) {
        throw "$Name exited with code $($process.ExitCode)"
    }
    if (-not (Test-CompletedRun -Run $Run)) {
        throw "$Name stopped without a complete 20-epoch log"
    }
}

if (-not $controlComplete) {
    Invoke-PairMember -Name 'control' -Config $controlConfig -Run $controlRun
}
if (-not $tokenComplete) {
    Invoke-PairMember -Name 'token' -Config $tokenConfig -Run $tokenRun
}

[pscustomobject]@{
    status = 'complete'
    completed_at = (Get-Date).ToString('o')
    control_run = $controlRun
    token_run = $tokenRun
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pairReport 'queue_status.json') -Encoding UTF8
