$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_d\d_hrqs1_gq1_b32_60e_testdev_local.yml'
$checkpoint = Join-Path $workspace 'weights\dfine_n_coco.pth'
$preflight = Join-Path $workspace 'outputs\D_HRQS1_PREFLIGHT\preflight.json'
$run = Join-Path $workspace 'outputs\D_HRQS1_GQ1_B32_60E_TESTDEV\seed0'

foreach ($file in @($python, $config, $checkpoint, $preflight)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing D-HRQS1 launch file: $file"
    }
}
$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.status -ne 'PASS') {
    throw 'D-HRQS1 preflight did not pass; training is forbidden'
}
if ([int]$preflightData.batch_size -ne 32) {
    throw "D-HRQS1 preflight batch is not 32: $($preflightData.batch_size)"
}
if ([double]$preflightData.selected_s8_queries_mean -ne 50.0) {
    throw "D-HRQS1 S8 query quota is invalid: $($preflightData.selected_s8_queries_mean)"
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and $_.CommandLine -like '*d_hrqs1_gq1_b32_60e_testdev_local.yml*'
}
if ($duplicate) {
    throw "D-HRQS1 is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "D-HRQS1 output already has a training log; refusing overwrite: $run"
}

$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_decoder.py'),
    (Join-Path $repo 'tools\diagnose_qrs_selection_gap.py'),
    (Join-Path $repo 'tools\preflight_d_hrqs1.py'),
    $preflight,
    (Join-Path $workspace 'diagnostics\QRS_DIAG0_GQ1_TESTDEV_FULL\selection_gap.json')
)
foreach ($artifact in $artifacts) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        throw "Missing D-HRQS1 artifact: $artifact"
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
    Experiment = 'D-HRQS1-GQ1-B32-60E-TESTDEV'
    PID = $process.Id
    Epochs = 60
    Batch = 32
    Log = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
