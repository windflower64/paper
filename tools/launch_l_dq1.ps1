$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_l\l_dq1_gq1_dense_o2o_mal_b32_60e_local.yml'
$checkpoint = Join-Path $workspace 'weights\dfine_n_coco.pth'
$run = Join-Path $workspace 'outputs\L_DQ1_GQ1_DENSE_O2O_MAL_B32_60E\seed0'
$preflight = Join-Path $workspace 'reports\50_supervision\L_DQ1_PREFLIGHT\preflight_b32.json'

foreach ($file in @($python, $config, $checkpoint, $preflight)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing L-DQ1 launch file: $file"
    }
}
$preflightData = Get-Content -LiteralPath $preflight -Raw -Encoding UTF8 | ConvertFrom-Json
if ($preflightData.PSObject.Properties.Name -notcontains 'status_ascii') {
    throw 'L-DQ1 preflight lacks status_ascii'
}
if ($preflightData.status_ascii -ne 'pass' -or -not $preflightData.augmentation_audit_passed) {
    throw 'L-DQ1 preflight failed; training is forbidden'
}
if ([int]$preflightData.batch_ascii -ne 32) {
    throw "L-DQ1 preflight batch is not 32: $($preflightData.batch_ascii)"
}
$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and $_.CommandLine -like '*l_dq1_gq1_dense_o2o_mal_b32_60e_local.yml*'
}
if ($duplicate) {
    throw "L-DQ1 is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'history.json')) {
    throw "L-DQ1 output already contains history; refusing overwrite: $run"
}

$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    (Join-Path $repo 'src\data\transforms\deim_mosaic.py'),
    (Join-Path $repo 'src\data\transforms\container.py'),
    (Join-Path $repo 'src\data\dataloader.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_criterion.py'),
    (Join-Path $repo 'src\nn\backbone\hgnetv2.py'),
    (Join-Path $repo 'src\nn\backbone\partialnet_pat_sf.py'),
    (Join-Path $repo 'tools\audit_l_dense_o2o_augmentations.py'),
    (Join-Path $repo 'tools\preflight_l_dq1.py'),
    (Join-Path $workspace '_third_party\DEIM\LICENSE')
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
'09d35d53d39ee3145a1e61e3a989b28b9468d1dd' | Set-Content -LiteralPath (Join-Path $artifactDir 'DEIM_COMMIT.txt') -Encoding ASCII

$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$arguments = @(
    'train.py',
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
    Experiment = 'L-DQ1-GQ1-DenseO2O-MAL-B32'
    PID = $process.Id
    Epochs = 60
    Batch = 32
    Log = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
