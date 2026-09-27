$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_d\d_control_gq1_b32_60e_testdev_local.yml'
$checkpoint = Join-Path $workspace 'weights\dfine_n_coco.pth'
$run = Join-Path $workspace 'outputs\D_CONTROL_GQ1_B32_60E_TESTDEV\seed0'

foreach ($file in @($python, $config, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing D control launch file: $file"
    }
}
$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and $_.CommandLine -like '*d_control_gq1_b32_60e_testdev_local.yml*'
}
if ($duplicate) {
    throw "D control is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "D control output already has a training log; refusing overwrite: $run"
}

$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
foreach ($artifact in @($config, (Join-Path $repo 'experiments\phase_d\d_hrqs1_gq1_b32_60e_testdev_local.yml'))) {
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
$arguments = @('-u', 'train.py', '-c', $config, '-t', $checkpoint, '--seed', '0', '--use-amp')
$process = Start-Process -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -RedirectStandardOutput (Join-Path $run 'train_console.log') `
    -RedirectStandardError (Join-Path $run 'train_error.log') `
    -WindowStyle Hidden `
    -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $run 'train.pid') -Encoding ASCII

[pscustomobject]@{
    Experiment = 'D-CONTROL-GQ1-B32-60E-TESTDEV'
    PID = $process.Id
    Epochs = 60
    Batch = 32
    Log = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
