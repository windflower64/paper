$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$config = Join-Path $repo 'experiments\phase_sc\sc2_gq1_local_hrbr_local.yml'
$checkpoint = Join-Path $workspace 'runs\30_channel\C_PAT_GQ1_S32_R4\seed0\best_stg1.pth'
$outputDir = Join-Path $workspace 'outputs\SC2_GQ1_LOCAL_HRBR_SEED0'

foreach ($required in @($python, $config, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "缺少必需文件：$required"
    }
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*train_s_hrbr1_refinebox.py*' -and
    $_.CommandLine -like '*SC2_GQ1_LOCAL_HRBR_SEED0*'
}
if ($duplicate) {
    throw "SC2已经在运行，PID：$($duplicate.ProcessId -join ', ')"
}

if (Test-Path -LiteralPath (Join-Path $outputDir 'history.json')) {
    throw "输出目录已有正式训练记录，拒绝覆盖：$outputDir"
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$stdout = Join-Path $outputDir 'train_console.log'
$stderr = Join-Path $outputDir 'train_error.log'

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$arguments = @(
    'tools/train_s_hrbr1_refinebox.py',
    '--config', $config,
    '--checkpoint', $checkpoint,
    '--output-dir', $outputDir,
    '--epochs', '12',
    '--batch-size', '16',
    '--feature-stages', '0,1,2',
    '--seed', '0',
    '--experiment-name', 'SC2-GQ1-LocalHRBR-S4S8S16-Seed0',
    '--detector-name', 'GQ1'
)

$process = Start-Process -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    Experiment = 'SC2-GQ1-LocalHRBR-S4S8S16-Seed0'
    PID = $process.Id
    OutputDir = $outputDir
    Stdout = $stdout
    Stderr = $stderr
}
