$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$trainer = Join-Path $repo 'tools\train_s_hrbr1_refinebox.py'
$config = Join-Path $repo 'experiments\phase_scm\scm1_gq1_mk1_local_hrbr_a075_testdev_local.yml'
$checkpoint = Join-Path $workspace 'outputs\M_SDTEC1R2_ABLATION_K1_TESTDEV\seed0\best_stg1.pth'
$outputDir = Join-Path $workspace 'outputs\SCM1_GQ1_MK1_LOCAL_HRBR_A075_SEED0'

foreach ($required in @($python, $trainer, $config, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing required file: $required"
    }
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*.exe' -and
    $_.CommandLine -like '*train_s_hrbr1_refinebox.py*' -and
    $_.CommandLine -like '*SCM1_GQ1_MK1_LOCAL_HRBR_A075_SEED0*'
}
if ($duplicate) {
    throw "SCM1 is already running, PID: $($duplicate.ProcessId -join ', ')"
}

if (Test-Path -LiteralPath (Join-Path $outputDir 'history.json')) {
    throw "Formal history already exists; refusing to overwrite: $outputDir"
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$stdout = Join-Path $outputDir 'train_console.log'
$stderr = Join-Path $outputDir 'train_error.log'

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$arguments = @(
    '-u', $trainer,
    '--config', $config,
    '--checkpoint', $checkpoint,
    '--output-dir', $outputDir,
    '--epochs', '12',
    '--batch-size', '16',
    '--feature-stages', '0,1,2',
    '--eval-residual-scale', '0.75',
    '--save-every-epoch',
    '--seed', '0',
    '--experiment-name', 'SCM1-GQ1-M1K1-LocalHRBR-A075-Seed0',
    '--detector-name', 'C-GQ1+M1-K1'
)

$process = Start-Process -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    Experiment = 'SCM1-GQ1-M1K1-LocalHRBR-A075-Seed0'
    PID = $process.Id
    OutputDir = $outputDir
    Stdout = $stdout
    Stderr = $stderr
}
