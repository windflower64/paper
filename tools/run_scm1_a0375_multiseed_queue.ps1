$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$trainer = Join-Path $repo 'tools\train_s_hrbr1_refinebox.py'
$config = Join-Path $repo 'experiments\phase_scm\scm1_gq1_mk1_local_hrbr_testdev_local.yml'
$checkpoint = Join-Path $workspace 'outputs\M_SDTEC1R2_ABLATION_K1_TESTDEV\seed0\best_stg1.pth'

foreach ($required in @($python, $trainer, $config, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing required file: $required"
    }
}

foreach ($seed in @(1, 2)) {
    $outputDir = Join-Path $workspace "outputs\SCM1_GQ1_MK1_LOCAL_HRBR_A0375_SEED$seed"
    if (Test-Path -LiteralPath (Join-Path $outputDir 'history.json')) {
        throw "Formal history already exists; refusing to overwrite: $outputDir"
    }
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $stdout = Join-Path $outputDir 'train_console.log'
    $stderr = Join-Path $outputDir 'train_error.log'
    $arguments = @(
        '-u', $trainer,
        '--config', $config,
        '--checkpoint', $checkpoint,
        '--output-dir', $outputDir,
        '--epochs', '12',
        '--batch-size', '16',
        '--feature-stages', '0,1,2',
        '--eval-residual-scale', '0.375',
        '--save-every-epoch',
        '--seed', "$seed",
        '--experiment-name', "SCM1-GQ1-M1K1-LocalHRBR-A0375-Seed$seed",
        '--detector-name', 'C-GQ1+M1-K1'
    )
    $process = Start-Process -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "SCM1 seed$seed failed with exit code $($process.ExitCode): $stderr"
    }
    "SCM1_A0375_SEED${seed}_COMPLETE"
}

'SCM1_A0375_MULTISEED_QUEUE_COMPLETE'

