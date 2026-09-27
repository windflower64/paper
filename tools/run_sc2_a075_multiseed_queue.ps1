$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$trainer = Join-Path $repo 'tools\train_s_hrbr1_refinebox.py'
$config = Join-Path $repo 'experiments\phase_sc\sc2_gq1_local_hrbr_local.yml'
$checkpoint = Join-Path $workspace 'runs\30_channel\C_PAT_GQ1_S32_R4\seed0\best_stg1.pth'

foreach ($required in @($python, $trainer, $config, $checkpoint)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "缺少必需文件：$required"
    }
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

foreach ($seed in @(1, 2)) {
    $outputDir = Join-Path $workspace "outputs\SC2_GQ1_LOCAL_HRBR_A075_SEED$seed"
    if (Test-Path -LiteralPath (Join-Path $outputDir 'history.json')) {
        throw "seed$seed输出目录已有训练记录，拒绝覆盖：$outputDir"
    }
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $stdout = Join-Path $outputDir 'train_console.log'
    $stderr = Join-Path $outputDir 'train_error.log'
    $arguments = @(
        $trainer,
        '--config', $config,
        '--checkpoint', $checkpoint,
        '--output-dir', $outputDir,
        '--epochs', '12',
        '--batch-size', '16',
        '--feature-stages', '0,1,2',
        '--eval-residual-scale', '0.75',
        '--save-every-epoch',
        '--seed', "$seed",
        '--experiment-name', "SC2-GQ1-LocalHRBR-A075-Seed$seed",
        '--detector-name', 'GQ1'
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
        throw "seed$seed训练失败，exit code=$($process.ExitCode)，查看：$stderr"
    }
}

'SC2_A075_SEED1_SEED2_QUEUE_COMPLETE'
