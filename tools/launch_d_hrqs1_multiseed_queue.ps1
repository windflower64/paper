$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$checkpoint = Join-Path $workspace 'weights\dfine_n_coco.pth'
$hrqsConfig = Join-Path $repo 'experiments\phase_d\d_hrqs1_gq1_b32_60e_testdev_local.yml'
$controlConfig = Join-Path $repo 'experiments\phase_d\d_control_gq1_b32_60e_testdev_local.yml'
$queueDir = Join-Path $workspace 'reports\70_detector\D_HRQS1_MULTISEED_QUEUE'

foreach ($file in @($python, $checkpoint, $hrqsConfig, $controlConfig)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing D-HRQS1 multiseed file: $file"
    }
}

$runs = @(
    [pscustomobject]@{
        Name = 'D-HRQS1-seed1'
        Seed = 1
        Config = $hrqsConfig
        Output = Join-Path $workspace 'outputs\D_HRQS1_GQ1_B32_60E_TESTDEV\seed1'
    },
    [pscustomobject]@{
        Name = 'D-CONTROL-GQ1-seed1'
        Seed = 1
        Config = $controlConfig
        Output = Join-Path $workspace 'outputs\D_CONTROL_GQ1_B32_60E_TESTDEV\seed1'
    },
    [pscustomobject]@{
        Name = 'D-HRQS1-seed2'
        Seed = 2
        Config = $hrqsConfig
        Output = Join-Path $workspace 'outputs\D_HRQS1_GQ1_B32_60E_TESTDEV\seed2'
    },
    [pscustomobject]@{
        Name = 'D-CONTROL-GQ1-seed2'
        Seed = 2
        Config = $controlConfig
        Output = Join-Path $workspace 'outputs\D_CONTROL_GQ1_B32_60E_TESTDEV\seed2'
    }
)

$activeTraining = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and $_.CommandLine -match 'train.py'
}
if ($activeTraining) {
    throw "Another training process is active. PID: $($activeTraining.ProcessId -join ', ')"
}
foreach ($run in $runs) {
    if (Test-Path -LiteralPath (Join-Path $run.Output 'log.txt') -PathType Leaf) {
        throw "Refusing to overwrite an existing run: $($run.Output)"
    }
}

New-Item -ItemType Directory -Path $queueDir -Force | Out-Null
$runs | Select-Object Name,Seed,Config,Output |
    ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $queueDir 'queue_plan.json') -Encoding UTF8

$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'

foreach ($run in $runs) {
    New-Item -ItemType Directory -Path $run.Output -Force | Out-Null
    $artifactDir = Join-Path $run.Output 'artifacts'
    New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
    Copy-Item -LiteralPath $run.Config -Destination $artifactDir -Force
    Copy-Item -LiteralPath (Join-Path $repo 'src\zoo\dfine\dfine.py') -Destination $artifactDir -Force
    Copy-Item -LiteralPath (Join-Path $repo 'src\zoo\dfine\dfine_decoder.py') -Destination $artifactDir -Force

    [pscustomobject]@{
        Name = $run.Name
        Seed = $run.Seed
        Status = 'running'
        StartedAt = (Get-Date).ToString('o')
        Output = $run.Output
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $queueDir 'current.json') -Encoding UTF8

    $arguments = @(
        '-u', 'train.py',
        '-c', $run.Config,
        '-t', $checkpoint,
        '--seed', [string]$run.Seed,
        '--use-amp',
        '--output-dir', $run.Output
    )
    $process = Start-Process -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $repo `
        -RedirectStandardOutput (Join-Path $run.Output 'train_console.log') `
        -RedirectStandardError (Join-Path $run.Output 'train_error.log') `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if ($process.ExitCode -ne 0) {
        throw "$($run.Name) failed with exit code $($process.ExitCode)"
    }
}

[pscustomobject]@{
    Status = 'complete'
    CompletedAt = (Get-Date).ToString('o')
    Runs = @($runs.Name)
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $queueDir 'complete.json') -Encoding UTF8
