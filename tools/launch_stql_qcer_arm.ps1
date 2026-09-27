param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('B0', 'B1', 'B2', 'B3', 'B4', 'B5')]
    [string]$Arm,

    [int]$Seed = 0,

    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$python = 'D:\conda\envs\hello_word\python.exe'
$commonInit = Join-Path $workspace 'weights\stql_qcer_common_init_v1.pth'
$configAudit = Join-Path $workspace 'reports\161_stql_qcer_v1\config_preflight.json'
$shortTrain = Join-Path $workspace 'outputs\STQL_QCER_PREFLIGHT_B4_B16A2\seed0\short_train_report.json'
$inferenceAudit = Join-Path $workspace 'reports\161_stql_qcer_v1\inference_contract.json'

$configs = @{
    B0 = 'b0_rgb.yml'
    B1 = 'b1_stql_sam.yml'
    B2 = 'b2_qcer.yml'
    B3 = 'b3_box_qcer.yml'
    B4 = 'b4_sam_qcer.yml'
    B5 = 'b5_uniform_qcer.yml'
}
$names = @{
    B0 = 'STQL_QCER_B0_RGB_B16A2_20E_TESTDEV'
    B1 = 'STQL_QCER_B1_STQL_SAM_B16A2_20E_TESTDEV'
    B2 = 'STQL_QCER_B2_QCER_B16A2_20E_TESTDEV'
    B3 = 'STQL_QCER_B3_BOX_QCER_B16A2_20E_TESTDEV'
    B4 = 'STQL_QCER_B4_SAM_QCER_B16A2_20E_TESTDEV'
    B5 = 'STQL_QCER_B5_UNIFORM_QCER_B16A2_20E_TESTDEV'
}

$config = Join-Path $repo ('experiments\phase_stql_qcer\' + $configs[$Arm])
$run = Join-Path $workspace ("outputs\{0}\seed{1}" -f $names[$Arm], $Seed)

foreach ($file in @($python, $config, $commonInit, $configAudit, $shortTrain, $inferenceAudit)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing required launch asset: $file"
    }
}

$configAuditData = Get-Content -LiteralPath $configAudit -Raw -Encoding UTF8 | ConvertFrom-Json
$shortTrainData = Get-Content -LiteralPath $shortTrain -Raw -Encoding UTF8 | ConvertFrom-Json
$inferenceAuditData = Get-Content -LiteralPath $inferenceAudit -Raw -Encoding UTF8 | ConvertFrom-Json
if ($configAuditData.status -ne 'PASS') { throw 'Configuration audit did not pass' }
if ($shortTrainData.status -ne 'PASS') { throw 'Five-step training and resume audit did not pass' }
if ($inferenceAuditData.status -ne 'PASS') { throw 'Inference contract audit did not pass' }
if ([int]$configAuditData.arms.$Arm.physical_batch -ne 16) { throw "$Arm physical batch is not 16" }
if ([int]$configAuditData.arms.$Arm.effective_batch -ne 32) { throw "$Arm effective batch is not 32" }
if (-not [bool]$configAuditData.common_rgb_state_bitwise_equal) {
    throw 'B0-B5 do not share a bitwise-identical RGB initialization'
}

if ($ValidateOnly) {
    return [pscustomobject]@{
        Status = 'PASS'
        Arm = $Arm
        Seed = $Seed
        Config = $config
        Output = $run
        PhysicalBatch = [int]$configAuditData.arms.$Arm.physical_batch
        EffectiveBatch = [int]$configAuditData.arms.$Arm.effective_batch
        CommonRGBStateBitwiseEqual = [bool]$configAuditData.common_rgb_state_bitwise_equal
        TrainingStarted = $false
    }
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python' -and $_.CommandLine -like "*$($configs[$Arm])*"
}
if ($duplicate) {
    throw "$Arm is already running. PID: $($duplicate.ProcessId -join ', ')"
}
if (Test-Path -LiteralPath (Join-Path $run 'log.txt') -PathType Leaf) {
    throw "Refusing to overwrite an existing training run: $run"
}

$artifactDir = Join-Path $run 'artifacts'
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$artifacts = @(
    $config,
    (Join-Path $repo 'experiments\phase_stql_qcer\common_v1.yml'),
    (Join-Path $repo 'experiments\phase_stql_qcer\qcer_rgbt_common.yml'),
    $commonInit,
    $configAudit,
    $shortTrain,
    $inferenceAudit,
    (Join-Path $repo 'src\zoo\dfine\stql_qcer.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_decoder.py'),
    (Join-Path $repo 'src\zoo\dfine\dfine_criterion.py'),
    (Join-Path $repo 'src\data\dataset\coco_dataset.py'),
    (Join-Path $repo 'src\solver\det_engine.py')
)
foreach ($artifact in $artifacts | Select-Object -Unique) {
    Copy-Item -LiteralPath $artifact -Destination $artifactDir -Force
}
$manifest = foreach ($artifact in Get-ChildItem -LiteralPath $artifactDir -File) {
    [pscustomobject]@{
        File = $artifact.Name
        Bytes = $artifact.Length
        SHA256 = (Get-FileHash -LiteralPath $artifact.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $artifactDir 'SHA256_MANIFEST.json') -Encoding UTF8

$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$arguments = @(
    '-u', 'train.py',
    '-c', $config,
    '-t', $commonInit,
    '--seed', "$Seed",
    '--use-amp',
    '--output-dir', $run
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
    Arm = $Arm
    PID = $process.Id
    Seed = $Seed
    Epochs = 20
    PhysicalBatch = 16
    GradientAccumulation = 2
    EffectiveBatch = 32
    Output = $run
    ConsoleLog = (Join-Path $run 'train_console.log')
    ErrorLog = (Join-Path $run 'train_error.log')
}
