$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$projectDir = 'E:\two_paper\D-FINE'
$pythonExe = 'D:\conda\envs\hello_word\python.exe'
$configPath = 'E:\two_paper\D-FINE\experiments\phase_s\s_box2_extreme_point_jointinit_b32_60e_local.yml'
$tuningPath = 'E:\two_paper\weights\dfine_n_coco.pth'
$outputDir = 'E:\two_paper\runs\23_sam_box_alignment\S_BOX2_EXTREME_POINT_JOINTINIT_B32_60E\seed0'

foreach ($requiredPath in @($workspace, $projectDir, $pythonExe, $configPath, $tuningPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "缺少必要路径：$requiredPath"
    }
}

if (Test-Path -LiteralPath $outputDir) {
    $existing = @(Get-ChildItem -LiteralPath $outputDir -Force -ErrorAction Stop)
    if ($existing.Count -gt 0) {
        throw "正式输出目录不是空目录，为避免覆盖已停止：$outputDir"
    }
} else {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$stdoutPath = Join-Path $outputDir 'train_console.log'
$stderrPath = Join-Path $outputDir 'train_stderr.log'
$arguments = @(
    '-X', 'utf8',
    'train.py',
    '-c', $configPath,
    '-t', $tuningPath,
    '--seed', '0',
    '--use-amp'
)

$startArguments = @{
    FilePath = $pythonExe
    ArgumentList = $arguments
    WorkingDirectory = $projectDir
    RedirectStandardOutput = $stdoutPath
    RedirectStandardError = $stderrPath
    WindowStyle = 'Hidden'
    PassThru = $true
}
$process = Start-Process @startArguments

[pscustomobject]@{
    pid = $process.Id
    output_dir = $outputDir
    stdout = $stdoutPath
    stderr = $stderrPath
} | ConvertTo-Json
