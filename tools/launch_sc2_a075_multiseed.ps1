$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$queueScript = Join-Path $repo 'tools\run_sc2_a075_multiseed_queue.ps1'
$reportDir = Join-Path $workspace 'reports\SC_joint\SC2_A075_MULTISEED'
$queueLog = Join-Path $reportDir 'queue.log'
$queueError = Join-Path $reportDir 'queue_error.log'

if (-not (Test-Path -LiteralPath $queueScript -PathType Leaf)) {
    throw "缺少队列脚本：$queueScript"
}
$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*run_sc2_a075_multiseed_queue.ps1*'
}
if ($duplicate) {
    throw "SC2 A075多seed队列已经在运行，PID：$($duplicate.ProcessId -join ', ')"
}

New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $queueScript
)
$process = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $queueLog `
    -RedirectStandardError $queueError `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    Experiment = 'SC2-A075-Multiseed-Seed1-Seed2'
    PID = $process.Id
    QueueLog = $queueLog
    QueueError = $queueError
}
