$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$queueScript = Join-Path $repo 'tools\run_scm1_a0375_multiseed_queue.ps1'
$reportDir = Join-Path $workspace 'reports\70_scm_joint\SCM1_A0375_MULTISEED'

if (-not (Test-Path -LiteralPath $queueScript -PathType Leaf)) {
    throw "Missing queue script: $queueScript"
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    $_.Name -like 'powershell*.exe' -and
    $_.CommandLine -like '*run_scm1_a0375_multiseed_queue.ps1*'
}
if ($duplicate) {
    throw "SCM1 multiseed queue is already running, PID: $($duplicate.ProcessId -join ', ')"
}

New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$stdout = Join-Path $reportDir 'queue_console.log'
$stderr = Join-Path $reportDir 'queue_error.log'
$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $queueScript
)
$process = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList $arguments `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    Experiment = 'SCM1-A0375-Multiseed'
    PID = $process.Id
    QueueLog = $stdout
    QueueError = $stderr
}
