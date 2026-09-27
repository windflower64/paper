param(
    [int]$InitialB0Pid = 0,
    [int]$Seed = 0
)

$ErrorActionPreference = 'Stop'

$workspace = 'E:\two_paper'
$repo = Join-Path $workspace 'D-FINE'
$launcher = Join-Path $repo 'tools\launch_stql_qcer_arm.ps1'
$queueDir = Join-Path $workspace 'outputs\STQL_QCER_QUEUE_B0_B5_20260921'
$queueLog = Join-Path $queueDir 'queue.log'
$statePath = Join-Path $queueDir 'queue_state.json'
$arms = @('B0', 'B1', 'B2', 'B5', 'B3', 'B4')
$runNames = @{
    B0 = 'STQL_QCER_B0_RGB_B16A2_20E_TESTDEV'
    B1 = 'STQL_QCER_B1_STQL_SAM_B16A2_20E_TESTDEV'
    B2 = 'STQL_QCER_B2_QCER_B16A2_20E_TESTDEV'
    B3 = 'STQL_QCER_B3_BOX_QCER_B16A2_20E_TESTDEV'
    B4 = 'STQL_QCER_B4_SAM_QCER_B16A2_20E_TESTDEV'
    B5 = 'STQL_QCER_B5_UNIFORM_QCER_B16A2_20E_TESTDEV'
}

New-Item -ItemType Directory -Path $queueDir -Force | Out-Null
$state = [ordered]@{
    schema = 'stql_qcer_b0_b5_serial_queue_v1'
    status = 'RUNNING'
    seed = $Seed
    order = $arms
    current_arm = 'B0'
    updated_at = (Get-Date).ToString('o')
    runs = @()
}

function Write-QueueLog([string]$Message) {
    $line = '[{0}] {1}' -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $queueLog -Value $line -Encoding UTF8
}

function Save-State {
    $state.updated_at = (Get-Date).ToString('o')
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Get-RunPath([string]$Arm) {
    return Join-Path $workspace ("outputs\{0}\seed{1}" -f $runNames[$Arm], $Seed)
}

function Assert-CompletedRun([string]$Arm, [string]$RunPath, [Nullable[int]]$ExitCode) {
    $lastCheckpoint = Join-Path $RunPath 'last.pth'
    $bestCheckpoint = Join-Path $RunPath 'best_stg1.pth'
    $epochLog = Join-Path $RunPath 'log.txt'
    $consoleLog = Join-Path $RunPath 'train_console.log'
    $errorLog = Join-Path $RunPath 'train_error.log'

    if ($null -ne $ExitCode -and $ExitCode.Value -ne 0) {
        throw "$Arm process exited with code $($ExitCode.Value)"
    }
    foreach ($file in @($lastCheckpoint, $bestCheckpoint, $epochLog, $consoleLog)) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "$Arm completed without required artifact: $file"
        }
    }

    $records = @()
    foreach ($line in Get-Content -LiteralPath $epochLog -Encoding UTF8) {
        if (-not [string]::IsNullOrWhiteSpace($line)) {
            $records += ($line | ConvertFrom-Json)
        }
    }
    if ($records.Count -ne 20) {
        throw "$Arm expected 20 epoch records, found $($records.Count)"
    }
    if ([int]$records[-1].epoch -ne 19) {
        throw "$Arm final epoch is $($records[-1].epoch), expected 19"
    }
    if (-not (Select-String -LiteralPath $consoleLog -Pattern '^Training time ' -Quiet)) {
        throw "$Arm console log has no normal Training time marker"
    }
    if (Test-Path -LiteralPath $errorLog -PathType Leaf) {
        $fatal = Select-String -LiteralPath $errorLog `
            -Pattern 'Traceback|RuntimeError|CUDA error|OutOfMemory|fatal:' `
            -CaseSensitive:$false
        if ($fatal) {
            throw "$Arm error log contains a fatal pattern: $($fatal[-1].Line)"
        }
    }

    $bbox = @($records[-1].test_coco_eval_bbox)
    return [ordered]@{
        arm = $Arm
        status = 'PASS'
        output = $RunPath
        epochs = $records.Count
        final_epoch = [int]$records[-1].epoch
        final_ap = if ($bbox.Count -gt 0) { [double]$bbox[0] } else { $null }
        last_checkpoint_sha256 = (Get-FileHash -LiteralPath $lastCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        best_checkpoint_sha256 = (Get-FileHash -LiteralPath $bestCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        completed_at = (Get-Date).ToString('o')
    }
}

function Wait-ForRun([string]$Arm, [int]$PidValue, [string]$RunPath) {
    Write-QueueLog "Waiting for $Arm PID=$PidValue output=$RunPath"
    $exitCode = $null
    try {
        $process = [System.Diagnostics.Process]::GetProcessById($PidValue)
        $process.WaitForExit()
        $exitCode = [Nullable[int]]$process.ExitCode
    }
    catch [System.ArgumentException] {
        Write-QueueLog "$Arm PID=$PidValue already exited; validating durable artifacts"
    }
    return Assert-CompletedRun -Arm $Arm -RunPath $RunPath -ExitCode $exitCode
}

try {
    if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
        throw "Launcher is missing: $launcher"
    }
    if ($InitialB0Pid -le 0) {
        throw 'InitialB0Pid must identify the already running B0 process'
    }
    Save-State
    Write-QueueLog "Queue started; order=$($arms -join ' -> ')"

    $b0Path = Get-RunPath 'B0'
    $state.current_arm = 'B0'
    Save-State
    $state.runs += Wait-ForRun -Arm 'B0' -PidValue $InitialB0Pid -RunPath $b0Path
    Save-State

    foreach ($arm in $arms[1..($arms.Count - 1)]) {
        $state.current_arm = $arm
        Save-State
        Write-QueueLog "Launching $arm"
        $launch = & $launcher -Arm $arm -Seed $Seed
        if ($null -eq $launch -or [int]$launch.PID -le 0) {
            throw "$arm launcher returned no valid PID"
        }
        $state.runs += [ordered]@{
            arm = $arm
            status = 'RUNNING'
            pid = [int]$launch.PID
            output = [string]$launch.Output
            started_at = (Get-Date).ToString('o')
        }
        Save-State
        $completed = Wait-ForRun -Arm $arm -PidValue ([int]$launch.PID) -RunPath ([string]$launch.Output)
        $state.runs = @($state.runs | Where-Object { -not ($_.arm -eq $arm -and $_.status -eq 'RUNNING') })
        $state.runs += $completed
        Save-State
    }

    $state.status = 'PASS'
    $state.current_arm = $null
    Save-State
    Write-QueueLog 'All B0-B5 runs passed the completion gate'
}
catch {
    $state.status = 'FAILED'
    $state.failure = $_.Exception.Message
    Save-State
    Write-QueueLog "QUEUE FAILED: $($_.Exception.Message)"
    throw
}
