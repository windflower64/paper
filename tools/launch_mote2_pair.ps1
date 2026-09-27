$ErrorActionPreference = 'Stop'
$python = 'D:\conda\envs\hello_word\python.exe'
$repo = 'E:\two_paper\D-FINE'
$root = 'E:\two_paper'
$statusPath = Join-Path $root 'reports\M_OTE2_C_S_PAIR_20E_TESTDEV\training_queue_status.json'
$runs = @(
    @{ Name = 'E0'; Config = Join-Path $repo 'experiments\phase_m\mote2_e0_no_m_b8a4_20e.yml'; Output = Join-Path $root 'outputs\M_OTE2_C_S_PAIR_20E_TESTDEV\E0\seed0' },
    @{ Name = 'E1'; Config = Join-Path $repo 'experiments\phase_m\mote2_e1_sd22_correct_ir_b8a4_20e.yml'; Output = Join-Path $root 'outputs\M_OTE2_C_S_PAIR_20E_TESTDEV\E1\seed0' },
    @{ Name = 'E2'; Config = Join-Path $repo 'experiments\phase_m\mote2_e2_ote2_b8a4_20e.yml'; Output = Join-Path $root 'outputs\M_OTE2_C_S_PAIR_20E_TESTDEV\E2\seed0' }
)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Python not found: $python" }
if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw "Repository not found: $repo" }
function Test-CompletedRun($output) {
    $logPath = Join-Path $output 'log.txt'
    $lastPath = Join-Path $output 'last.pth'
    $consolePath = Join-Path $output 'train_console.log'
    if (-not (Test-Path -LiteralPath $logPath -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath $lastPath -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath $consolePath -PathType Leaf)) { return $false }
    $lines = @(Get-Content -LiteralPath $logPath)
    if ($lines.Count -ne 20) { return $false }
    try { $finalEpoch = [int](($lines[-1] | ConvertFrom-Json).epoch) }
    catch { return $false }
    if ($finalEpoch -ne 19) { return $false }
    return [bool](Select-String -LiteralPath $consolePath -Pattern '^Training time ' -Quiet)
}
foreach ($run in $runs) {
    if (-not (Test-Path -LiteralPath $run.Config -PathType Leaf)) { throw "Config not found: $($run.Config)" }
    if (Test-Path -LiteralPath $run.Output) {
        $items = @(Get-ChildItem -LiteralPath $run.Output -Force)
        if ($items.Count -gt 0 -and -not (Test-CompletedRun $run.Output)) {
            throw "Refusing to overwrite incomplete non-empty output directory: $($run.Output)"
        }
    }
}

$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Path (Split-Path -Parent $statusPath) -Force | Out-Null
function Write-Status($data) {
    $data | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

Write-Status @{ status = 'preflight_passed'; updated_at = (Get-Date).ToString('o'); plan = 'E0 -> E1 -> E2'; batch = 8; accumulation = 4; epochs = 20 }
foreach ($run in $runs) {
    if (Test-CompletedRun $run.Output) {
        Write-Status @{ status = "completed_$($run.Name)"; output = $run.Output; updated_at = (Get-Date).ToString('o') }
        continue
    }
    New-Item -ItemType Directory -Path $run.Output -Force | Out-Null
    $stdout = Join-Path $run.Output 'train_console.log'
    $stderr = Join-Path $run.Output 'train_error.log'
    Write-Status @{ status = "training_$($run.Name)"; config = $run.Config; output = $run.Output; updated_at = (Get-Date).ToString('o') }
    $arguments = @('-u', 'train.py', '-c', $run.Config, '--seed', '0', '--use-amp')
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
    $process.Refresh()
    if ($null -eq $process.ExitCode -or $process.ExitCode -ne 0 -or -not (Test-CompletedRun $run.Output)) {
        Write-Status @{ status = "failed_$($run.Name)"; exit_code = $process.ExitCode; output = $run.Output; updated_at = (Get-Date).ToString('o') }
        throw "$($run.Name) failed or did not complete 20 epochs; exit code $($process.ExitCode). See $stderr"
    }
    Write-Status @{ status = "completed_$($run.Name)"; output = $run.Output; updated_at = (Get-Date).ToString('o') }
}
Write-Status @{ status = 'completed_all'; outputs = @($runs | ForEach-Object { $_.Output }); updated_at = (Get-Date).ToString('o') }
