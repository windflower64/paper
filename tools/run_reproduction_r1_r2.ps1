$ErrorActionPreference = 'Stop'
$py = 'D:\conda\envs\hello_word\python.exe'
$artifact = 'E:\two_paper\outputs\C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV\seed0\artifacts'
$root = 'E:\two_paper'
$runs = @(
    @{ Name='R1'; Config='E:\two_paper\D-FINE\experiments\reproduction\r1_historical_no_m_20e.yml'; Output='E:\two_paper\outputs\REPRO_R1_HISTORICAL_NO_M_SGC2_20E_TESTDEV\seed0' },
    @{ Name='R2'; Config='E:\two_paper\D-FINE\experiments\reproduction\r2_historical_no_sgc2_20e.yml'; Output='E:\two_paper\outputs\REPRO_R2_HISTORICAL_NO_SGC2_20E_TESTDEV\seed0' }
)
$statusPath = Join-Path $root 'reports\161_baseline_reproduction\queue_status.json'
New-Item -ItemType Directory -Path (Split-Path $statusPath) -Force | Out-Null
function Write-Status($obj) { $obj | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding UTF8 }
Write-Status @{ status='waiting_for_R0'; r0_pid=13636; updated_at=(Get-Date).ToString('o') }
if (Get-Process -Id 13636 -ErrorAction SilentlyContinue) { Wait-Process -Id 13636 }
foreach ($run in $runs) {
    if (Test-Path $run.Output) {
        $existing = @(Get-ChildItem -LiteralPath $run.Output -Force)
        if ($existing.Count -gt 0) { throw "$($run.Name) output is not empty: $($run.Output)" }
    } else { New-Item -ItemType Directory -Path $run.Output -Force | Out-Null }
    $stdout = Join-Path $run.Output 'train_console.log'
    $stderr = Join-Path $run.Output 'train_error.log'
    $args = @('-u','train.py','-c',$run.Config,'-t','E:\two_paper\weights\m_sd2_joint_coco_thermal_identity_init.pth','--seed','0','--use-amp')
    Write-Status @{ status="training_$($run.Name)"; config=$run.Config; output=$run.Output; updated_at=(Get-Date).ToString('o') }
    $p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $artifact -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $p.WaitForExit()
    $p.Refresh()
    if ($p.ExitCode -ne 0) { throw "$($run.Name) failed with exit code $($p.ExitCode)" }
    Write-Status @{ status="completed_$($run.Name)"; output=$run.Output; updated_at=(Get-Date).ToString('o') }
}
Write-Status @{ status='completed_R0_R2'; updated_at=(Get-Date).ToString('o') }
