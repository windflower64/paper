$ErrorActionPreference = 'Stop'
$py = 'D:\conda\envs\hello_word\python.exe'
$repo = 'E:\two_paper\D-FINE'
$report = 'E:\two_paper\reports\162_stql_qcer_causal'
$tool = Join-Path $repo 'tools\eval_stql_qcer_causal.py'
$jobs = @(
    @{ Arm='B4'; Mode='normal'; Config=(Join-Path $repo 'experiments\phase_stql_qcer\b4_sam_qcer.yml'); Checkpoint='E:\two_paper\outputs\STQL_QCER_B4_SAM_QCER_B16A2_20E_TESTDEV\seed0\best_stg1.pth' },
    @{ Arm='B4'; Mode='bypass'; Config=(Join-Path $repo 'experiments\phase_stql_qcer\b4_sam_qcer.yml'); Checkpoint='E:\two_paper\outputs\STQL_QCER_B4_SAM_QCER_B16A2_20E_TESTDEV\seed0\best_stg1.pth' },
    @{ Arm='B4'; Mode='zero_available'; Config=(Join-Path $repo 'experiments\phase_stql_qcer\b4_sam_qcer.yml'); Checkpoint='E:\two_paper\outputs\STQL_QCER_B4_SAM_QCER_B16A2_20E_TESTDEV\seed0\best_stg1.pth' },
    @{ Arm='B4'; Mode='batch_shuffle'; Config=(Join-Path $repo 'experiments\phase_stql_qcer\b4_sam_qcer.yml'); Checkpoint='E:\two_paper\outputs\STQL_QCER_B4_SAM_QCER_B16A2_20E_TESTDEV\seed0\best_stg1.pth' },
    @{ Arm='B4'; Mode='unavailable'; Config=(Join-Path $repo 'experiments\phase_stql_qcer\b4_sam_qcer.yml'); Checkpoint='E:\two_paper\outputs\STQL_QCER_B4_SAM_QCER_B16A2_20E_TESTDEV\seed0\best_stg1.pth' },
    @{ Arm='B4'; Mode='token_order_shuffle'; Config=(Join-Path $repo 'experiments\phase_stql_qcer\b4_sam_qcer.yml'); Checkpoint='E:\two_paper\outputs\STQL_QCER_B4_SAM_QCER_B16A2_20E_TESTDEV\seed0\best_stg1.pth' },
    @{ Arm='B5'; Mode='normal'; Config=(Join-Path $repo 'experiments\phase_stql_qcer\b5_uniform_qcer.yml'); Checkpoint='E:\two_paper\outputs\STQL_QCER_B5_UNIFORM_QCER_B16A2_20E_TESTDEV\seed0\best_stg1.pth' },
    @{ Arm='B5'; Mode='bypass'; Config=(Join-Path $repo 'experiments\phase_stql_qcer\b5_uniform_qcer.yml'); Checkpoint='E:\two_paper\outputs\STQL_QCER_B5_UNIFORM_QCER_B16A2_20E_TESTDEV\seed0\best_stg1.pth' }
)
New-Item -ItemType Directory -Path $report -Force | Out-Null
foreach ($job in $jobs) {
    $output = Join-Path $report ("{0}_{1}.json" -f $job.Arm, $job.Mode)
    if (Test-Path $output) { continue }
    $status = @{
        status = 'running'
        arm = $job.Arm
        mode = $job.Mode
        output = $output
        updated_at = (Get-Date).ToString('o')
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report 'queue_status.json') -Encoding UTF8
    & $py -B $tool --repo $repo --config $job.Config --checkpoint $job.Checkpoint --mode $job.Mode --output $output
    if ($LASTEXITCODE -ne 0) { throw "$($job.Arm) $($job.Mode) failed: $LASTEXITCODE" }
}
@{ status='completed'; updated_at=(Get-Date).ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report 'queue_status.json') -Encoding UTF8
