$ErrorActionPreference = 'Stop'
$python = 'D:\conda\envs\hello_word\python.exe'
$repo = 'E:\two_paper\D-FINE'
$run = 'E:\two_paper\outputs\SBR3_REPLACE_20261004'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$pre=Get-Content -LiteralPath (Join-Path $run 'preflight.json') -Raw | ConvertFrom-Json
if($pre.status -ne 'PASS' -or $pre.precision -ne 'float32'){throw 'Missing FP32 preflight'}
$other=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'train_sbr[123].*\.py|train\.py'})
if($other.Count){throw 'Another training process is running'}
$lock=[System.IO.File]::Open((Join-Path $run 'queue.lock'),[System.IO.FileMode]::OpenOrCreate,[System.IO.FileAccess]::ReadWrite,[System.IO.FileShare]::None)
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
try {
    foreach($arm in @('R2_SAM','R1_GT_ONLY')) {
        if(Test-Path -LiteralPath (Join-Path $run "$arm\complete.json")){continue}
        $stdout=Join-Path $run "$arm.$stamp.stdout.log"
        $stderr=Join-Path $run "$arm.$stamp.stderr.log"
        if((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)){throw 'Log collision'}
        $process=Start-Process -FilePath $python -ArgumentList @('-u','tools\train_sbr3_replace.py','--arm',$arm,'--batch-size',([string]$pre.batch_size),'--epochs','30') -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $handle=$process.Handle
        @{state='running';arm=$arm;queue_pid=$PID;child_pid=$process.Id;stdout=$stdout;stderr=$stderr;time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'queue_status.json') -Encoding UTF8
        $process.WaitForExit()
        if($null -eq $process.ExitCode -or $process.ExitCode -ne 0){throw "Training failed: $arm, exit=$($process.ExitCode)"}
        if(-not (Test-Path -LiteralPath (Join-Path $run "$arm\complete.json"))){throw 'Missing completion marker'}
        $process.Dispose()
    }
    @{state='complete';time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'queue_status.json') -Encoding UTF8
} catch {
    $failure=@{state='failed';message=$_.Exception.Message;time=(Get-Date).ToString('o')}
    $failure | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'queue_status.json') -Encoding UTF8
    $failure | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'status.json') -Encoding UTF8
    throw
} finally {$lock.Dispose()}


