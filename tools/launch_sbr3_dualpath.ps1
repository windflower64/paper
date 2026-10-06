$ErrorActionPreference='Stop'
$repo='E:\two_paper\D-FINE'
$run='E:\two_paper\outputs\SBR3_DUALPATH_20261006'
$python='D:\conda\envs\hello_word\python.exe'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$pre=Get-Content -LiteralPath "$run\preflight.json" -Raw | ConvertFrom-Json
if($pre.status -ne 'PASS'){throw 'Preflight missing'}
if((Get-FileHash -LiteralPath "$repo\tools\train_sbr3_dualpath.py" -Algorithm SHA256).Hash.ToLower() -ne $pre.script_sha){throw 'Script changed after preflight'}
$other=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'train_.*\.py'})
if($other.Count){throw 'Another trainer is running'}
if(Test-Path -LiteralPath "$run\DUAL"){throw 'Run already exists; inspect before restarting'}
$lock=[IO.File]::Open("$run\queue.lock",[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
try {
 $stdout="$run\DUAL.$stamp.stdout.log"
 $stderr="$run\DUAL.$stamp.stderr.log"
 $p=Start-Process -FilePath $python -ArgumentList @('-u','tools\train_sbr3_dualpath.py','--arm','DUAL','--batch-size','16','--epochs','20') -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
 $handle=$p.Handle
 @{state='running';arm='DUAL';pid=$p.Id;stdout=$stdout;stderr=$stderr} | ConvertTo-Json | Set-Content -LiteralPath "$run\queue_status.json" -Encoding UTF8
 $p.WaitForExit()
 if($null -eq $p.ExitCode -or $p.ExitCode -ne 0){throw 'Training failed'}
 if(!(Test-Path -LiteralPath "$run\DUAL\complete.json")){throw 'Missing completion marker'}
 $p.Dispose()
 @{state='complete';time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath "$run\queue_status.json" -Encoding UTF8
} catch {
 @{state='failed';error=$_.Exception.Message} | ConvertTo-Json | Set-Content -LiteralPath "$run\queue_status.json" -Encoding UTF8
 throw
} finally {$lock.Dispose()}
