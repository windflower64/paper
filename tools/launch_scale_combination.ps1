$ErrorActionPreference='Stop'
$repo='E:\two_paper\D-FINE'
$run='E:\two_paper\outputs\SCALE_COMBINATION_20261005'
$python='D:\conda\envs\hello_word\python.exe'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$pre=Get-Content -LiteralPath "$run\preflight.json" -Raw | ConvertFrom-Json
if($pre.status -ne 'PASS'){throw 'Preflight missing'}
$other=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match 'train_.*\.py'})
if($other.Count){throw 'Another trainer is running'}
$lock=[IO.File]::Open("$run\queue.lock",[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
try {
 foreach($arm in @('MULTISCALE','S8')) {
  if(Test-Path -LiteralPath "$run\$arm\complete.json"){continue}
  if(Test-Path -LiteralPath "$run\$arm"){throw 'Partial run exists; inspect before restarting'}
  $stdout="$run\$arm.$stamp.stdout.log"
  $stderr="$run\$arm.$stamp.stderr.log"
  $p=Start-Process -FilePath $python -ArgumentList @('-u','tools\train_scale_combination.py','--arm',$arm,'--batch-size',([string]$pre.batch_size),'--epochs','20') -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  $handle=$p.Handle
  @{state='running';arm=$arm;pid=$p.Id;stdout=$stdout;stderr=$stderr} | ConvertTo-Json | Set-Content -LiteralPath "$run\queue_status.json" -Encoding UTF8
  $p.WaitForExit()
  if($null -eq $p.ExitCode -or $p.ExitCode -ne 0){throw "Training failed $arm"}
  if(!(Test-Path -LiteralPath "$run\$arm\complete.json")){throw 'Missing completion marker'}
  $p.Dispose()
 }
 @{state='complete';time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath "$run\queue_status.json" -Encoding UTF8
} catch {
 @{state='failed';error=$_.Exception.Message} | ConvertTo-Json | Set-Content -LiteralPath "$run\queue_status.json" -Encoding UTF8
 throw
} finally {$lock.Dispose()}


