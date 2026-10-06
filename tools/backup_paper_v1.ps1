$ErrorActionPreference='Stop'
$src='E:\two_paper\D-FINE'
$dst='E:\two_paper\D-FINE-paper-publish'
$root='E:\two_paper'
if((git -C $dst status --porcelain)){throw 'Publish checkout is not clean'}
git -C $dst switch -c codex/paper-v1-20261006
if($LASTEXITCODE -ne 0){throw 'Cannot create backup branch'}
$files=@(
'src/zoo/dfine/dfine.py','src/zoo/dfine/sam_side_search.py','src/zoo/dfine/target_evidence_thermal_fusion.py',
'src/zoo/dfine/sam_multiscale_group.py','src/zoo/dfine/sam_side_search_multiscale.py','src/zoo/dfine/thermal_p32_extension.py',
'tools/train_sbr1.py','tools/preflight_sqer1.py','tools/evaluate_m_layout_recovery.py',
'tools/train_sbr3_replace.py','tools/train_sgc_multiscale_joint.py','tools/train_sbr3_context.py','tools/train_scale_combination.py','tools/train_m_p32.py','tools/train_sbr3_dualpath.py',
'tools/analyze_sbr3_replace_results.py','tools/analyze_sgc_multiscale_joint.py','tools/analyze_sbr3_context.py','tools/analyze_scale_combination.py','tools/analyze_m_p32.py','tools/analyze_sbr3_dualpath.py',
'tools/check_sbr3_context_dependence.py','tools/check_m_p32_best_intervention.py','tools/diagnose_s16_gradients.py','tools/inventory_scale_candidates.py','tools/evaluate_paper_v1.py',
'tools/preflight_sgc_multiscale.py','tools/test_sgc_multiscale.py','tools/audit_sgc_multiscale_coverage.py',
'tools/launch_sbr3_replace_queue.ps1','tools/launch_sgc_multiscale_joint.ps1','tools/launch_sbr3_context.ps1','tools/launch_scale_combination.ps1','tools/launch_m_p32.ps1','tools/launch_sbr3_dualpath.ps1','tools/backup_paper_v1.ps1')
foreach($f in $files){
 $from=Join-Path $src $f; $to=Join-Path $dst $f
 if(!(Test-Path -LiteralPath $from -PathType Leaf)){throw "Missing $from"}
 New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
 Copy-Item -LiteralPath $from -Destination $to
}
$docs=@(Get-ChildItem -LiteralPath "$root\knowledge\论文写作前置准备" -File | Where-Object {$_.Name -match '^2026100[4-6]_.*\.md$'})
foreach($d in $docs){
 $to=Join-Path "$dst\knowledge\论文写作前置准备" $d.Name
 New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
 Copy-Item -LiteralPath $d.FullName -Destination $to
}
$runs=@('SBR3_REPLACE_20261004','SGC_MULTISCALE_JOINT_20261005','SBR3_CONTEXT_20261005','M_P32_20261005','SCALE_COMBINATION_20261005','SBR3_DUALPATH_20261006')
foreach($run in $runs){
 foreach($f in Get-ChildItem -LiteralPath "$root\outputs\$run" -Recurse -File | Where-Object {$_.Name -in @('protocol.json','log.jsonl','complete.json','preflight.json')}){
  $relative=$f.FullName.Substring($root.Length+1)
  $to=Join-Path "$dst\paper_v1_evidence" $relative
  New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
  Copy-Item -LiteralPath $f.FullName -Destination $to
 }
}
foreach($run in ($runs+@('S16_GRADIENT_AUDIT_20261006','SCALE_CANDIDATE_SELECTION_20261006','PAPER_V1_20261006'))){
 $folder="$root\reports\$run"
 if(Test-Path -LiteralPath $folder){
  foreach($f in Get-ChildItem -LiteralPath $folder -File | Where-Object {$_.Extension -in @('.json','.md') -and $_.Name -ne 'sources.json'}){
   $to=Join-Path "$dst\paper_v1_evidence\reports\$run" $f.Name
   New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
   Copy-Item -LiteralPath $f.FullName -Destination $to
  }
 }
}
$weight="$root\outputs\SBR3_CONTEXT_20261005\CONTEXT\best.pth"
if((Get-FileHash -LiteralPath $weight -Algorithm SHA256).Hash.ToLower() -ne '351edff4f2fb1dd9010b5064505101ed63e629b5d25a4c6155a0742385801829'){throw 'Weight changed'}
New-Item -ItemType Directory -Force -Path "$dst\paper_v1_evidence\weights" | Out-Null
Copy-Item -LiteralPath $weight -Destination "$dst\paper_v1_evidence\weights\paper_v1_full_best.pth"
git -C $src add -- $files
if($LASTEXITCODE -ne 0){throw 'Local staging failed'}
git -C $src commit -m 'Freeze PAPER_V1 and preserve scale studies and dual-path diagnostics'
if($LASTEXITCODE -ne 0){throw 'Local commit failed'}
$docPaths=@($docs | ForEach-Object {$_.FullName.Substring($root.Length+1)})
git -C $root add -- $docPaths
git -C $root commit -m 'Document PAPER_V1 selection and scale experiment evidence'
if($LASTEXITCODE -ne 0){throw 'Knowledge commit failed'}
git -C $dst add -- $files 'knowledge/论文写作前置准备'
git -C $dst add -f -- paper_v1_evidence
if($LASTEXITCODE -ne 0){throw 'Publish staging failed'}
git -C $dst diff --cached --stat
git -C $dst commit -m 'PAPER_V1: freeze RGB-T SAM-guided SBR3 model with evidence and checkpoint'
if($LASTEXITCODE -ne 0){throw 'Publish commit failed'}
