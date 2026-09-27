#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/rgbt_experiments/D-FINE
run=/root/autodl-tmp/rgbt_experiments/runs/20_spatial_importance/S_PDBR1_POSDIR_BOUNDARY_S8_S16/seed0
queue_log="$run/queue.log"
mkdir -p "$run"
cd "$repo"

echo "$(date -Is) waiting for HBS1 to release the GPU" >> "$queue_log"
while pgrep -f "train.py.*s_hbs1_bg_smooth_aux_p16_p32.yml" >/dev/null; do
  sleep 30
done

echo "$(date -Is) HBS1 finished; repeating PDBR preflight" >> "$queue_log"
OMP_NUM_THREADS=4 /root/autodl-tmp/envs/dfine/bin/python tools/preflight_s_pdbr1.py \
  >> "$queue_log" 2>&1
OMP_NUM_THREADS=4 /root/autodl-tmp/envs/dfine/bin/python tools/smoke_s_pdbr1_detector.py \
  >> "$queue_log" 2>&1

echo "$(date -Is) launching 60-epoch PDBR1 training" >> "$queue_log"
tools/launch_s_pdbr1.sh >> "$queue_log" 2>&1
pid=$(cat "$run/train.pid")
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done

if [[ ! -s "$run/best_stg1.pth" ]]; then
  echo "$(date -Is) ERROR: training ended without best_stg1.pth" >> "$queue_log"
  exit 1
fi

echo "$(date -Is) training finished; running causal controls" >> "$queue_log"
tools/run_s_pdbr1_controls.sh >> "$queue_log" 2>&1
echo "$(date -Is) PDBR1 queue complete" >> "$queue_log"
