#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/rgbt_experiments
REPO="$ROOT/D-FINE"
PY=/root/autodl-tmp/envs/dfine/bin/python
A00="$ROOT/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth"
QUEUE_LOG="$ROOT/reports/30_channel/C_ID_S16_R25_queue.log"
PATTERN='train.py.*c_id_s16_r25_'

if pgrep -af "$PATTERN" >/dev/null 2>&1; then
  echo "A C-ID-S16-R25 process is already running:"
  pgrep -af "$PATTERN"
  exit 1
fi

mkdir -p "$ROOT/reports/30_channel"

run_one() {
  local name=$1
  local cfg=$2
  local out=$3
  mkdir -p "$out"
  echo "[$(date '+%F %T')] START $name" | tee -a "$QUEUE_LOG"
  "$PY" -u train.py -c "$cfg" -t "$A00" --seed 0 \
    >"$out/train.log" 2>&1
  echo "[$(date '+%F %T')] DONE $name" | tee -a "$QUEUE_LOG"
}

cd "$REPO"
run_one TASK \
  "$REPO/experiments/phase_c/c_id_s16_r25_task.yml" \
  "$ROOT/runs/30_channel/C_ID_S16_R25/TASK/seed0"
run_one MAGNITUDE \
  "$REPO/experiments/phase_c/c_id_s16_r25_magnitude.yml" \
  "$ROOT/runs/30_channel/C_ID_S16_R25/MAGNITUDE/seed0"
run_one RANDOM \
  "$REPO/experiments/phase_c/c_id_s16_r25_random.yml" \
  "$ROOT/runs/30_channel/C_ID_S16_R25/RANDOM/index_seed0/train_seed0"

echo "[$(date '+%F %T')] ALL COMPLETE" | tee -a "$QUEUE_LOG"
