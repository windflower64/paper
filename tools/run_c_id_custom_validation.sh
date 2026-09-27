#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/rgbt_experiments
REPO="$ROOT/D-FINE"
PY=/root/autodl-tmp/envs/dfine/bin/python
EVAL="$REPO/evaluate_custom_sizes_and_importance.py"
BASE="$ROOT/reports/30_channel/C_ID_S16_R25_VALIDATION"

run_one() {
  local name=$1
  local cfg=$2
  local ckpt=$3
  local out="$BASE/$name"
  mkdir -p "$out"
  echo "START $name"
  "$PY" -u "$EVAL" \
    --repo "$REPO" --config "$cfg" --checkpoint "$ckpt" \
    --output-dir "$out" --weight-source ema --skip-importance \
    >"$out/run.log" 2>&1
  echo "DONE $name"
}

run_one A00 \
  "$REPO/experiments/dfine/dfine_n_visible_640x512_full.yml" \
  "$ROOT/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth"
run_one TASK \
  "$REPO/experiments/phase_c/c_id_s16_r25_task.yml" \
  "$ROOT/runs/30_channel/C_ID_S16_R25/TASK/seed0/best_stg1.pth"
run_one MAGNITUDE \
  "$REPO/experiments/phase_c/c_id_s16_r25_magnitude.yml" \
  "$ROOT/runs/30_channel/C_ID_S16_R25/MAGNITUDE/seed0/best_stg1.pth"
run_one RANDOM \
  "$REPO/experiments/phase_c/c_id_s16_r25_random.yml" \
  "$ROOT/runs/30_channel/C_ID_S16_R25/RANDOM/index_seed0/train_seed0/best_stg1.pth"

echo "ALL COMPLETE"
