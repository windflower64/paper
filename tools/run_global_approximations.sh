#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/rgbt_experiments
PY=/root/autodl-tmp/envs/dfine/bin/python
REPO="$ROOT/D-FINE"
REPORT="$ROOT/reports/30_channel/C_PAT_SF1_VALIDATION"
CKPT="$ROOT/runs/30_channel/C_PAT_SF1_S32_R4/seed0/best_stg1.pth"

cd "$REPO"

for MODE in global_query value_mean; do
  "$PY" tools/evaluate_pat_sf_intervention.py \
    --repo "$REPO" \
    --config "$REPO/experiments/phase_c/c_pat_sf_s32_r4.yml" \
    --checkpoint "$CKPT" \
    --mode "$MODE" \
    --output-dir "$REPORT/${MODE}" \
    2>&1 | tee "$REPORT/${MODE}.log"
done
