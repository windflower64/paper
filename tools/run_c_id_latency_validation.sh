#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/rgbt_experiments
REPO="$ROOT/D-FINE"
PY=/root/autodl-tmp/envs/dfine/bin/python
BENCH="$REPO/benchmark_inference_cost.py"
OUT="$ROOT/reports/30_channel/C_ID_S16_R25_VALIDATION/latency"
mkdir -p "$OUT"

for precision in fp16 fp32; do
  "$PY" "$BENCH" --repo "$REPO" \
    --config "$REPO/experiments/dfine/dfine_n_visible_640x512_full.yml" \
    --checkpoint "$ROOT/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth" \
    --precision "$precision" --batch-size 1 --warmup 50 --iterations 300 \
    --output "$OUT/A00_${precision}_b1.json" >"$OUT/A00_${precision}_b1.log" 2>&1
  "$PY" "$BENCH" --repo "$REPO" \
    --config "$REPO/experiments/phase_c/c_id_s16_r25_task.yml" \
    --checkpoint "$ROOT/runs/30_channel/C_ID_S16_R25/TASK/seed0/best_stg1.pth" \
    --precision "$precision" --batch-size 1 --warmup 50 --iterations 300 \
    --output "$OUT/TASK_${precision}_b1.json" >"$OUT/TASK_${precision}_b1.log" 2>&1
done

echo "LATENCY COMPLETE"
