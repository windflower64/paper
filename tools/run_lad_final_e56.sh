#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/root/autodl-tmp/rgbt_experiments
TASK_REPO="$TASK_ROOT/D-FINE"
TASK_PYTHON=/root/autodl-tmp/envs/dfine/bin/python
TASK_LAD_BASE="$TASK_ROOT/runs/20_spatial_importance/S_LAD1_S8_S16/seed0"
TASK_RUN="$TASK_LAD_BASE/final_e56_validation"
TASK_LAD_CKPT="$TASK_RUN/best_epoch56_snapshot.pth"
TASK_A00_CKPT="$TASK_ROOT/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth"
TASK_LAD_CONFIG="$TASK_REPO/experiments/phase_s/s_lad1_s8_s16.yml"
TASK_A00_CONFIG="$TASK_REPO/experiments/phase_s/visible_60e_base.yml"

mkdir -p "$TASK_RUN/eval" "$TASK_RUN/gradient" "$TASK_RUN/latency"
cp -f "$TASK_LAD_BASE/best_stg1.pth" "$TASK_LAD_CKPT"

for TASK_MODE in learned uniform shuffled; do
  echo "=== final e56 LAD mode=$TASK_MODE start $(date -Is) ==="
  CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" \
    "$TASK_REPO/tools/evaluate_custom_sizes_and_importance.py" \
    --repo "$TASK_REPO" --config "$TASK_LAD_CONFIG" \
    --checkpoint "$TASK_LAD_CKPT" --output-dir "$TASK_RUN/eval" \
    --lad-mode "$TASK_MODE"
done

echo "=== final e56 gradient audit start $(date -Is) ==="
CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" "$TASK_REPO/tools/audit_spatial_gradient_flow.py" \
  --repo "$TASK_REPO" --config "$TASK_LAD_CONFIG" --checkpoint "$TASK_LAD_CKPT" \
  --batch-size 2 --batches 4 --output "$TASK_RUN/gradient/lad_epoch56.json"

echo "=== fixed-protocol latency start $(date -Is) ==="
CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" "$TASK_REPO/tools/benchmark_inference_cost.py" \
  --repo "$TASK_REPO" --config "$TASK_A00_CONFIG" --checkpoint "$TASK_A00_CKPT" \
  --height 512 --width 640 --batch-size 1 --warmup 50 --iterations 200 \
  --precision fp16 --output "$TASK_RUN/latency/a00_fp16_b1.json"
CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" "$TASK_REPO/tools/benchmark_inference_cost.py" \
  --repo "$TASK_REPO" --config "$TASK_LAD_CONFIG" --checkpoint "$TASK_LAD_CKPT" \
  --height 512 --width 640 --batch-size 1 --warmup 50 --iterations 200 \
  --precision fp16 --output "$TASK_RUN/latency/lad_fp16_b1.json"

echo "=== final e56 validation done $(date -Is) ==="
