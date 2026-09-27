#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/root/autodl-tmp/rgbt_experiments
TASK_REPO="$TASK_ROOT/D-FINE"
TASK_PYTHON=/root/autodl-tmp/envs/dfine/bin/python
LAD_BASE="$TASK_ROOT/runs/20_spatial_importance/S_LAD1_S8_S16/seed0"
LAD_RUN="$LAD_BASE/causal_epoch43"
LAD_CKPT="$LAD_RUN/best_epoch43_snapshot.pth"
AUX_BASE="$TASK_ROOT/runs/20_spatial_importance/S_AUX1_M/seed0"
AUX_CKPT="$AUX_BASE/best_stg1.pth"
AUX_CONFIG="$TASK_REPO/experiments/phase_s/s_aux1_m.yml"
LAD_CONFIG="$TASK_REPO/experiments/phase_s/s_lad1_s8_s16.yml"

mkdir -p "$LAD_RUN/eval" "$LAD_RUN/gradient" "$LAD_RUN/budget"

echo "=== causal LAD evaluation start $(date -Is) ==="
for TASK_MODE in learned uniform shuffled; do
  echo "--- LAD mode=$TASK_MODE start $(date -Is) ---"
  CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" \
    "$TASK_REPO/tools/evaluate_custom_sizes_and_importance.py" \
    --repo "$TASK_REPO" \
    --config "$LAD_CONFIG" \
    --checkpoint "$LAD_CKPT" \
    --output-dir "$LAD_RUN/eval" \
    --lad-mode "$TASK_MODE"
  echo "--- LAD mode=$TASK_MODE done $(date -Is) ---"
done

# Gradient audits use backward passes.  Wait until the main 60-epoch training
# process exits so the audit cannot create a transient out-of-memory failure.
TASK_TRAIN_PID=""
if [[ -f "$LAD_BASE/train.pid" ]]; then
  TASK_TRAIN_PID="$(cat "$LAD_BASE/train.pid")"
fi
while [[ -n "$TASK_TRAIN_PID" ]] && kill -0 "$TASK_TRAIN_PID" 2>/dev/null; do
  echo "main training pid=$TASK_TRAIN_PID still active; gradient/budget jobs wait $(date -Is)"
  sleep 30
done

echo "=== gradient audits start $(date -Is) ==="
CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" "$TASK_REPO/tools/audit_spatial_gradient_flow.py" \
  --repo "$TASK_REPO" --config "$LAD_CONFIG" --checkpoint "$LAD_CKPT" \
  --batch-size 2 --batches 4 --output "$LAD_RUN/gradient/lad_epoch43.json"
CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" "$TASK_REPO/tools/audit_spatial_gradient_flow.py" \
  --repo "$TASK_REPO" --config "$AUX_CONFIG" --checkpoint "$AUX_CKPT" \
  --batch-size 2 --batches 4 --output "$LAD_RUN/gradient/s_aux_best.json"

echo "=== S-BUDGET baseline start $(date -Is) ==="
CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" \
  "$TASK_REPO/tools/evaluate_custom_sizes_and_importance.py" \
  --repo "$TASK_REPO" --config "$AUX_CONFIG" --checkpoint "$AUX_CKPT" \
  --output-dir "$LAD_RUN/budget"

for TASK_PCT in 5 10 20 30; do
  for TASK_MODE in learned random; do
    echo "--- S-BUDGET pct=$TASK_PCT mode=$TASK_MODE start $(date -Is) ---"
    CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" \
      "$TASK_REPO/tools/evaluate_custom_sizes_and_importance.py" \
      --repo "$TASK_REPO" --config "$AUX_CONFIG" --checkpoint "$AUX_CKPT" \
      --output-dir "$LAD_RUN/budget" \
      --spatial-budget-pct "$TASK_PCT" --spatial-budget-mode "$TASK_MODE"
  done
done

"$TASK_PYTHON" "$TASK_REPO/tools/summarize_spatial_diagnostics.py" \
  --run-dir "$LAD_RUN" --output "$LAD_RUN/SPATIAL_DIAGNOSTICS_EPOCH43_ZH.md"
echo "=== all spatial diagnostics done $(date -Is) ==="
