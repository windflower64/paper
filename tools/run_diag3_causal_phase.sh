#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/rgbt_experiments
DFINE_ENV=/root/autodl-tmp/envs/dfine
MASKS="$ROOT/reports/20_spatial_importance/S_DIAG3_TRUE_CONTOUR/masks_val"
OUT="$ROOT/reports/20_spatial_importance/S_DIAG3_TRUE_CONTOUR/causal_val"
LOG="$ROOT/logs/20_spatial_importance/S_DIAG3_TRUE_CONTOUR_CAUSAL.log"

if [[ ! -f "$MASKS/manual_visual_audit.pass" ]]; then
  echo "Manual visual audit has not been approved; refusing to run detector causality." >&2
  exit 4
fi
if [[ ! -e /dev/nvidia0 ]]; then
  echo "No GPU is attached." >&2
  exit 3
fi

mkdir -p "$OUT" "$(dirname "$LOG")"
cd "$ROOT/D-FINE"
"$DFINE_ENV/bin/python" tools/diagnose_true_contour_causality.py \
  --repo "$ROOT/D-FINE" \
  --config "$ROOT/D-FINE/experiments/phase_s/visible_60e_base.yml" \
  --checkpoint "$ROOT/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth" \
  --contour-records "$MASKS/records.json" \
  --output-dir "$OUT" \
  --batch-size 2 \
  2>&1 | tee "$LOG"

touch "$OUT/causal_phase.complete"
