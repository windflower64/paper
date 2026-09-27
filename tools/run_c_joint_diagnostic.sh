#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/rgbt_experiments
PY=/root/autodl-tmp/envs/dfine/bin/python
SCRIPT="$ROOT/diagnostics/diagnose_c_channel_joint.py"
OUT="$ROOT/reports/30_channel/C_D1_D2_D3_JOINT"
LOG="$OUT/run.log"
PIDFILE="$OUT/run.pid"
PATTERN='diagnose_c_channel_joint.py.*C_D1_D2_D3_JOINT'

mkdir -p "$OUT"

if pgrep -af "$PATTERN" >/dev/null 2>&1; then
  echo "C-D1/D2/D3 joint diagnostic is already running:"
  pgrep -af "$PATTERN"
  exit 1
fi

nohup "$PY" -u "$SCRIPT" \
  --repo "$ROOT/D-FINE" \
  --data-root "$ROOT/data/antiuav6k_common" \
  --config "$ROOT/D-FINE/experiments/dfine/dfine_n_visible_640x512_full.yml" \
  --checkpoint "$ROOT/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth" \
  --output-dir "$OUT" \
  --attribution-samples 512 \
  --component-samples 128 \
  --intervention-samples 192 \
  --intervention-batch-size 8 \
  --bootstrap-repeats 64 \
  --seed 20260810 \
  >"$LOG" 2>&1 &

pid=$!
echo "$pid" >"$PIDFILE"
echo "Started PID=$pid"
echo "Log: $LOG"
