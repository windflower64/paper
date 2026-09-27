#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/rgbt_experiments/D-FINE
out=/root/autodl-tmp/rgbt_experiments/reports/20_spatial_importance/S_RESET_D1_CAUSALITY/full_val
mkdir -p "$out/artifacts"
cp tools/diagnose_s0_multistage_boundary_causality.py "$out/artifacts/"
cp experiments/phase_s/visible_60e_base.yml "$out/artifacts/"

nohup /root/autodl-tmp/envs/dfine/bin/python \
  tools/diagnose_s0_multistage_boundary_causality.py \
  --repo . \
  --config experiments/phase_s/visible_60e_base.yml \
  --checkpoint /root/autodl-tmp/rgbt_experiments/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth \
  --output-dir "$out" \
  --batch-size 2 \
  > "$out/console.log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$out/pid"
echo "PID=$pid"
