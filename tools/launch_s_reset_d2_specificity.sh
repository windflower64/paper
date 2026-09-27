#!/usr/bin/env bash
set -euo pipefail

transition="${1:?transition is required}"
case "$transition" in
  input_to_s2|s2_to_s4|s4_to_s8|s8_to_s16|s16_to_s32) ;;
  *) echo "unsupported transition: $transition" >&2; exit 2 ;;
esac

cd /root/autodl-tmp/rgbt_experiments/D-FINE
out="/root/autodl-tmp/rgbt_experiments/reports/20_spatial_importance/S_RESET_D2_SPECIFICITY/${transition}"
mkdir -p "$out/artifacts"
cp tools/diagnose_s0_multistage_boundary_causality.py "$out/artifacts/"
cp tools/diagnose_s0_boundary_specificity.py "$out/artifacts/"
cp experiments/phase_s/visible_60e_base.yml "$out/artifacts/"

nohup /root/autodl-tmp/envs/dfine/bin/python \
  tools/diagnose_s0_boundary_specificity.py \
  --repo . \
  --config experiments/phase_s/visible_60e_base.yml \
  --checkpoint /root/autodl-tmp/rgbt_experiments/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth \
  --output-dir "$out" \
  --transition "$transition" \
  --batch-size 2 \
  > "$out/console.log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$out/pid"
echo "PID=$pid TRANSITION=$transition"
