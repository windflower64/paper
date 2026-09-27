#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/rgbt_experiments/D-FINE
python_bin=/root/autodl-tmp/envs/dfine/bin/python
config="$repo/experiments/phase_s/s_pdbr1_posdir_boundary_s8_s16.yml"
run=/root/autodl-tmp/rgbt_experiments/runs/20_spatial_importance/S_PDBR1_POSDIR_BOUNDARY_S8_S16/seed0
checkpoint="$run/best_stg1.pth"
output="$run/final_causal_controls"

cd "$repo"
mkdir -p "$output"

for spec in \
  learned:directional \
  shifted:directional \
  crossed:directional \
  one:directional \
  zero:directional \
  learned:isotropic
do
  mask="${spec%%:*}"
  detail="${spec##*:}"
  "$python_bin" evaluate_custom_sizes_and_importance.py \
    --repo "$repo" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --weight-source ema \
    --skip-importance \
    --pdbr-mask-mode "$mask" \
    --pdbr-detail-mode "$detail" \
    > "$output/${mask}_${detail}.log" 2>&1
done

echo completed > "$output/pdbr_controls.done"
