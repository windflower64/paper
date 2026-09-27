#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/rgbt_experiments/D-FINE
python_bin=/root/autodl-tmp/envs/dfine/bin/python
config="$repo/experiments/phase_s/s_btrd1_act_trans_s8_s16.yml"
checkpoint=/root/autodl-tmp/rgbt_experiments/runs/20_spatial_importance/S_BTRD1_ACT_TRANS_S8_S16/seed0/best_stg1.pth
output=/root/autodl-tmp/rgbt_experiments/runs/20_spatial_importance/S_BTRD1_ACT_TRANS_S8_S16/seed0/final_fixed_best_validation

cd "$repo"
for mode in shuffled constant; do
  "$python_bin" evaluate_custom_sizes_and_importance.py \
    --repo "$repo" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --weight-source ema \
    --skip-importance \
    --btrd-mode "$mode" \
    > "$output/btrd_${mode}.log" 2>&1
done

echo completed > "$output/btrd_controls.done"

