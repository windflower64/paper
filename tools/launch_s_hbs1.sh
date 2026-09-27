#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/rgbt_experiments/D-FINE
run=/root/autodl-tmp/rgbt_experiments/runs/20_spatial_importance/S_HBS1_BG_SMOOTH_AUX_P16_P32/seed0
python_bin=/root/autodl-tmp/envs/dfine/bin/python
config=experiments/phase_s/s_hbs1_bg_smooth_aux_p16_p32.yml
report=/root/autodl-tmp/rgbt_experiments/reports/20_spatial_importance/S4_HBS_PREFLIGHT

cd "$repo"
if pgrep -af "train.py.*s_hbs1_bg_smooth_aux_p16_p32.yml" >/dev/null; then
  echo "S_HBS1 is already running"
  pgrep -af "train.py.*s_hbs1_bg_smooth_aux_p16_p32.yml"
  exit 0
fi

mkdir -p "$run/artifacts"
cp "$config" "$run/artifacts/"
cp experiments/phase_s/visible_60e_base.yml "$run/artifacts/"
cp src/zoo/dfine/dfine.py "$run/artifacts/"
cp src/zoo/dfine/dfine_criterion.py "$run/artifacts/"
cp tools/preflight_s_hbs1.py "$run/artifacts/"
cp "$report/preflight.json" "$run/artifacts/strict_preflight.json"
cp "$report/source_formula_preflight_foreground_spill.json" "$run/artifacts/"
sha256sum "$run"/artifacts/* > "$run/artifacts/SHA256SUMS"

nohup env PYTHONUNBUFFERED=1 "$python_bin" train.py \
  -c "$config" \
  --seed 0 \
  --use-amp \
  > "$run/train_console.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$run/train.pid"
echo "launched pid=$pid log=$run/train_console.log"

