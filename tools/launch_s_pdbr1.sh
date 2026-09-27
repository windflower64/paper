#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/rgbt_experiments/D-FINE
run=/root/autodl-tmp/rgbt_experiments/runs/20_spatial_importance/S_PDBR1_POSDIR_BOUNDARY_S8_S16/seed0
python_bin=/root/autodl-tmp/envs/dfine/bin/python
config=experiments/phase_s/s_pdbr1_posdir_boundary_s8_s16.yml

cd "$repo"
if pgrep -af "train.py.*s_pdbr1_posdir_boundary_s8_s16.yml" >/dev/null; then
  echo "S_PDBR1 is already running"
  pgrep -af "train.py.*s_pdbr1_posdir_boundary_s8_s16.yml"
  exit 0
fi

mkdir -p "$run/artifacts"
cp "$config" "$run/artifacts/"
cp experiments/phase_s/visible_60e_base.yml "$run/artifacts/"
cp src/nn/backbone/hgnetv2.py "$run/artifacts/"
cp src/zoo/dfine/dfine.py "$run/artifacts/"
cp src/zoo/dfine/dfine_criterion.py "$run/artifacts/"
cp evaluate_custom_sizes_and_importance.py "$run/artifacts/"
cp tools/preflight_s_pdbr1.py "$run/artifacts/"
cp tools/smoke_s_pdbr1_detector.py "$run/artifacts/"
cp tools/run_s_pdbr1_controls.sh "$run/artifacts/"
cp /root/autodl-tmp/rgbt_experiments/reports/20_spatial_importance/S5_PDBR1_PREFLIGHT/preflight.json "$run/artifacts/"
sha256sum "$run"/artifacts/* > "$run/artifacts/SHA256SUMS"

nohup env PYTHONUNBUFFERED=1 "$python_bin" train.py \
  -c "$config" \
  --seed 0 \
  --use-amp \
  > "$run/train_console.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$run/train.pid"
echo "launched pid=$pid log=$run/train_console.log"
