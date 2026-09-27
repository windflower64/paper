#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/rgbt_experiments/D-FINE
PY=/root/autodl-tmp/envs/dfine/bin/python
CONFIG=experiments/phase_c/c_pat_gm_s32_r4.yml
RUN=/root/autodl-tmp/rgbt_experiments/runs/30_channel/C_PAT_GM1_S32_R4/seed0

timeout 10s nvidia-smi -L 2>/dev/null | grep -q '^GPU '
timeout 15s "$PY" -c 'import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)'
if [[ -f "$RUN/train.pid" ]] && kill -0 "$(cat "$RUN/train.pid")" 2>/dev/null; then
  echo "C-PAT-GM1 already running: pid=$(cat "$RUN/train.pid")" >&2
  exit 3
fi

mkdir -p "$RUN/artifacts" "$RUN/preflight"
cd "$REPO"
cp experiments/phase_s/visible_60e_base.yml "$RUN/artifacts/"
cp "$CONFIG" "$RUN/artifacts/"
cp src/nn/backbone/hgnetv2.py "$RUN/artifacts/"
cp src/nn/backbone/partialnet_pat_sf.py "$RUN/artifacts/"
cp tools/smoke_c_pat_gm.py "$RUN/artifacts/"
sha256sum "$RUN"/artifacts/* > "$RUN/artifacts/SHA256SUMS.txt"

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" tools/smoke_c_pat_gm.py \
  > "$RUN/preflight/smoke.log" 2>&1

nohup setsid "$PY" train.py \
  -c "$CONFIG" \
  -t /root/autodl-tmp/rgbt_experiments/weights/dfine_n_coco.pth \
  --seed 0 --use-amp \
  </dev/null > "$RUN/train_console.log" 2>&1 &
echo $! > "$RUN/train.pid"
echo "C_PAT_GM1_STARTED pid=$(cat "$RUN/train.pid")"
