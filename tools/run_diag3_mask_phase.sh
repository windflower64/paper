#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/rgbt_experiments
SAM_ENV=/root/autodl-tmp/envs/sam2_diag3
SAM_REPO="$ROOT/_third_party/sam2"
OUT="$ROOT/reports/20_spatial_importance/S_DIAG3_TRUE_CONTOUR/masks_val"
LOG="$ROOT/logs/20_spatial_importance/S_DIAG3_TRUE_CONTOUR_MASKS.log"

if [[ ! -f "$ROOT/weights/sam2/diag3_setup.done" ]]; then
  echo "SAM2 setup is incomplete" >&2
  exit 2
fi
if [[ ! -e /dev/nvidia0 ]]; then
  echo "No GPU is attached. Enable the AutoDL GPU instance before running this phase." >&2
  exit 3
fi

mkdir -p "$OUT" "$(dirname "$LOG")"
cd "$ROOT/D-FINE"

"$SAM_ENV/bin/python" tools/generate_sam2_contours.py \
  --annotations "$ROOT/data/antiuav6k/annotations/instances_visible_val.json" \
  --image-root "$ROOT/data/antiuav6k_common/images/val" \
  --sam2-repo "$SAM_REPO" \
  --checkpoint "$ROOT/weights/sam2/sam2.1_hiera_b+.pt" \
  --output-dir "$OUT" \
  ${MAX_IMAGES:+--max-images "$MAX_IMAGES"} \
  2>&1 | tee "$LOG"

"$SAM_ENV/bin/python" tools/make_contour_contact_sheets.py \
  --records "$OUT/records.json" \
  --overlay-dir "$OUT/audit_overlays" \
  --output-dir "$OUT/contact_sheets"

"$SAM_ENV/bin/python" tools/audit_contour_projection.py \
  --records "$OUT/records.json" \
  --causality-script-dir "$ROOT/D-FINE/tools" \
  --output "$OUT/projection_s8_summary.json"

touch "$OUT/mask_phase.complete"
echo "Mask phase complete. Manual contact-sheet review is required before causal evaluation."
