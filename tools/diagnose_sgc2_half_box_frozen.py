"""Frozen BOX EMA M on/off target diagnostics, using historical sources."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
FROZEN = ROOT / 'reports/148_sgc2_half_box/frozen_repo'
sys.path.insert(0, str(FROZEN))
import src.core
sys.path.insert(0, str(REPO))
from tools.validate_sgc2_joint_strength import main


if __name__ == '__main__':
    run = ROOT / 'outputs/C_PLUS_M_SD22_HALF_SGC2_BOX_DECAY9_14_B8A4_20E_TESTDEV/seed0'
    main(config=FROZEN / 'experiments/phase_s/s_sgc2_box_c_plus_m_sd22_half_b8a4_20e.yml',
         run=run, out=ROOT / 'reports/148_sgc2_half_box/diagnostics', epochs=(7,), strengths=(0.,1.),
         checkpoint_paths={7: run / 'best_stg1.pth'})
