"""启动零SAM梯度对照，复用既有初始化、快照和守护流程。"""
import json
from pathlib import Path
import launch_s_sgc2_c_plus_m as base

base.REPORT = base.ROOT / 'reports/113_sgc0_half_control'
base.RUN = base.ROOT / 'outputs/C_PLUS_M_SD22_HALF_SGC0_B8A4_20E_TESTDEV/seed0'
base.CONFIG = base.REPO / 'experiments/phase_s/c_plus_m_sd22_half_sgc0_b8a4_20e.yml'
base.EXTRA_SOURCES = [Path(__file__), base.REPO/'tools/preflight_sgc0_half.py']

if __name__ == '__main__':
    try:
        report = json.loads((base.REPORT/'preflight.json').read_text(encoding='utf-8'))
        assert report['zero_sam_gradient'] and report['same_initial_weights']
        assert report['only_sgc_weight_changed'] and report['sgc_aux_weight'] == 0.0
        base.main()
    except Exception as error:
        base.write_status(status='failed',error=str(error))
        raise
