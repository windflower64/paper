"""复用原训练命令与守护器；独立输出，禁止覆盖原实验。"""
import json
from pathlib import Path
import launch_s_sgc2_c_plus_m as base

base.REPORT = base.ROOT / 'reports/112_sgc2_rgbt_half'
base.RUN = base.ROOT / 'outputs/C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0'
base.CONFIG = base.REPO / 'experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'
base.EXTRA_SOURCES = [Path(__file__), base.REPO / 'tools/preflight_s_sgc2_c_plus_m_half.py',
                      base.REPO / 'tests/test_sd22_final_residual_scale.py']


if __name__ == '__main__':
    try:
        report = json.loads((base.REPORT / 'preflight.json').read_text(encoding='utf-8'))
        assert report['final_residual_scale'] == 0.5
        assert report['same_initial_weights'] and report['config_only_scale_changed']
        base.main()
    except Exception as error:
        base.write_status(status='failed', error=str(error))
        raise
