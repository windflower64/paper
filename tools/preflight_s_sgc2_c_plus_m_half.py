"""固定最终残差0.5：单变量配置、相同初始化和真实批次预检。"""
import copy
import hashlib
import json
import torch
import preflight_s_sgc2_c_plus_m as base

CONFIG = base.REPO / 'experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'
REPORT = base.ROOT / 'reports/112_sgc2_rgbt_half/preflight.json'


def main():
    old_cfg = base.YAMLConfig(str(base.CONFIG))
    new_cfg = base.YAMLConfig(str(CONFIG))
    old = copy.deepcopy(old_cfg.yaml_cfg)
    new = copy.deepcopy(new_cfg.yaml_cfg)
    assert new['DFINE'].pop('rgbt_sd2_final_residual_scale') == 0.5
    for cfg in (old, new):
        cfg.pop('__include__', None)
        cfg.pop('output_dir')
    assert old == new, 'Unexpected experiment changes beyond final M scale'
    reference_manifest = base.ROOT / 'outputs/C_PLUS_M_SD22_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0'
    manifest = json.loads((reference_manifest / 'artifacts/manifest.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(base.CHECKPOINT.read_bytes()).hexdigest()
    assert digest == manifest[str(base.CHECKPOINT)]
    protected = [name for name in manifest if '/data/' in name.replace('\\', '/')
                 or '/reports/104_sam3_role_control/' in name.replace('\\', '/')]
    assert protected
    for name in protected:
        assert hashlib.sha256(base.Path(name).read_bytes()).hexdigest() == manifest[name], name
    initial = []
    for cfg, scale in ((old_cfg, 1.0), (new_cfg, 0.5)):
        torch.manual_seed(0)
        model = cfg.model
        shim = base.BaseSolver.__new__(base.BaseSolver)
        shim.model = model
        shim.load_tuning_state(str(base.CHECKPOINT))
        assert model.sd2_conditioner.final_residual_scale == scale
        initial.append(base.model_digest(model))
    assert initial[0] == initial[1], 'Initial weights must be identical'
    del model, shim, old_cfg, new_cfg
    base.CONFIG = CONFIG
    base.REPORT = REPORT
    base.main()
    result = json.loads(REPORT.read_text(encoding='utf-8'))
    result.update(final_residual_scale=0.5, config_only_scale_changed=True,
                  same_initial_weights=True, pre_forward_initial_sha256=initial[0],
                  unchanged_data_and_mask_files=len(protected),
                  original_init_checkpoint_hash_matches=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('HALF_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
