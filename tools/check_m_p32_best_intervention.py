"""Evaluate best P32 weights with their learned P32 projection disabled."""
import json
import torch
import train_m_p32 as job
from preflight_sqer1 import seed_all, sha256_file


def main():
    seed_all(0)
    torch.set_num_threads(4)
    cfg, source, refiner, _ = job.setup(16, 'P32')
    checkpoint = job.RUN/'P32/best.pth'
    saved = torch.load(checkpoint, map_location='cpu')
    source.model.load_state_dict(saved['base'], strict=True)
    refiner.load_state_dict(saved['refiner'], strict=True)
    source.model.eval()
    refiner.eval()
    reference = saved['last_row']['validation']['coco_eval_bbox']
    handle = source.model.mote_fusion.object_token_p32.register_forward_hook(
        lambda module, inputs, output: torch.zeros_like(output))
    try:
        disabled = job.joint.task.evaluate(cfg, source, refiner)
    finally:
        handle.remove()
        source.close()
    result = {'checkpoint_sha256': sha256_file(checkpoint),
              'reference_bbox_stats': reference,
              'p32_projection_disabled_bbox_stats': disabled['coco_eval_bbox'],
              'delta_pp': [(a-b)*100 for a,b in zip(reference, disabled['coco_eval_bbox'])],
              'images': disabled['images'],
              'interpretation_limit': 'inference intervention on trained P32 model; not a retraining ablation'}
    out = job.joint.task.engine.ROOT/'reports/M_P32_20261005'
    out.mkdir(parents=True, exist_ok=True)
    job.joint.task.engine.write_json(out/'best_p32_intervention.json',result)
    print(json.dumps({'AP_enabled':reference[0], 'AP_disabled':disabled['coco_eval_bbox'][0],
                      'delta_pp':result['delta_pp'][:5]},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
