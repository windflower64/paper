"""构造M-A1严格起点：成熟C+D可见模型、成熟红外流和零输出软对齐读取器。"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def checkpoint_state(checkpoint, source_name):
    if isinstance(checkpoint.get("ema"), dict) and isinstance(
        checkpoint["ema"].get("module"), dict
    ):
        return checkpoint["ema"]["module"], "ema.module"
    if isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"], "model"
    raise RuntimeError(f"{source_name}缺少ema.module和model状态")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visible-d", type=Path, required=True)
    parser.add_argument("--thermal-detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    visible_checkpoint = torch.load(
        args.visible_d, map_location="cpu", weights_only=False
    )
    thermal_checkpoint = torch.load(
        args.thermal_detector, map_location="cpu", weights_only=False
    )
    visible_state, visible_field = checkpoint_state(
        visible_checkpoint, "C+D可见模型"
    )
    thermal_state, thermal_field = checkpoint_state(
        thermal_checkpoint, "红外单模态模型"
    )

    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    expected = model.state_dict()
    hybrid = OrderedDict(
        (key, value.detach().clone()) for key, value in expected.items()
    )

    copied_visible = []
    for key, value in visible_state.items():
        if key.startswith(("thermal_backbone.", "thermal_encoder.")):
            continue
        if key.startswith("decoder.sdtec_"):
            continue
        if key in hybrid and hybrid[key].shape == value.shape:
            hybrid[key] = value.detach().clone()
            copied_visible.append(key)

    copied_thermal = []
    for source_prefix, target_prefix in (
        ("backbone.", "thermal_backbone."),
        ("encoder.", "thermal_encoder."),
    ):
        for source_key, value in thermal_state.items():
            if not source_key.startswith(source_prefix):
                continue
            target_key = target_prefix + source_key[len(source_prefix) :]
            if target_key not in hybrid or hybrid[target_key].shape != value.shape:
                raise RuntimeError(f"红外流张量不兼容：{source_key}")
            hybrid[target_key] = value.detach().clone()
            copied_thermal.append((source_key, target_key))

    if not copied_visible or not copied_thermal:
        raise RuntimeError("M-A1初始化没有复制到完整的成熟模态流")
    final_key = "decoder.sdtec_aligned_calibrator.delta_head.2.weight"
    affine_delta_key = "decoder.sdtec_aligned_calibrator.affine_delta"
    if torch.count_nonzero(hybrid[final_key]):
        raise RuntimeError("M-A1最终校准投影不是严格零初始化")
    if torch.count_nonzero(hybrid[affine_delta_key]):
        raise RuntimeError("M-A1仿射残差不是严格零初始化")

    configured_affine = torch.as_tensor(
        cfg.yaml_cfg["DFINETransformer"]["sdtec_alignment_affine_init"],
        dtype=hybrid["decoder.sdtec_aligned_calibrator.affine_base"].dtype,
    ).reshape(3, 2)
    if not torch.equal(
        hybrid["decoder.sdtec_aligned_calibrator.affine_base"],
        configured_affine,
    ):
        raise RuntimeError("M-A1仿射先验没有按配置写入模型")

    model.load_state_dict(hybrid, strict=True)
    output_checkpoint = {
        "model": hybrid,
        "ema": {"module": OrderedDict(hybrid), "updates": 0},
        "ma1_p1_init": {
            "seed": args.seed,
            "visible_d_path": str(args.visible_d.resolve()),
            "visible_d_field": visible_field,
            "visible_d_sha256": sha256(args.visible_d),
            "thermal_detector_path": str(args.thermal_detector.resolve()),
            "thermal_detector_field": thermal_field,
            "thermal_detector_sha256": sha256(args.thermal_detector),
            "visible_keys_copied": len(copied_visible),
            "thermal_stream_keys_copied": len(copied_thermal),
            "affine_visible_to_thermal": configured_affine.tolist(),
            "strict_model_key_count": len(expected),
            "functional_start": "M-A1输出层为零，严格等价于成熟C+D模型",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    print(f"output={args.output}")
    print(f"visible_keys_copied={len(copied_visible)}")
    print(f"thermal_stream_keys_copied={len(copied_thermal)}")
    print(f"strict_model_key_count={len(expected)}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
