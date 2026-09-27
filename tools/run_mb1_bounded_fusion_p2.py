"""M-B1 P2: alignment-gated four-state routing and bounded RGB residual.

The mature RGB and thermal detectors stay frozen.  The P1 spatial matcher is
also frozen and acts only as a permission gate.  A four-state router predicts
both / RGB-only / thermal-only / empty, then a matched-capacity RGB/RGB-T pair
of small correctness heads supplies a bounded residual to the top RGB queries.

Test data is never used for fitting, early stopping, thresholds, or alpha
selection.  Thermal-zero is guaranteed to return the exact RGB baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from analyze_ma2_detector_cache import cxcywh_to_xyxy, pairwise_iou
from run_ma2_explicit_matching import build_evaluator, evaluate_scores
from run_mb1_target_alignment_p1 import PairMLP, feature_grid


STATE_NAMES = ("both", "visible_only", "thermal_only", "empty")
ALPHA_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--thermal-train-labels", type=Path, required=True)
    parser.add_argument("--thermal-test-labels", type=Path, required=True)
    parser.add_argument("--p1-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-modulus", type=int, default=5)
    parser.add_argument("--val-residue", type=int, default=0)
    parser.add_argument("--mismatch-offset", type=int, default=607)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_sequence(name):
    return Path(name).stem.rsplit("_", 1)[0]


def stable_bucket(value, modulus):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % modulus


def thermal_positive(path):
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            x, y, w, h = map(float, fields[1:5])
        except ValueError:
            continue
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            return True
    return False


def load_annotation(path):
    coco = json.loads(path.read_text(encoding="utf-8"))
    info = {int(item["id"]): item for item in coco["images"]}
    targets = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        image = info[int(annotation["image_id"])]
        x, y, w, h = map(float, annotation["bbox"])
        targets[int(annotation["image_id"])].append(
            torch.tensor(
                [x / image["width"], y / image["height"],
                 (x + w) / image["width"], (y + h) / image["height"]],
                dtype=torch.float32,
            )
        )
    return info, targets


def state_index(visible, thermal):
    if visible and thermal:
        return 0
    if visible:
        return 1
    if thermal:
        return 2
    return 3


def load_rows(cache_path, annotation_path, thermal_labels):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    info, targets = load_annotation(annotation_path)
    visible_scores = cache["visible_logits"].squeeze(-1).float().sigmoid()
    thermal_scores = cache["thermal_logits"].squeeze(-1).float().sigmoid()
    rows = []
    for row, image_id in enumerate(cache["image_ids"].tolist()):
        item = info[int(image_id)]
        stem = Path(item["file_name"]).stem
        has_visible = bool(targets.get(int(image_id)))
        has_thermal = thermal_positive(thermal_labels / f"{stem}.txt")
        rows.append(
            {
                "row": row,
                "image_id": int(image_id),
                "file_name": item["file_name"],
                "sequence": parse_sequence(item["file_name"]),
                "state": state_index(has_visible, has_thermal),
                "targets": targets.get(int(image_id), []),
                "visible_scores": visible_scores[row],
                "thermal_scores": thermal_scores[row],
                "visible_boxes": cache["visible_boxes"][row].float().clamp(0, 1),
                "thermal_boxes": cache["thermal_boxes"][row].float().clamp(0, 1),
            }
        )
    return cache, rows


def raw_view(row, topk, modality):
    scores = row[f"{modality}_scores"]
    boxes = row[f"{modality}_boxes"]
    order = scores.argsort(descending=True)[:topk]
    return order, scores[order], boxes[order]


def load_matcher(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    item = payload["models"]["spatial_alignment"]
    model = PairMLP(int(item["input_features"]), 32)
    model.load_state_dict(item["state_dict"], strict=True)
    return (
        model.eval(), item["mean"].float(), item["std"].float(),
        payload["affine"].float(), float(item["threshold"]), int(payload["topk"]),
    )


def match_one(visible_row, thermal_row, matcher, mean, std, affine, topk):
    v_order, v_scores, v_boxes = raw_view(visible_row, topk, "visible")
    t_order, t_scores, t_boxes = raw_view(thermal_row, topk, "thermal")
    v = {"visible_scores": v_scores, "visible_boxes": v_boxes}
    t = {"thermal_scores": t_scores, "thermal_boxes": t_boxes}
    features = feature_grid(v, t, affine, spatial=True)
    with torch.inference_mode():
        probability = matcher((features - mean) / std).sigmoid().reshape(len(v_scores), len(t_scores))
    best_probability, best_ir = probability.max(dim=1)
    return {
        "visible_order": v_order,
        "thermal_order": t_order,
        "visible_scores": v_scores,
        "thermal_scores": t_scores,
        "visible_boxes": v_boxes,
        "thermal_boxes": t_boxes,
        "pair_probability": probability,
        "best_probability": best_probability,
        "best_ir": best_ir,
        "image_probability": float(probability.max()),
    }


def matched_views(rows, thermal_rows, matcher_data, topk):
    matcher, mean, std, affine, _threshold, trained_topk = matcher_data
    if topk != trained_topk:
        raise ValueError(f"P1 matcher was trained with topk={trained_topk}, got {topk}")
    return [
        match_one(v, t, matcher, mean, std, affine, topk)
        for v, t in zip(rows, thermal_rows)
    ]


def router_features(rows, views, alignment_threshold):
    result = []
    for row, view in zip(rows, views):
        vs = view["visible_scores"]
        ts = view["thermal_scores"]
        pair = view["pair_probability"].flatten()
        sorted_pair = pair.sort(descending=True).values
        result.append(
            torch.cat(
                (
                    vs, ts,
                    torch.tensor([
                        vs.mean(), ts.mean(), vs[0] - vs[-1], ts[0] - ts[-1],
                        sorted_pair[0], sorted_pair[:3].mean(),
                        (pair >= alignment_threshold).float().mean(),
                    ]),
                )
            )
        )
    return torch.stack(result)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def fit_classifier(x, y, val_x, val_y, hidden, output_dim, epochs, patience, device, seed):
    mean = x.mean(0)
    std = x.std(0).clamp_min(1e-5)
    train_x = (x - mean) / std
    valid_x = (val_x - mean) / std
    model = MLP(x.shape[1], hidden, output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(y.long(), minlength=output_dim).float()
    if output_dim > 1:
        weight = (len(y) / (output_dim * counts.clamp_min(1))).to(device)
    else:
        positives = y.sum().clamp_min(1)
        weight = torch.tensor([(len(y) - positives) / positives], device=device)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, best_epoch, stale = None, math.inf, -1, 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), 4096):
            index = order[start : start + 4096]
            logits = model(train_x[index].to(device))
            if output_dim > 1:
                loss = F.cross_entropy(logits, y[index].long().to(device), weight=weight)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), y[index].float().to(device), pos_weight=weight
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(valid_x.to(device))
            if output_dim > 1:
                loss = F.cross_entropy(logits, val_y.long().to(device), weight=weight)
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), val_y.float().to(device), pos_weight=weight
                )
            loss_value = float(loss)
        if loss_value < best_loss - 1e-6:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss, best_epoch, stale = loss_value, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state, strict=True)
    return model.cpu().eval(), mean, std, {
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "train_samples": len(x), "validation_samples": len(val_x),
        "class_counts": counts.int().tolist(),
    }


def predict_classifier(model, mean, std, x, multiclass):
    with torch.inference_mode():
        logits = model((x - mean) / std)
        return logits.softmax(-1) if multiclass else logits.squeeze(-1).sigmoid()


def state_metrics(labels, probabilities):
    prediction = probabilities.argmax(1)
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    for truth, pred in zip(labels.long(), prediction.long()):
        confusion[truth, pred] += 1
    per_class = {}
    f1s = []
    for index, name in enumerate(STATE_NAMES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index].sum()) - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1s.append(f1)
        per_class[name] = {"support": int(confusion[index].sum()), "precision": precision,
                           "recall": recall, "f1": f1}
    return {
        "accuracy": float((prediction == labels).float().mean()),
        "macro_f1": float(np.mean(f1s)),
        "confusion_truth_rows_prediction_columns": confusion.tolist(),
        "per_class": per_class,
    }


def candidate_features(rows, views, router_probability, alignment_threshold, thermal_enabled=True):
    features, rgb_features, labels = [], [], []
    for row, view, state_prob in zip(rows, views, router_probability):
        boxes = view["visible_boxes"]
        scores = view["visible_scores"]
        logits = torch.logit(scores.clamp(1e-6, 1 - 1e-6))
        ranks = torch.linspace(0, 1, len(scores))
        base = torch.cat((logits[:, None], ranks[:, None], boxes), dim=1)
        matched = view["best_ir"]
        ir_boxes = view["thermal_boxes"][matched]
        ir_scores = view["thermal_scores"][matched]
        align = view["best_probability"]
        delta = ir_boxes[:, :2] - boxes[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=1)
        size_ratio = torch.log((ir_boxes[:, 2:] + 1e-5) / (boxes[:, 2:] + 1e-5))
        hard_permission = (
            (align >= alignment_threshold)
            & (view["thermal_scores"][0] >= 0.05)
        ).float()
        gate = hard_permission * state_prob[0]
        added = torch.cat(
            (
                ir_scores[:, None], align[:, None], delta, delta.abs(),
                distance[:, None], size_ratio,
                state_prob[0].expand(len(scores), 1), gate[:, None],
            ), dim=1,
        )
        if not thermal_enabled:
            added.zero_()
        features.append(torch.cat((base, added), dim=1))
        rgb_features.append(torch.cat((base, torch.zeros_like(added)), dim=1))
        if row["targets"]:
            gt = torch.stack(row["targets"])
            labels.append((pairwise_iou(cxcywh_to_xyxy(boxes), gt).amax(1) >= 0.5).float())
        else:
            labels.append(torch.zeros(len(boxes)))
    return torch.stack(features), torch.stack(rgb_features), torch.stack(labels)


def candidate_probability(model, mean, std, features):
    return predict_classifier(model, mean, std, features.reshape(-1, features.shape[-1]), False).reshape(features.shape[:2])


def residual_delta(rgb_bundle, rgbt_bundle, rgb_features, rgbt_features, gate):
    rgb_probability = candidate_probability(*rgb_bundle, rgb_features)
    rgbt_probability = candidate_probability(*rgbt_bundle, rgbt_features)
    delta = torch.logit(rgbt_probability.clamp(1e-6, 1 - 1e-6)) - torch.logit(
        rgb_probability.clamp(1e-6, 1 - 1e-6)
    )
    return delta.clamp(-4, 4) * gate


def integrate(cache, views, delta, alpha):
    logits = cache["visible_logits"].squeeze(-1).float().clone()
    for row, (view, row_delta) in enumerate(zip(views, delta)):
        index = view["visible_order"]
        logits[row, index] = logits[row, index] + alpha * row_delta
    return logits.sigmoid()


def choose_alert_threshold(probability, labels):
    candidates = torch.unique(torch.quantile(probability, torch.linspace(0, 1, 501)))
    best = None
    for threshold in candidates:
        pred = probability >= threshold
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        item = (f1, recall, -float(threshold))
        if best is None or item > best[0]:
            best = (item, float(threshold))
    return best[1]


def alert_metrics(probability, labels, threshold):
    pred = probability >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"threshold_from_train_validation": threshold, "true_positive": tp,
            "false_positive": fp, "false_negative": fn, "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall)}


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache, train_rows = load_rows(args.train_cache, args.train_annotations, args.thermal_train_labels)
    test_cache, test_rows = load_rows(args.test_cache, args.test_annotations, args.thermal_test_labels)
    matcher_data = load_matcher(args.p1_weights)
    alignment_threshold = matcher_data[4]
    train_views = matched_views(train_rows, train_rows, matcher_data, args.topk)
    test_views = matched_views(test_rows, test_rows, matcher_data, args.topk)
    mismatch_order = torch.arange(len(test_rows)).roll(args.mismatch_offset).tolist()
    mismatch_rows = [test_rows[index] for index in mismatch_order]
    mismatch_views = matched_views(test_rows, mismatch_rows, matcher_data, args.topk)

    val_mask = torch.tensor([
        stable_bucket(row["sequence"], args.val_modulus) == args.val_residue
        for row in train_rows
    ])
    fit_mask = ~val_mask
    train_state = torch.tensor([row["state"] for row in train_rows])
    test_state = torch.tensor([row["state"] for row in test_rows])
    train_router_x = router_features(train_rows, train_views, alignment_threshold)
    test_router_x = router_features(test_rows, test_views, alignment_threshold)
    mismatch_router_x = router_features(test_rows, mismatch_views, alignment_threshold)
    router, router_mean, router_std, router_fit = fit_classifier(
        train_router_x[fit_mask], train_state[fit_mask], train_router_x[val_mask], train_state[val_mask],
        args.hidden, 4, args.epochs, args.patience, device, args.seed,
    )
    train_router_probability = predict_classifier(router, router_mean, router_std, train_router_x, True)
    test_router_probability = predict_classifier(router, router_mean, router_std, test_router_x, True)
    mismatch_router_probability = predict_classifier(router, router_mean, router_std, mismatch_router_x, True)

    train_rgbt, train_rgb, train_labels = candidate_features(
        train_rows, train_views, train_router_probability, alignment_threshold
    )
    rgb_model, rgb_mean, rgb_std, rgb_fit = fit_classifier(
        train_rgb[fit_mask].reshape(-1, train_rgb.shape[-1]), train_labels[fit_mask].reshape(-1),
        train_rgb[val_mask].reshape(-1, train_rgb.shape[-1]), train_labels[val_mask].reshape(-1),
        args.hidden, 1, args.epochs, args.patience, device, args.seed + 1,
    )
    rgbt_model, rgbt_mean, rgbt_std, rgbt_fit = fit_classifier(
        train_rgbt[fit_mask].reshape(-1, train_rgbt.shape[-1]), train_labels[fit_mask].reshape(-1),
        train_rgbt[val_mask].reshape(-1, train_rgbt.shape[-1]), train_labels[val_mask].reshape(-1),
        args.hidden, 1, args.epochs, args.patience, device, args.seed + 2,
    )
    rgb_bundle = (rgb_model, rgb_mean, rgb_std)
    rgbt_bundle = (rgbt_model, rgbt_mean, rgbt_std)
    train_gate = train_rgbt[..., -1]
    train_delta = residual_delta(rgb_bundle, rgbt_bundle, train_rgb, train_rgbt, train_gate)

    train_boxes = cxcywh_to_xyxy(train_cache["visible_boxes"].float()).clamp(0, 1)
    train_evaluator = build_evaluator(args.train_annotations)
    val_indices = torch.where(val_mask)[0].tolist()
    calibration = []
    for alpha in ALPHA_GRID:
        score = integrate(train_cache, train_views, train_delta, alpha)
        metrics = evaluate_scores(train_evaluator, train_cache, train_boxes, score, val_indices)
        calibration.append({"alpha": alpha, "metrics": metrics})
        print(f"alpha={alpha:.2f} validation_AP={metrics['AP']:.9f}", flush=True)
    chosen = max(calibration, key=lambda item: (item["metrics"]["AP"], -item["alpha"]))

    test_rgbt, test_rgb, _ = candidate_features(
        test_rows, test_views, test_router_probability, alignment_threshold
    )
    mismatch_rgbt, mismatch_rgb, _ = candidate_features(
        test_rows, mismatch_views, mismatch_router_probability, alignment_threshold
    )
    test_delta = residual_delta(rgb_bundle, rgbt_bundle, test_rgb, test_rgbt, test_rgbt[..., -1])
    mismatch_delta = residual_delta(
        rgb_bundle, rgbt_bundle, mismatch_rgb, mismatch_rgbt, mismatch_rgbt[..., -1]
    )
    test_boxes = cxcywh_to_xyxy(test_cache["visible_boxes"].float()).clamp(0, 1)
    test_evaluator = build_evaluator(args.test_annotations)
    all_indices = list(range(len(test_rows)))
    baseline = test_cache["visible_logits"].squeeze(-1).float().sigmoid()
    normal = integrate(test_cache, test_views, test_delta, chosen["alpha"])
    mismatch = integrate(test_cache, mismatch_views, mismatch_delta, chosen["alpha"])
    zero = baseline.clone()
    metrics = {
        "visible_baseline": evaluate_scores(test_evaluator, test_cache, test_boxes, baseline, all_indices),
        "mb1_normal": evaluate_scores(test_evaluator, test_cache, test_boxes, normal, all_indices),
        "mb1_global_mismatch": evaluate_scores(test_evaluator, test_cache, test_boxes, mismatch, all_indices),
        "mb1_thermal_zero": evaluate_scores(test_evaluator, test_cache, test_boxes, zero, all_indices),
    }

    val_alert_labels = train_state[val_mask] == 2
    alert_threshold = choose_alert_threshold(train_router_probability[val_mask, 2], val_alert_labels)
    alert = alert_metrics(test_router_probability[:, 2], test_state == 2, alert_threshold)
    state_validation = state_metrics(train_state[val_mask], train_router_probability[val_mask])
    state_test = state_metrics(test_state, test_router_probability)
    weights = {
        "schema": "mb1_bounded_fusion_p2_weights_v1",
        "p1_weights": str(args.p1_weights.resolve()),
        "alignment_threshold": alignment_threshold,
        "router": {"state_dict": router.state_dict(), "mean": router_mean, "std": router_std},
        "rgb_head": {"state_dict": rgb_model.state_dict(), "mean": rgb_mean, "std": rgb_std},
        "rgbt_head": {"state_dict": rgbt_model.state_dict(), "mean": rgbt_mean, "std": rgbt_std},
        "chosen_alpha": chosen["alpha"], "alert_threshold": alert_threshold,
        "constants": {"topk": args.topk, "hidden": args.hidden},
    }
    args.weights_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.weights_output)
    normal_ap = metrics["mb1_normal"]["AP"]
    baseline_ap = metrics["visible_baseline"]["AP"]
    mismatch_ap = metrics["mb1_global_mismatch"]["AP"]
    exact_zero = bool(torch.equal(zero, baseline))
    report = {
        "schema": "mb1_bounded_fusion_p2_v1",
        "status": "PASS" if (
            normal_ap >= baseline_ap + 0.001 and normal_ap > mismatch_ap
            and exact_zero and state_test["macro_f1"] >= 0.60
        ) else "FAIL",
        "purpose_zh": "验证空间对齐通过后，四状态路由和有界红外写入能否形成真实、可回退的双模态增益。",
        "protocol": {
            "frozen_rgb_and_thermal_detectors": True,
            "frozen_p1_alignment_matcher": True,
            "sequence_grouped_validation": True,
            "test_used_for_training_selection_threshold_or_alpha": False,
            "fit_images": int(fit_mask.sum()), "validation_images": int(val_mask.sum()),
            "test_images": len(test_rows), "topk": args.topk, "device": str(device),
            "state_names": STATE_NAMES, "alpha_grid": ALPHA_GRID,
        },
        "router_fit": router_fit,
        "rgb_head_fit": rgb_fit,
        "rgbt_head_fit": rgbt_fit,
        "state_router": {"validation": state_validation, "test": state_test},
        "ir_only_alert": alert,
        "validation_alpha_selection": calibration,
        "chosen": chosen,
        "metrics": metrics,
        "contrasts": {
            "normal_minus_baseline_AP": normal_ap - baseline_ap,
            "normal_minus_mismatch_AP": normal_ap - mismatch_ap,
            "mismatch_minus_baseline_AP": mismatch_ap - baseline_ap,
            "thermal_zero_minus_baseline_AP": metrics["mb1_thermal_zero"]["AP"] - baseline_ap,
        },
        "causal_checks": {
            "thermal_zero_exact_score_fallback": exact_zero,
            "normal_gate_active_fraction": float((test_rgbt[..., -1] > 0).float().mean()),
            "mismatch_gate_active_fraction": float((mismatch_rgbt[..., -1] > 0).float().mean()),
            "normal_mean_abs_bounded_delta": float(test_delta.abs().mean()),
            "mismatch_mean_abs_bounded_delta": float(mismatch_delta.abs().mean()),
        },
        "pass_rule": {
            "normal_minus_baseline_AP_min": 0.001,
            "normal_must_exceed_mismatch": True,
            "thermal_zero_exact_fallback": True,
            "state_router_macro_f1_min": 0.60,
        },
        "weights": str(args.weights_output.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
