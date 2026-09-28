"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import faster_coco_eval.core.mask as coco_mask
from faster_coco_eval.utils.pytorch import FasterCocoDetection
import numpy as np
import torch
import torchvision
import os
import json
import math
from pathlib import Path
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset

torchvision.disable_beta_transforms_warning()
Image.MAX_IMAGE_PIXELS = None

__all__ = ["CocoDetection", "RGBTCocoDetection"]


@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    __inject__ = [
        "transforms",
    ]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        sam_mask_root=None,
    ):
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.sam_mask_root = Path(sam_mask_root) if sam_mask_root else None
        self.sam_mask_accepted_ids = set()
        self.sam_mask_quality = {}
        self.sam_mask_files = {}
        if self.sam_mask_root is not None:
            records_path = self.sam_mask_root / "records.json"
            if not records_path.is_file():
                raise FileNotFoundError(f"SAM mask records not found: {records_path}")
            records = json.loads(records_path.read_text(encoding="utf-8"))
            self.sam_mask_accepted_ids = {
                int(record["image_id"])
                for record in records
                if bool(record.get("accepted", False))
            }
            for record in records:
                if not bool(record.get("accepted", False)):
                    continue
                if "instance_mask_files" in record:
                    files = record["instance_mask_files"]
                    if not isinstance(files, list) or not files or not all(isinstance(name, str) for name in files):
                        raise ValueError("Invalid SAM instance_mask_files")
                    if any(Path(name).name != name for name in files):
                        raise ValueError("SAM instance mask file names must not include directories")
                    self.sam_mask_files[int(record["image_id"])] = files
                if "supervision_weight" in record:
                    weight = float(record["supervision_weight"])
                    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                        raise ValueError("Invalid explicit SAM supervision_weight")
                    self.sam_mask_quality[int(record["image_id"])] = weight
                    continue
                pred_iou = float(record.get("pred_iou", 0.0))
                stability = float(record.get("stability", 0.0))
                weight_iou = min(1.0, max(0.0, (pred_iou - 0.75) / 0.15))
                weight_stability = min(
                    1.0, max(0.0, (stability - 0.85) / 0.10)
                )
                self.sam_mask_quality[int(record["image_id"])] = math.sqrt(
                    weight_iou * weight_stability
                )

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        image_path = os.path.join(self.img_folder, self.coco.loadImgs(image_id)[0]["file_name"])
        target = {"image_id": image_id, "image_path": image_path, "annotations": target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        if self.sam_mask_root is not None:
            height, width = image.size[1], image.size[0]
            num_objects = int(target["boxes"].shape[0])
            masks = torch.zeros((num_objects, height, width), dtype=torch.uint8)
            if num_objects > 0 and image_id in self.sam_mask_accepted_ids:
                names = self.sam_mask_files.get(int(image_id))
                if names is None:
                    if num_objects != 1:
                        raise RuntimeError(f"Multi-object SAM masks missing instance_mask_files: image_id={image_id}")
                    names = [f"{int(image_id):06d}.png"]
                if len(names) != num_objects:
                    raise RuntimeError(f"SAM mask count differs from box count: image_id={image_id}")
                for object_index, name in enumerate(names):
                    mask_path = self.sam_mask_root / "masks" / name
                    if not mask_path.is_file():
                        raise FileNotFoundError(f"Accepted SAM mask is missing: {mask_path}")
                    mask = Image.open(mask_path).convert("L")
                    if mask.size != image.size:
                        raise RuntimeError(
                            f"SAM mask/image size mismatch for image_id={image_id}: "
                            f"mask={mask.size}, image={image.size}"
                        )
                    masks[object_index] = torch.from_numpy(np.array(mask, dtype=np.uint8, copy=True)).gt(0)
            target["masks"] = masks
            target["sam_quality"] = torch.tensor(
                [self.sam_mask_quality.get(int(image_id), 0.0)], dtype=torch.float32
            )

        target["idx"] = torch.tensor([idx])

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(
                target["boxes"], key="boxes", spatial_size=image.size[::-1]
            )

        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")

        return image, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        s += f" sam_mask_root: {self.sam_mask_root}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        if hasattr(self, "_preset") and self._preset is not None:
            s += f" preset:\n   {repr(self._preset)}"
        return s

    @property
    def categories(self):
        return self.coco.dataset["categories"]

    @property
    def category2name(self):
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category2label(self):
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self):
        return {i: cat["id"] for i, cat in enumerate(self.categories)}


@register()
class RGBTCocoDetection(CocoDetection):
    """Paired RGB-T COCO dataset with Visible-space detection supervision.

    The two images are transformed together so every geometric augmentation
    uses the same sampled parameters.  They are concatenated only after the
    transform pipeline, yielding a conventional ``[6, H, W]`` tensor that is
    compatible with the existing collator and training engine.  Infrared box
    coordinates never replace the Visible detection target.  They are carried
    as a second BoundingBoxes tensor so torchvision v2 applies the exact same
    sampled crop/flip/resize to both coordinate systems.  M-B2 can therefore
    supervise cross-modal alignment without pretending the cameras are pixel
    aligned.
    """

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        infrared_folder,
        infrared_label_folder=None,
        infrared_index_offset=0,
        return_masks=False,
        remap_mscoco_category=False,
        sam_mask_root=None,
    ):
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
            sam_mask_root=sam_mask_root,
        )
        self.infrared_folder = Path(infrared_folder)
        self.infrared_label_folder = (
            Path(infrared_label_folder) if infrared_label_folder else None
        )
        # Evaluation-only causal control.  A non-zero offset pairs each RGB
        # sample with a far-away infrared sample from the same split while
        # leaving the RGB image and its detection target untouched.
        self.infrared_index_offset = int(infrared_index_offset)
        if not self.infrared_folder.is_dir():
            raise FileNotFoundError(
                f"Infrared image directory not found: {self.infrared_folder}"
            )
        if (
            self.infrared_label_folder is not None
            and not self.infrared_label_folder.is_dir()
        ):
            raise FileNotFoundError(
                "Infrared label directory not found: "
                f"{self.infrared_label_folder}"
            )

    def _infrared_presence(self, stem):
        if self.infrared_label_folder is None:
            return 0.0
        label_path = self.infrared_label_folder / f"{stem}.txt"
        if not label_path.is_file():
            return 0.0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if len(fields) >= 5:
                try:
                    values = [float(value) for value in fields[:5]]
                except ValueError:
                    continue
                if values[3] > 0.0 and values[4] > 0.0:
                    return 1.0
        return 0.0

    def _infrared_boxes(self, stem, spatial_size):
        """Load YOLO-normalized infrared boxes on the resized Visible canvas."""
        if self.infrared_label_folder is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            label_path = self.infrared_label_folder / f"{stem}.txt"
            parsed = []
            if label_path.is_file():
                for line in label_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    fields = line.strip().split()
                    if len(fields) < 5:
                        continue
                    try:
                        center_x, center_y, width, height = map(
                            float, fields[1:5]
                        )
                    except ValueError:
                        continue
                    if not (
                        0.0 <= center_x <= 1.0
                        and 0.0 <= center_y <= 1.0
                        and 0.0 < width <= 1.0
                        and 0.0 < height <= 1.0
                    ):
                        continue
                    parsed.append(
                        [
                            center_x - width / 2.0,
                            center_y - height / 2.0,
                            center_x + width / 2.0,
                            center_y + height / 2.0,
                        ]
                    )
            boxes = torch.as_tensor(parsed, dtype=torch.float32).reshape(-1, 4)
        height, width = map(int, spatial_size)
        if len(boxes):
            boxes = boxes * boxes.new_tensor([width, height, width, height])
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
        return convert_to_tv_tensor(
            boxes,
            key="boxes",
            box_format="XYXY",
            spatial_size=(height, width),
        )

    @staticmethod
    def _valid_infrared_boxes(boxes):
        """Drop boxes made empty by a crop while preserving BoundingBoxes metadata."""
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise RuntimeError(
                "infrared_boxes must have shape [N, 4], got "
                f"{tuple(boxes.shape)}"
            )
        if len(boxes) == 0:
            return boxes
        values = torch.as_tensor(boxes)
        box_format = getattr(getattr(boxes, "format", None), "value", "")
        if str(box_format).upper() == "XYXY":
            extent = values[:, 2:] - values[:, :2]
        else:
            # All standard training presets finish with normalized CXCYWH.
            extent = values[:, 2:]
        keep = torch.isfinite(values).all(dim=1) & (extent > 1e-6).all(dim=1)
        return boxes[keep]

    def __getitem__(self, idx):
        visible, target = self.load_item(idx)
        visible_path = Path(target["image_path"])
        infrared_idx = (idx + self.infrared_index_offset) % len(self.ids)
        infrared_image_id = self.ids[infrared_idx]
        infrared_name = self.coco.loadImgs(infrared_image_id)[0]["file_name"]
        infrared_path = self.infrared_folder / infrared_name
        infrared_stem = Path(infrared_name).stem
        if not infrared_path.is_file():
            raise FileNotFoundError(
                f"Paired infrared image not found for {visible_path.name}: "
                f"{infrared_path}"
            )

        with Image.open(infrared_path) as image:
            infrared = image.convert("RGB")
        if infrared.size != visible.size:
            infrared = infrared.resize(visible.size, resample=Image.Resampling.BILINEAR)

        target["infrared_boxes"] = self._infrared_boxes(
            infrared_stem, visible.size[::-1]
        )
        modalities = {"visible": visible, "infrared": infrared}
        if self._transforms is not None:
            modalities, target, _ = self._transforms(modalities, target, self)

        target["infrared_boxes"] = self._valid_infrared_boxes(
            target["infrared_boxes"]
        )

        visible_tensor = modalities["visible"]
        infrared_tensor = modalities["infrared"]
        if visible_tensor.ndim != 3 or infrared_tensor.ndim != 3:
            raise RuntimeError(
                "RGB-T transforms must return CHW tensors, got "
                f"{tuple(visible_tensor.shape)} and {tuple(infrared_tensor.shape)}"
            )
        if visible_tensor.shape[-2:] != infrared_tensor.shape[-2:]:
            raise RuntimeError(
                "Paired RGB-T tensors must share a spatial size after transforms, got "
                f"{tuple(visible_tensor.shape)} and {tuple(infrared_tensor.shape)}"
            )

        target["infrared_presence"] = torch.tensor(
            [float(len(target["infrared_boxes"]) > 0)], dtype=torch.float32
        )
        # Presence describes whether the annotation contains an object;
        # availability describes whether the physical modality exists.  QCER
        # must not confuse a valid negative IR frame with a missing modality.
        target["infrared_available"] = torch.tensor([True], dtype=torch.bool)
        target["infrared_label_known"] = torch.tensor(
            [self.infrared_label_folder is not None], dtype=torch.bool
        )
        target["infrared_image_path"] = str(infrared_path)
        target["infrared_index_offset"] = torch.tensor(
            [self.infrared_index_offset], dtype=torch.int64
        )
        paired = torch.cat((visible_tensor, infrared_tensor), dim=0)
        return paired, target

    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f"\n infrared_folder: {self.infrared_folder}"
        s += f"\n infrared_label_folder: {self.infrared_label_folder}"
        s += f"\n infrared_index_offset: {self.infrared_index_offset}"
        return s


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        image_path = target["image_path"]

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get("category2label", None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["image_path"] = image_path
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
