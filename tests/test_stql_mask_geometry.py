from pathlib import Path
import sys

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data._misc import convert_to_tv_tensor  # noqa: E402
from src.data.transforms import PadToSize, RandomCrop, RandomHorizontalFlip, Resize  # noqa: E402


def _mask_box(mask: torch.Tensor) -> torch.Tensor:
    points = torch.nonzero(torch.as_tensor(mask)[0] > 0, as_tuple=False)
    assert points.numel() > 0
    y0, x0 = points.min(dim=0).values
    y1, x1 = points.max(dim=0).values + 1
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32)


def test_sam_mask_tracks_box_through_flip_resize_crop_and_pad():
    """The SAM mask and its source box must share every geometric transform."""

    torch.manual_seed(20260921)
    height, width = 96, 128
    image = Image.new("RGB", (width, height), color=(32, 64, 96))
    mask = torch.zeros((1, height, width), dtype=torch.uint8)
    mask[:, 20:76, 24:104] = 1
    target = {
        "boxes": convert_to_tv_tensor(
            torch.tensor([[24.0, 20.0, 104.0, 76.0]]),
            key="boxes",
            box_format="xyxy",
            spatial_size=(height, width),
        ),
        "masks": convert_to_tv_tensor(mask, key="masks"),
        "labels": torch.tensor([0]),
    }

    transforms = (
        RandomHorizontalFlip(p=1.0),
        Resize(size=[80, 112]),
        RandomCrop(size=[64, 88]),
        PadToSize(size=[96, 80]),
    )
    for transform in transforms:
        image, target = transform(image, target)

    transformed_box = torch.as_tensor(target["boxes"])[0].float()
    transformed_mask_box = _mask_box(target["masks"])
    assert torch.allclose(transformed_box, transformed_mask_box, atol=1.0)

    transformed_mask = torch.as_tensor(target["masks"])[0]
    assert tuple(transformed_mask.shape) == (80, 96)
    assert tuple(image.size) == (96, 80)
    assert torch.count_nonzero(transformed_mask[:, 88:]) == 0
    assert torch.count_nonzero(transformed_mask[64:, :]) == 0

    # The STQL target is downsampled with area interpolation.  Its values must
    # remain bounded while retaining fractional boundary coverage.
    soft_target = torch.nn.functional.interpolate(
        transformed_mask[None, None].float(),
        size=(10, 12),
        mode="area",
    )[0, 0]
    assert torch.isfinite(soft_target).all()
    assert 0.0 <= soft_target.min() <= soft_target.max() <= 1.0
    assert ((soft_target > 0.0) & (soft_target < 1.0)).any()


if __name__ == "__main__":
    test_sam_mask_tracks_box_through_flip_resize_crop_and_pad()
    print("PASS test_sam_mask_tracks_box_through_flip_resize_crop_and_pad")
