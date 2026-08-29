"""
How a chest X-ray gets prepared before it goes into the model.

Everything imports from this one file -- classifier training, report generator
training, evaluation, and the backend. If you define the transforms somewhere
else as well, training and inference drift apart and the model quietly gets
worse. So: one file, imported everywhere.

    from cxr_transforms import build_transform
    train_tf = build_transform("train")
    eval_tf  = build_transform("eval")
"""
from __future__ import annotations
import numpy as np, torch
from PIL import Image
from torchvision import transforms

IMG_SIZE = 384
AUG_DEGREES = 5.0
AUG_TRANSLATE = (0.03, 0.03)
AUG_SCALE = (0.97, 1.03)
USE_CLAHE = False
CLAHE_CLIP, CLAHE_GRID = 2.0, (8, 8)
LEGACY_IMAGENET_MEAN = [0.485, 0.456, 0.406]
LEGACY_IMAGENET_STD = [0.229, 0.224, 0.225]
DATASET_GRAY_MEAN = 0.4732
DATASET_GRAY_STD = 0.3036
_EPS = 1e-6


class ToGrayscalePIL:
    """Force PIL mode 'L'. Uploaded images may arrive RGB / RGBA / P."""
    def __call__(self, img):
        return img if img.mode == "L" else img.convert("L")
    def __repr__(self):
        return "ToGrayscalePIL()"


class CLAHE:
    def __init__(self, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_GRID):
        self.clip_limit, self.tile_grid, self._c = clip_limit, tile_grid, None
    def _get(self):
        if self._c is None:
            import cv2
            self._c = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        return self._c
    def __call__(self, img):
        return Image.fromarray(self._get().apply(np.asarray(img, dtype=np.uint8)), mode="L")
    def __repr__(self):
        return f"CLAHE(clip={self.clip_limit}, grid={self.tile_grid})"


class PerImageZScore:
    """
    Normalise each image using its own mean and standard deviation, then copy it
    into 3 channels because ConvNeXt expects colour input.
    (1,H,W) in [0,1] -> (3,H,W) with mean about 0 and std about 1.

    The std > _EPS check matters. A completely flat image has zero standard
    deviation, and dividing by that gives NaN, which then spreads through
    everything without raising an error. Our dataset has no such images, but an
    uploaded file might.
    """
    def __init__(self, out_channels: int = 3):
        self.out_channels = out_channels
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] != 1:
            t = t[:1]
        s = t.std()
        t = (t - t.mean()) / s if s > _EPS else t - t.mean()
        return t.repeat(self.out_channels, 1, 1) if self.out_channels > 1 else t
    def __repr__(self):
        return f"PerImageZScore(out_channels={self.out_channels})"


def build_transform(split: str, img_size: int = IMG_SIZE,
                    use_clahe: bool = USE_CLAHE, normalize: str = "per_image"):
    """
    split     : "train" adds augmentation. "val"/"test"/"eval"/"inference" don't.
    normalize : "per_image" is what we use. "dataset" and "imagenet" are kept
                for comparison only -- imagenet measured 4.4x worse.
    returns   : a Compose that gives you (3, img_size, img_size) float32
    """
    split = split.lower()
    if split not in {"train", "val", "test", "eval", "inference"}:
        raise ValueError(f"unknown split {split!r}")
    if normalize not in {"per_image", "dataset", "imagenet"}:
        raise ValueError(f"unknown normalize {normalize!r}")

    # Resize runs on every split. It does nothing for our 384x384 dataset, but
    # it handles odd-sized uploads and guarantees train and inference match.
    ops = [ToGrayscalePIL(), transforms.Resize((img_size, img_size))]
    if use_clahe:
        ops.append(CLAHE())
    if split == "train":
        # Small geometric changes only.
        # No horizontal flip: that would put the heart on the wrong side, which
        # is a real condition and not something the model should learn to ignore.
        # No brightness/contrast jitter either, because brightness IS the signal
        # for things like edema and lung opacity.
        ops.append(transforms.RandomAffine(
            degrees=AUG_DEGREES, translate=AUG_TRANSLATE, scale=AUG_SCALE,
            interpolation=transforms.InterpolationMode.BILINEAR, fill=0))
    ops.append(transforms.ToTensor())
    if normalize == "per_image":
        ops.append(PerImageZScore(3))
    elif normalize == "dataset":
        ops.append(transforms.Lambda(lambda t: t.repeat(3, 1, 1)))
        ops.append(transforms.Normalize([DATASET_GRAY_MEAN]*3, [DATASET_GRAY_STD]*3))
    else:
        ops.append(transforms.Lambda(lambda t: t.repeat(3, 1, 1)))
        ops.append(transforms.Normalize(LEGACY_IMAGENET_MEAN, LEGACY_IMAGENET_STD))
    return transforms.Compose(ops)


def transform_config(**kw) -> dict:
    """A plain dict of the current settings. Save it next to the checkpoint so
    you can always tell how a model was trained."""
    cfg = dict(stage=2, img_size=IMG_SIZE, normalize=kw.get("normalize", "per_image"),
               use_clahe=kw.get("use_clahe", USE_CLAHE), aug_degrees=AUG_DEGREES,
               aug_translate=list(AUG_TRANSLATE), aug_scale=list(AUG_SCALE),
               horizontal_flip=False, color_jitter=False, random_autocontrast=False,
               resize_always_applied=True, out_channels=3,
               dataset_gray_mean=DATASET_GRAY_MEAN, dataset_gray_std=DATASET_GRAY_STD)
    cfg.update(kw)
    return cfg


__all__ = ["build_transform", "transform_config", "PerImageZScore", "CLAHE",
           "ToGrayscalePIL", "IMG_SIZE"]
