"""
utils/transforms.py
===================
MONAI transform pipelines for LDCT breast 3D segmentation.

Adapted from BTCV SwinUNETR data_utils.py
(https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)
and adjusted for:
  - single-class binary breast tumor segmentation
  - LDCT HU windowing (soft-tissue breast window)
  - 0.7 × 0.7 × 1.5 mm target spacing
"""

import random

import torch
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom transform: add default text prompt if missing
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    "breast tumor",
    "breast lesion",
    "tumor in breast CT",
    "malignant breast mass",
    "breast cancer segmentation",
]


class AddDefaultTextd(MapTransform):
    """
    If the data dict has no 'text' key (or it is empty / None),
    randomly assign one of the default prompts.
    """

    def __init__(self, key: str = "text", prompts: list[str] = DEFAULT_PROMPTS):
        super().__init__(keys=[key], allow_missing_keys=True)
        self.key = key
        self.prompts = prompts

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        if not d.get(self.key):
            d[self.key] = random.choice(self.prompts)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Transform pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(cfg: dict) -> Compose:
    """
    Full augmentation pipeline for training.

    cfg keys used:
        augmentation.{a_min, a_max, b_min, b_max,
                      space_x, space_y, space_z,
                      roi_x, roi_y, roi_z,
                      RandFlipd_prob, RandRotate90d_prob,
                      RandScaleIntensityd_prob, RandShiftIntensityd_prob}
    """
    aug = cfg["augmentation"]
    roi = (aug["roi_x"], aug["roi_y"], aug["roi_z"])
    spacing = (aug["space_x"], aug["space_y"], aug["space_z"])

    return Compose(
        [
            # ── Load ──────────────────────────────────────────────────────
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),         # (1,H,W,D)
            AddDefaultTextd(key="text"),

            # ── Spatial preprocessing ──────────────────────────────────────
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),

            # ── Intensity preprocessing (LDCT breast window) ───────────────
            ScaleIntensityRanged(
                keys=["image"],
                a_min=aug["a_min"], a_max=aug["a_max"],
                b_min=aug["b_min"], b_max=aug["b_max"],
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                allow_smaller=True,
            ),

            # ── Random crop: equal pos/neg sampling ───────────────────────
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),

            # ── Spatial augmentation ──────────────────────────────────────
            RandFlipd(
                keys=["image", "label"],
                prob=aug["RandFlipd_prob"],
                spatial_axis=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=aug["RandFlipd_prob"],
                spatial_axis=1,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=aug["RandFlipd_prob"],
                spatial_axis=2,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=aug["RandRotate90d_prob"],
                max_k=3,
            ),

            # ── Intensity augmentation ────────────────────────────────────
            RandScaleIntensityd(
                keys=["image"],
                factors=0.1,
                prob=aug["RandScaleIntensityd_prob"],
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=aug["RandShiftIntensityd_prob"],
            ),

            ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transforms(cfg: dict) -> Compose:
    """
    Deterministic pipeline for validation (no augmentation, no random crop).
    """
    aug = cfg["augmentation"]
    spacing = (aug["space_x"], aug["space_y"], aug["space_z"])

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            AddDefaultTextd(key="text"),

            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=aug["a_min"], a_max=aug["a_max"],
                b_min=aug["b_min"], b_max=aug["b_max"],
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                allow_smaller=True,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_test_transforms(cfg: dict) -> Compose:
    """
    Pipeline for test/inference (no label required).
    """
    aug = cfg["augmentation"]
    spacing = (aug["space_x"], aug["space_y"], aug["space_z"])

    return Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            AddDefaultTextd(key="text"),

            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=spacing,
                mode="bilinear",
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=aug["a_min"], a_max=aug["a_max"],
                b_min=aug["b_min"], b_max=aug["b_max"],
                clip=True,
            ),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                allow_smaller=True,
            ),
            ToTensord(keys=["image"]),
        ]
    )
