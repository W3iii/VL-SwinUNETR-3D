"""
datasets/ct_text_dataset.py
===========================
Dataset and DataLoader utilities for LDCT breast 3D segmentation
with paired text prompts.

Data format (JSON – decathlon-style)
-------------------------------------
{
  "training": [
    {"image": "imagesTr/case001.nii.gz",
     "label": "labelsTr/case001.nii.gz",
     "text":  "breast tumor"},        ← optional; defaults to random prompt
    ...
  ],
  "validation": [
    {"image": "imagesVal/case001.nii.gz",
     "label": "labelsVal/case001.nii.gz"},
    ...
  ]
}

Each DataLoader batch contains:
  batch["image"]          : (B, 1, H, W, D)   float32
  batch["label"]          : (B, 1, H, W, D)   int64
  batch["input_ids"]      : (B, L)             int64
  batch["attention_mask"] : (B, L)             int64
  batch["text"]           : list[str]          raw prompt strings
"""

import os
import random

import torch
from monai import data as monai_data
from monai.data import load_decathlon_datalist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from utils.transforms import DEFAULT_PROMPTS, get_test_transforms, get_train_transforms, get_val_transforms

# ─────────────────────────────────────────────────────────────────────────────
# Distributed sampler (identical to BTCV reference)
# ─────────────────────────────────────────────────────────────────────────────

import math

import numpy as np
import torch.distributed


class DistributedSampler(torch.utils.data.Sampler):
    """Distributed sampler with optional padding to ensure equal-length batches."""

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(dataset) / num_replicas))
        self.total_size = self.num_samples * num_replicas
        indices = list(range(len(dataset)))
        self.valid_length = len(indices[rank : self.total_size : num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even and len(indices) < self.total_size:
            extra = np.random.randint(0, len(indices), self.total_size - len(indices))
            indices += [indices[i] for i in extra]
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# ─────────────────────────────────────────────────────────────────────────────
# Text-aware dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CTTextDataset(Dataset):
    """
    Wraps a MONAI dataset to add text tokenization.

    For each sample the underlying MONAI dataset returns a dict with
    "image", "label", and optionally "text".  This wrapper:
      1. Retrieves the sample from the base dataset.
      2. Fills in a random default prompt if "text" is missing.
      3. Tokenizes the text with PubMedBERT tokenizer.
      4. Adds "input_ids" and "attention_mask" to the sample dict.

    Parameters
    ----------
    base_dataset : Dataset
        A MONAI Dataset / CacheDataset that already applies spatial transforms.
    tokenizer    : AutoTokenizer
        HuggingFace tokenizer (PubMedBERT).
    max_length   : int
        Max token length (padding & truncation).
    default_prompts : list[str]
        Pool of prompts used when JSON entry has no "text" field.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer: AutoTokenizer,
        max_length: int = 32,
        default_prompts: list[str] = DEFAULT_PROMPTS,
    ):
        self.base = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.default_prompts = default_prompts

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]

        # ── Resolve text prompt ──────────────────────────────────────────
        # RandCropByPosNegLabeld returns a list of dicts; handle both.
        if isinstance(sample, list):
            # each crop shares the same text
            text = sample[0].get("text") or random.choice(self.default_prompts)
        else:
            text = sample.get("text") or random.choice(self.default_prompts)

        # ── Tokenise ──────────────────────────────────────────────────────
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].squeeze(0)       # (L,)
        attention_mask = encoded["attention_mask"].squeeze(0)  # (L,)

        # ── Attach to sample(s) ───────────────────────────────────────────
        if isinstance(sample, list):
            for s in sample:
                s["text"]           = text
                s["input_ids"]      = input_ids
                s["attention_mask"] = attention_mask
        else:
            sample["text"]           = text
            sample["input_ids"]      = input_ids
            sample["attention_mask"] = attention_mask

        return sample


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_loader(cfg: dict, distributed: bool = False):
    """
    Build train / val DataLoaders for LDCT breast segmentation.

    Parameters
    ----------
    cfg         : full config dict (from train.yaml)
    distributed : whether to use DistributedSampler

    Returns
    -------
    (train_loader, val_loader)
    """
    data_cfg  = cfg["data"]
    text_cfg  = cfg["text"]
    train_cfg = cfg["training"]

    data_dir     = data_cfg["data_dir"]
    json_path    = os.path.join(data_dir, data_cfg["json_list"])
    max_length   = text_cfg["max_length"]
    prompts      = text_cfg["default_prompts"]
    workers      = data_cfg["workers"]
    batch_size   = train_cfg["batch_size"]

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_model_name"])

    # ── Transforms ────────────────────────────────────────────────────────
    train_tf = get_train_transforms(cfg)
    val_tf   = get_val_transforms(cfg)

    # ── Load JSON manifest ────────────────────────────────────────────────
    train_files = load_decathlon_datalist(json_path, True, "training",  base_dir=data_dir)
    val_files   = load_decathlon_datalist(json_path, True, "validation", base_dir=data_dir)

    # ── MONAI base datasets ───────────────────────────────────────────────
    if data_cfg.get("use_normal_dataset", False):
        monai_train_ds = monai_data.Dataset(data=train_files, transform=train_tf)
    else:
        monai_train_ds = monai_data.CacheDataset(
            data=train_files,
            transform=train_tf,
            cache_num=data_cfg["cache_num"],
            cache_rate=data_cfg["cache_rate"],
            num_workers=workers,
        )
    monai_val_ds = monai_data.Dataset(data=val_files, transform=val_tf)

    # ── Wrap with text tokenisation ───────────────────────────────────────
    train_ds = CTTextDataset(monai_train_ds, tokenizer, max_length, prompts)
    val_ds   = CTTextDataset(monai_val_ds,   tokenizer, max_length, prompts)

    # ── Samplers ──────────────────────────────────────────────────────────
    train_sampler = DistributedSampler(train_ds) if distributed else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if distributed else None

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        sampler=val_sampler,
        pin_memory=True,
    )

    return train_loader, val_loader
