"""
scripts/train.py
================
Training script for VL-SwinUNETR-3D – LDCT Breast 3D Segmentation.

Adapted from BTCV SwinUNETR main.py + trainer.py
(https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)

Key differences from BTCV:
  - Model: VL_SwinUNETR3D (SwinUNETR v2 + PubMedBERT + LAVT fusion)
  - Forward pass: model(image, input_ids, attention_mask)
  - Config: YAML file instead of argparse
  - Task: binary breast tumor segmentation (out_channels=2)

Usage
-----
  # single GPU
  python scripts/train.py --config configs/train.yaml

  # resume from checkpoint
  python scripts/train.py --config configs/train.yaml --checkpoint runs/vl_swinunetr3d/model.pt
"""

import argparse
import os
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import yaml
from torch.cuda.amp import GradScaler, autocast

# Ensure project root is on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.ct_text_dataset import get_loader
from models import VL_SwinUNETR3D
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.utils.enums import MetricReduction

try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Running average for scalars."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def save_checkpoint(model, epoch, logdir, filename="model.pt",
                    best_acc=0.0, optimizer=None, scheduler=None):
    state = {
        "epoch":     epoch,
        "best_acc":  best_acc,
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    path = os.path.join(logdir, filename)
    torch.save(state, path)
    print(f"  Checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Warmup-Cosine LR scheduler (mirrors BTCV optimizers/)
# ─────────────────────────────────────────────────────────────────────────────

class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = self.last_epoch / max(self.warmup_epochs, 1)
        else:
            import math
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.max_epochs - self.warmup_epochs, 1
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, cfg):
    """One training epoch. Handles (image, input_ids, attention_mask) inputs."""
    model.train()
    run_loss = AverageMeter()
    start    = time.time()

    for idx, batch in enumerate(loader):
        image          = batch["image"].cuda()           # (B, 1, H, W, D)
        label          = batch["label"].cuda()           # (B, 1, H, W, D)
        input_ids      = batch["input_ids"].cuda()       # (B, L)
        attention_mask = batch["attention_mask"].cuda()  # (B, L)

        optimizer.zero_grad()

        with autocast(enabled=cfg["training"]["amp"]):
            logits = model(image, input_ids, attention_mask)
            loss   = loss_func(logits, label)

        if cfg["training"]["amp"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_loss.update(loss.item(), n=cfg["training"]["batch_size"])

        print(
            f"  Epoch {epoch}/{cfg['training']['max_epochs']}  "
            f"[{idx}/{len(loader)}]  "
            f"loss={run_loss.avg:.4f}  "
            f"time={time.time()-start:.1f}s"
        )
        start = time.time()

    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, cfg,
              model_inferer=None, post_label=None, post_pred=None):
    """One validation epoch with sliding-window inference."""
    model.eval()
    run_acc = AverageMeter()
    start   = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            image          = batch["image"].cuda()
            label          = batch["label"].cuda()
            input_ids      = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            with autocast(enabled=cfg["training"]["amp"]):
                if model_inferer is not None:
                    # Sliding window: text is constant across all patches
                    logits = model_inferer(image, input_ids, attention_mask)
                else:
                    logits = model(image, input_ids, attention_mask)

            val_labels   = decollate_batch(label)
            val_labels   = [post_label(t) for t in val_labels]
            val_outputs  = decollate_batch(logits)
            val_outputs  = [post_pred(t) for t in val_outputs]

            acc_func.reset()
            acc_func(y_pred=val_outputs, y=val_labels)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            avg = float(np.mean(run_acc.avg))
            print(
                f"  Val {epoch}/{cfg['training']['max_epochs']}  "
                f"[{idx}/{len(loader)}]  "
                f"dice={avg:.4f}  "
                f"time={time.time()-start:.1f}s"
            )
            start = time.time()

    return run_acc.avg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VL-SwinUNETR-3D training")
    parser.add_argument("--config",     default="configs/train.yaml", help="YAML config file")
    parser.add_argument("--checkpoint", default=None,                 help="resume from checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tr_cfg   = cfg["training"]
    m_cfg    = cfg["model"]
    os.makedirs(tr_cfg["logdir"], exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader = get_loader(cfg, distributed=False)
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = VL_SwinUNETR3D(
        img_size       = tuple(m_cfg["img_size"]),
        in_channels    = m_cfg["in_channels"],
        out_channels   = m_cfg["out_channels"],
        feature_size   = m_cfg["feature_size"],
        bert_model_name= m_cfg["bert_model_name"],
        freeze_bert    = m_cfg["freeze_bert"],
        num_attn_heads = m_cfg["num_attn_heads"],
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.1f} M")

    # ── Loss / metrics ────────────────────────────────────────────────────
    if tr_cfg.get("squared_dice", False):
        loss_func = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True,
            smooth_nr=tr_cfg["smooth_nr"], smooth_dr=tr_cfg["smooth_dr"],
        )
    else:
        loss_func = DiceCELoss(to_onehot_y=True, softmax=True)

    acc_func   = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN,
        get_not_nans=True,
    )
    post_label = AsDiscrete(to_onehot=True, n_classes=m_cfg["out_channels"])
    post_pred  = AsDiscrete(argmax=True, to_onehot=True, n_classes=m_cfg["out_channels"])

    # ── Sliding-window inferer (text-aware) ───────────────────────────────
    roi = tuple(cfg["augmentation"][f"roi_{k}"] for k in ("x", "y", "z"))

    def model_inferer_fn(image, input_ids, attention_mask):
        predictor = lambda patch: model(patch, input_ids, attention_mask)
        return sliding_window_inference(
            inputs=image,
            roi_size=roi,
            sw_batch_size=tr_cfg["sw_batch_size"],
            predictor=predictor,
            overlap=tr_cfg["infer_overlap"],
        )

    # ── Optimizer ─────────────────────────────────────────────────────────
    if tr_cfg["optim_name"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=tr_cfg["optim_lr"], weight_decay=tr_cfg["reg_weight"]
        )
    elif tr_cfg["optim_name"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=tr_cfg["optim_lr"], weight_decay=tr_cfg["reg_weight"]
        )
    elif tr_cfg["optim_name"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=tr_cfg["optim_lr"],
            momentum=tr_cfg["momentum"], nesterov=True,
            weight_decay=tr_cfg["reg_weight"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {tr_cfg['optim_name']}")

    # ── LR Scheduler ──────────────────────────────────────────────────────
    if tr_cfg["lrschedule"] == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=tr_cfg["warmup_epochs"],
            max_epochs=tr_cfg["max_epochs"],
        )
    elif tr_cfg["lrschedule"] == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tr_cfg["max_epochs"]
        )
    else:
        scheduler = None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc    = 0.0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        start_epoch = ckpt.get("epoch", 0)
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"Resumed from {args.checkpoint}  epoch={start_epoch}  best_acc={best_acc:.4f}")

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=tr_cfg["logdir"]) if HAS_TENSORBOARD else None

    # ── AMP Scaler ────────────────────────────────────────────────────────
    scaler = GradScaler() if tr_cfg["amp"] else None

    # ─────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tr_cfg["max_epochs"]):
        print(f"\n{'='*60}\nEpoch {epoch}  [{time.ctime()}]")

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch, loss_func, cfg)
        print(f"  ▶ train_loss={train_loss:.4f}  elapsed={time.time()-t0:.1f}s")

        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
        if scheduler:
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch) if writer else None
            scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % tr_cfg["val_every"] == 0:
            t0 = time.time()
            val_avg = val_epoch(
                model, val_loader, epoch, acc_func, cfg,
                model_inferer=model_inferer_fn,
                post_label=post_label, post_pred=post_pred,
            )
            val_avg = float(np.mean(val_avg))
            print(f"  ▶ val_dice={val_avg:.4f}  elapsed={time.time()-t0:.1f}s")

            if writer:
                writer.add_scalar("val/dice", val_avg, epoch)

            if tr_cfg["save_checkpoint"]:
                save_checkpoint(
                    model, epoch, tr_cfg["logdir"],
                    filename="model_final.pt",
                    best_acc=best_acc,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

            if val_avg > best_acc:
                best_acc = val_avg
                print(f"  ★ New best dice: {best_acc:.4f}")
                if tr_cfg["save_checkpoint"]:
                    shutil.copyfile(
                        os.path.join(tr_cfg["logdir"], "model_final.pt"),
                        os.path.join(tr_cfg["logdir"], "model_best.pt"),
                    )

    print(f"\nTraining complete. Best Dice: {best_acc:.4f}")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
