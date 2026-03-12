"""
scripts/inference.py
====================
Sliding-window inference for VL-SwinUNETR-3D – LDCT Breast 3D Segmentation.

Runs inference on test cases and saves the segmentation output as NIfTI
in the original image spacing.

Usage
-----
  python scripts/inference.py \\
      --config  configs/train.yaml \\
      --checkpoint runs/vl_swinunetr3d/model_best.pt \\
      --data_dir  ./data/breast_ct \\
      --json_list dataset.json \\
      --text "breast tumor" \\
      --output_dir ./outputs/predictions \\
      --overlap 0.5
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VL_SwinUNETR3D
from monai import data as monai_data
from monai.data import load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    Invertd,
    SaveImaged,
)
from transformers import AutoTokenizer
from utils.transforms import get_test_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="VL-SwinUNETR-3D inference")
    parser.add_argument("--config",      default="configs/train.yaml",   help="YAML config file")
    parser.add_argument("--checkpoint",  required=True,                   help="model checkpoint (.pt)")
    parser.add_argument("--data_dir",    default=None,  help="override data_dir from config")
    parser.add_argument("--json_list",   default=None,  help="override json_list from config")
    parser.add_argument("--split",       default="validation",
                        choices=["training", "validation", "test"],
                        help="JSON split to run inference on")
    parser.add_argument("--text",        default=None,
                        help="text prompt (overrides per-case text in JSON)")
    parser.add_argument("--output_dir",  default="./outputs/predictions",
                        help="directory to save prediction NIfTI files")
    parser.add_argument("--overlap",     default=None, type=float,
                        help="sliding window overlap (overrides config)")
    parser.add_argument("--sw_batch",    default=None, type=int,
                        help="sliding window batch size (overrides config)")
    parser.add_argument("--no_amp",      action="store_true",
                        help="disable AMP for inference")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m_cfg  = cfg["model"]
    tr_cfg = cfg["training"]
    aug    = cfg["augmentation"]
    tx_cfg = cfg["text"]

    # Allow CLI to override config values
    data_dir   = args.data_dir  or cfg["data"]["data_dir"]
    json_list  = args.json_list or cfg["data"]["json_list"]
    overlap    = args.overlap   or tr_cfg["infer_overlap"]
    sw_batch   = args.sw_batch  or tr_cfg["sw_batch_size"]
    use_amp    = not args.no_amp and tr_cfg["amp"]
    roi        = (aug["roi_x"], aug["roi_y"], aug["roi_z"])
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer & default text ──────────────────────────────────────────
    tokenizer      = AutoTokenizer.from_pretrained(m_cfg["bert_model_name"])
    default_prompt = args.text or tx_cfg["default_prompts"][0]
    max_length     = tx_cfg["max_length"]

    def tokenize(text: str):
        enc = tokenizer(
            text, max_length=max_length,
            padding="max_length", truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    # ── Load model ────────────────────────────────────────────────────────
    model = VL_SwinUNETR3D(
        img_size        = tuple(m_cfg["img_size"]),
        in_channels     = m_cfg["in_channels"],
        out_channels    = m_cfg["out_channels"],
        feature_size    = m_cfg["feature_size"],
        bert_model_name = m_cfg["bert_model_name"],
        freeze_bert     = True,  # always freeze at inference
        num_attn_heads  = m_cfg["num_attn_heads"],
    ).cuda()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch={ckpt.get('epoch','?')})")

    # ── Post-processing ───────────────────────────────────────────────────
    post_pred = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True),
    ])

    # ── Test transforms ───────────────────────────────────────────────────
    test_tf    = get_test_transforms(cfg)
    json_path  = os.path.join(data_dir, json_list)
    test_files = load_decathlon_datalist(json_path, True, args.split, base_dir=data_dir)

    print(f"Running inference on {len(test_files)} cases from split='{args.split}'")

    # ─────────────────────────────────────────────────────────────────────
    # Inference loop
    # ─────────────────────────────────────────────────────────────────────
    for i, case in enumerate(test_files):
        image_path = case["image"]
        case_name  = os.path.splitext(os.path.basename(image_path))[0].replace(".nii", "")

        # Text prompt: CLI override → per-case JSON field → default
        text = args.text or case.get("text") or default_prompt
        input_ids, attention_mask = tokenize(text)

        # ── Load & transform ──────────────────────────────────────────────
        ds     = monai_data.Dataset(data=[case], transform=test_tf)
        sample = ds[0]
        image  = sample["image"].unsqueeze(0).cuda()  # (1, 1, H, W, D)

        print(f"  [{i+1}/{len(test_files)}]  {case_name}  shape={tuple(image.shape[2:])}  text='{text}'")

        # ── Sliding-window inference ───────────────────────────────────────
        with torch.no_grad():
            with autocast(enabled=use_amp):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=roi,
                    sw_batch_size=sw_batch,
                    predictor=lambda patch: model(patch, input_ids, attention_mask),
                    overlap=overlap,
                )

        # ── Post-process & save ────────────────────────────────────────────
        pred = post_pred(logits[0])              # (out_channels, H, W, D)
        pred_np = pred.argmax(0).cpu().numpy().astype(np.uint8)

        # Try to preserve affine from original NIfTI
        try:
            orig_nib = nib.load(image_path)
            affine   = orig_nib.affine
        except Exception:
            affine = np.eye(4)

        out_path = os.path.join(args.output_dir, f"{case_name}_pred.nii.gz")
        nib.save(nib.Nifti1Image(pred_np, affine), out_path)
        print(f"    Saved → {out_path}")

    print(f"\nDone. Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
