"""
vl_swinunetr3d.py
=================
Main VL-SwinUNETR-3D model.

Combines MONAI SwinUNETR v2 (visual backbone) with LAVT-style
multi-stage language-aware fusion driven by Medical BERT (PubMedBERT).

Architecture overview
---------------------
image (B,C,H,W,D) ──► SwinViT ──► hidden_states[0..4]
                                        │
                                   LanguageVisionFusion[i]  ◄── text_tokens (B,L,768)
                                        │
                                  encoder/decoder blocks
                                        │
                                   segmentation logits

References
----------
- MONAI SwinUNETR v2: https://github.com/Project-MONAI/MONAI
- LAVT: Language-Aware Vision Transformer for Referring Image Segmentation
- PubMedBERT: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from .text_encoder import MedicalTextEncoder
from .fusion import LanguageVisionFusion

# ─────────────────────────────────────────────────────────────────────────────
# VL-SwinUNETR-3D  Main Model
# ─────────────────────────────────────────────────────────────────────────────

class VL_SwinUNETR3D(nn.Module):
    """
    3D Vision-Language Segmentation Model.

    Reuses MONAI SwinUNETR v2 internal modules **without** modifying
    the installed MONAI source.  The SwinViT forward pass is called
    directly to obtain all 5 hierarchical feature maps, language fusion
    is applied at each stage, and the fused features are forwarded
    through the SwinUNETR encoder/decoder blocks.

    Channel dimensions (default feature_size=48)
    ─────────────────────────────────────────────
    stage 0 :  48  channels   (H/2,  W/2,  D/2)
    stage 1 :  96  channels   (H/4,  W/4,  D/4)
    stage 2 : 192  channels   (H/8,  W/8,  D/8)
    stage 3 : 384  channels   (H/16, W/16, D/16)
    stage 4 : 768  channels   (H/32, W/32, D/32)  ← bottleneck

    Parameters
    ----------
    img_size        : tuple[int,int,int]  spatial size of input images (H, W, D).
    in_channels     : int                 number of image input channels (e.g. 1 for CT).
    out_channels    : int                 number of segmentation classes.
    feature_size    : int                 base feature size for SwinUNETR (default 48).
    bert_model_name : str                 HuggingFace checkpoint for medical BERT.
    freeze_bert     : bool                freeze BERT weights if True.
    num_attn_heads  : int                 preferred number of cross-attention heads.
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 2,
        feature_size: int = 48,
        bert_model_name: str = MedicalTextEncoder.PUBMED_BERT,
        freeze_bert: bool = False,
        num_attn_heads: int = 8,
    ):
        super().__init__()

        # ── Visual backbone: MONAI SwinUNETR v2 ──────────────────────────
        self.swinunetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_v2=True,
        )

        # ── Text encoder: Medical BERT ────────────────────────────────────
        self.text_encoder = MedicalTextEncoder(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
        )
        bert_dim = self.text_encoder.out_channels  # 768

        # ── Channel sizes at each of the 5 Swin stages ───────────────────
        self.stage_channels = [
            feature_size,        # stage 0   (e.g. 48)
            feature_size * 2,    # stage 1   (e.g. 96)
            feature_size * 4,    # stage 2   (e.g. 192)
            feature_size * 8,    # stage 3   (e.g. 384)
            feature_size * 16,   # stage 4   (e.g. 768) — bottleneck
        ]

        # ── LAVT-style fusion at every stage ──────────────────────────────
        self.fusion_modules = nn.ModuleList([
            LanguageVisionFusion(
                visual_channels=ch,
                text_channels=bert_dim,
                num_heads=self._safe_num_heads(ch, num_attn_heads),
            )
            for ch in self.stage_channels
        ])

    # ── Utility ──────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_num_heads(channels: int, requested: int) -> int:
        """Return the largest divisor of `channels` that is ≤ `requested`."""
        for h in [requested, 8, 4, 2, 1]:
            if channels % h == 0:
                return h
        return 1

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        image: torch.Tensor,           # (B, C, H, W, D)
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, out_channels, H, W, D)
        """
        image          : (B, C, H, W, D)
        input_ids      : (B, L)
        attention_mask : (B, L)
        returns        : segmentation logits (B, out_channels, H, W, D)
        """

        # ── 1. Text encoding ──────────────────────────────────────────────
        # text_tokens: (B, L, 768)
        text_tokens = self.text_encoder(input_ids, attention_mask)

        # ── 2. Hierarchical visual features from SwinViT ──────────────────
        # hidden_states_out: list of 5 tensors
        #   [0]: (B,  48, H/2,  W/2,  D/2)
        #   [1]: (B,  96, H/4,  W/4,  D/4)
        #   [2]: (B, 192, H/8,  W/8,  D/8)
        #   [3]: (B, 384, H/16, W/16, D/16)
        #   [4]: (B, 768, H/32, W/32, D/32)
        hidden_states_out = self.swinunetr.swinViT(image, self.swinunetr.normalize)

        # ── 3. LAVT-style multi-stage language fusion ─────────────────────
        # Apply LanguageVisionFusion at every Swin encoder stage.
        fused = [
            fusion_mod(feat, text_tokens, attention_mask)
            for feat, fusion_mod in zip(hidden_states_out, self.fusion_modules)
        ]
        # fused[i]: same shape as hidden_states_out[i], enriched with language

        # ── 4. SwinUNETR encoder blocks  (skip connections) ───────────────
        # enc0 : processed raw-image skip  (B, feat,   H,   W,   D)
        # enc1 : processed fused stage-0   (B, feat,   H/2, W/2, D/2)
        # enc2 : processed fused stage-1   (B, feat*2, H/4, W/4, D/4)
        # enc3 : processed fused stage-2   (B, feat*4, H/8, W/8, D/8)
        # dec4 : bottleneck from fused stage-4
        enc0 = self.swinunetr.encoder1(image)           # raw image skip
        enc1 = self.swinunetr.encoder2(fused[0])        # fused stage 0
        enc2 = self.swinunetr.encoder3(fused[1])        # fused stage 1
        enc3 = self.swinunetr.encoder4(fused[2])        # fused stage 2
        dec4 = self.swinunetr.encoder10(fused[4])       # bottleneck (stage 4)

        # ── 5. SwinUNETR decoder blocks ───────────────────────────────────
        # fused[3] used directly as skip (stage-3 language-fused feature)
        dec3 = self.swinunetr.decoder5(dec4, fused[3])  # bottleneck + stage-3 skip
        dec2 = self.swinunetr.decoder4(dec3, enc3)      # + stage-2 skip
        dec1 = self.swinunetr.decoder3(dec2, enc2)      # + stage-1 skip
        dec0 = self.swinunetr.decoder2(dec1, enc1)      # + stage-0 skip
        out  = self.swinunetr.decoder1(dec0, enc0)      # + raw image skip

        # ── 6. Segmentation head ──────────────────────────────────────────
        logits = self.swinunetr.out(out)  # (B, out_channels, H, W, D)

        return logits
