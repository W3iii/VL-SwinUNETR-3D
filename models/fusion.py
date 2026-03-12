"""
fusion.py
=========
LAVT-style language-vision fusion module with a language gate,
designed for 3D medical image feature maps.
"""

import torch
import torch.nn as nn


class LanguageVisionFusion(nn.Module):
    """
    LAVT-style language-aware fusion for **3D** visual feature maps,
    extended with a **language gate** that controls per-channel how much
    cross-attended language information is injected into each visual token.

    Gate design (inspired by LAVT PWAM)
    ------------------------------------
    The global text signal is derived by mean-pooling real (non-padding) text
    tokens → (B, C).  A sigmoid MLP maps this to a per-channel gate in [0, 1].
    The gate modulates the cross-attended output before the residual add:

        gate   = sigmoid( W_gate · text_global )    # (B, 1, C)
        fused  = LN( visual + gate ⊙ attended )

    This gives the model a learned, text-conditioned switch per channel and
    per stage, preventing unrelated language signal from corrupting visual
    features when the text offers no useful information for that stage.

    Steps
    -----
    1. Flatten 3D feature map  (B, C, H, W, D) → visual tokens (B, N, C)
    2. Project text tokens to C dimensions
    3. Multi-head cross-attention: visual queries ← text keys/values
    4. Compute language gate from mean-pooled real text tokens
    5. Gate-modulated residual connection + LayerNorm
    6. Reshape fused tokens (B, N, C) → (B, C, H, W, D)

    Parameters
    ----------
    visual_channels : int   channel size of the visual feature map.
    text_channels   : int   channel size of the incoming text tokens.
    num_heads       : int   number of attention heads (must divide visual_channels).
    """

    def __init__(
        self,
        visual_channels: int,
        text_channels: int,
        num_heads: int = 8,
    ):
        super().__init__()

        # Project text tokens into the visual channel space
        self.text_proj = nn.Linear(text_channels, visual_channels)

        # Cross-attention: visual tokens as queries, text tokens as keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_channels,
            num_heads=num_heads,
            batch_first=True,
        )

        # Language gate: maps global text summary → per-channel sigmoid gate
        # text_global (B, C) → gate (B, C)  then broadcast over N tokens
        self.gate_proj = nn.Sequential(
            nn.Linear(visual_channels, visual_channels),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(visual_channels)

    def forward(
        self,
        visual_feat: torch.Tensor,        # (B, C, H, W, D)
        text_tokens: torch.Tensor,        # (B, L, text_channels)
        text_padding_mask: torch.Tensor,  # (B, L), 1=real token, 0=padding
    ) -> torch.Tensor:                    # (B, C, H, W, D)

        B, C, H, W, D = visual_feat.shape
        N = H * W * D

        # ── Step 1: Flatten spatial dims → token sequence ────────────────
        # (B, C, H, W, D) → (B, C, N) → (B, N, C)
        visual_tokens = visual_feat.view(B, C, N).permute(0, 2, 1)

        # ── Step 2: Project text tokens to match visual channel dim ──────
        # (B, L, text_channels) → (B, L, C)
        text_proj = self.text_proj(text_tokens)

        # ── Step 3: Cross-attention ───────────────────────────────────────
        # key_padding_mask: True = ignore (padding).  attention_mask: 1=real → invert.
        key_padding_mask = (text_padding_mask == 0)  # (B, L), True at padding positions

        attended, _ = self.cross_attn(
            query=visual_tokens,    # (B, N, C)
            key=text_proj,          # (B, L, C)
            value=text_proj,        # (B, L, C)
            key_padding_mask=key_padding_mask,
        )                           # attended: (B, N, C)

        # ── Step 4: Language gate ─────────────────────────────────────────
        # Global text summary: mean-pool over real (non-padding) tokens only.
        # text_padding_mask: (B, L), 1=real → used as float weight for mean.
        real_mask = text_padding_mask.float().unsqueeze(-1)  # (B, L, 1)
        text_global = (text_proj * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1e-6)
        # text_global: (B, C)

        gate = self.gate_proj(text_global).unsqueeze(1)  # (B, 1, C) – broadcast over N

        # ── Step 5: Gate-modulated residual + LayerNorm ───────────────────
        # gate ∈ (0,1) controls per-channel language injection strength
        fused_tokens = self.norm(visual_tokens + gate * attended)  # (B, N, C)

        # ── Step 6: Reshape back to 3D feature map ────────────────────────
        # (B, N, C) → (B, C, N) → (B, C, H, W, D)
        fused_feat = fused_tokens.permute(0, 2, 1).view(B, C, H, W, D)

        return fused_feat
