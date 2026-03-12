"""
VL-SwinUNETR-3D
===============
3D Vision-Language Segmentation Model for Medical Images.

Combines MONAI SwinUNETR v2 (visual backbone) with LAVT-style
multi-stage language-aware fusion driven by Medical BERT (PubMedBERT).

Language fusion is applied at **every** SwinViT encoder stage (0–4)
before features are forwarded into the SwinUNETR decoder.

Architecture overview
---------------------
image (B,C,H,W,D) ──► SwinViT ──► hidden_states[0..4]  ──► LanguageVisionFusion[i] ──► enc/dec blocks ──► logits
text  (B,L)        ──► MedicalBERT ──► text_tokens (B,L,768)  ──────────────────────────────────►

References
----------
- MONAI SwinUNETR v2: https://github.com/Project-MONAI/MONAI
- LAVT: Language-Aware Vision Transformer for Referring Image Segmentation
- PubMedBERT: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from transformers import AutoModel

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Medical Text Encoder
# ─────────────────────────────────────────────────────────────────────────────

class MedicalTextEncoder(nn.Module):
    """
    Wraps a HuggingFace medical BERT model to produce token-level embeddings.

    Parameters
    ----------
    bert_model_name : str
        HuggingFace model identifier.
    out_channels : int | None
        If given, projects BERT hidden size → out_channels.
        If None, returns raw BERT hidden size (typically 768).
    freeze_bert : bool
        Whether to freeze BERT parameters during training.

    Forward
    -------
    input_ids      : (B, L)
    attention_mask : (B, L)
    returns        : (B, L, out_channels)
    """

    PUBMED_BERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    def __init__(
        self,
        bert_model_name: str = PUBMED_BERT,
        out_channels: int | None = None,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # 768

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Optional linear projection to match visual feature channels
        if out_channels is not None and out_channels != self.bert_hidden_size:
            self.proj: nn.Module = nn.Linear(self.bert_hidden_size, out_channels)
            self.out_channels = out_channels
        else:
            self.proj = nn.Identity()
            self.out_channels = self.bert_hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:                 # (B, L, out_channels)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, L, bert_hidden_size)
        return self.proj(token_embeddings)            # (B, L, out_channels)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Language-Vision Fusion Module  (LAVT-style + Language Gate)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VL-SwinUNETR-3D  Main Model
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
