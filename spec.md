Create a PyTorch research prototype for 3D medical image segmentation that integrates MONAI SwinUNETR v2 with LAVT-style multi-stage language-aware fusion.

Goal
Build a 3D vision-language segmentation model where:
- visual backbone = MONAI SwinUNETR (use_v2=True)
- language encoder = Medical BERT
- language fusion = LAVT-style multi-stage fusion at every Swin stage

Inputs
- image: (B, C, H, W, D)
- input_ids: (B, L)
- attention_mask: (B, L)

Language Encoder
Implement a MedicalTextEncoder using HuggingFace Transformers.

Use this checkpoint:
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

Requirements:
- return token embeddings of shape (B, L, T)
- add optional freeze_bert flag
- allow projection to match visual feature channel size

Fusion Design (LAVT-style)
Implement multi-stage language-aware fusion.

Language fusion must be applied at every Swin encoder stage:
- stage 0
- stage 1
- stage 2
- stage 3
- stage 4 / bottleneck

Do NOT implement only bottleneck fusion.
Do NOT simplify to FiLM or gating.

Fusion Module
Implement:

LanguageVisionFusion(nn.Module)

Inputs
- visual feature map: (B, C, H, W, D)
- text tokens: (B, L, C)
- text_padding_mask: (B, L)

Steps
1. flatten visual features to tokens
   (B, C, H, W, D) -> (B, N, C)

2. apply multi-head cross attention
   visual tokens attend to text tokens

3. fuse language-aware information

4. reshape tokens back to 3D feature maps

Use:
nn.MultiheadAttention(batch_first=True)

Add:
- residual connection
- layer normalization

Main Model
Implement:

VL_SwinUNETR3D(nn.Module)

Pipeline
image -> SwinUNETR encoder -> hierarchical visual features
text -> Medical BERT -> text tokens

Apply LanguageVisionFusion at each Swin stage feature.

Pass fused multi-scale features into the SwinUNETR decoder.

Output:
segmentation logits

Forward API

def forward(self, image, input_ids, attention_mask):
    """
    image: (B, C, H, W, D)
    input_ids: (B, L)
    attention_mask: (B, L)
    returns: segmentation logits
    """

Implementation Rules

- use PyTorch
- use MONAI
- use HuggingFace Transformers

Important Constraint

Do NOT modify MONAI package source files.
Do NOT edit installed MONAI library code.

Instead:
- build a custom model that reuses MONAI SwinUNETR internal modules
- rewrite the SwinUNETR forward logic inside the custom class
- insert language fusion at each Swin stage

All custom code must live in a new file such as:
models/vl_swinunetr3d.py

Code Requirements

Generate complete research prototype code including:
- imports
- class definitions
- forward methods
- reshape logic for 3D tokens
- comments explaining tensor shapes

Git

Also provide:
- a suggested git commit
- a concise commit message describing the implementation

Reuse SwinUNETR.swinViT outputs and encoder features to perform multi-stage fusion before decoder stages.