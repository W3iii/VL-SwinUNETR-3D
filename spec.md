Create a PyTorch model for 3D medical image segmentation that combines MONAI SwinUNETR v2 with LAVT-style language-aware fusion.

Model goal:
Build a 3D vision-language segmentation model for medical images, where the visual backbone is SwinUNETR v2 and the language encoder is Medical BERT.

Requirements:

1. Use MONAI's SwinUNETR as the visual backbone with:
   - use_v2=True
   - support for 3D medical image segmentation
   - input image shape: (B, C, H, W, D)

2. Use a Medical BERT model as the text encoder.
   - Use HuggingFace Transformers.
   - The text encoder should be loaded from a medical-domain BERT checkpoint.
   - Example: PubMedBERT, Bio_ClinicalBERT, or another medical BERT model.
   - Input text should be tokenized text prompts such as:
     "breast tumor"
     "breast lesion"
     "tumor in breast CT"
   - The text input format should include:
     input_ids
     attention_mask

3. The model should accept:
   - image tensor: (B, C, H, W, D)
   - input_ids: (B, L)
   - attention_mask: (B, L)

4. Encode text using Medical BERT:
   - obtain token-level embeddings of shape (B, L, T)
   - optionally project them into the same channel dimension as visual features

5. Implement LAVT-style language-aware fusion:
   - extract multi-scale visual features from the SwinUNETR encoder
   - for each selected visual stage, flatten 3D feature maps into visual tokens
   - apply cross-attention between visual tokens and Medical BERT text tokens
   - fuse the attended language information back into the visual features
   - reshape fused tokens back to 3D feature maps

6. Implement a reusable fusion module:
   class LanguageVisionFusion(nn.Module)
   - inputs:
       visual_tokens: (B, N, C)
       text_tokens: (B, L, C)
       text_padding_mask: (B, L)
   - use nn.MultiheadAttention for cross-attention
   - return fused visual tokens: (B, N, C)

7. The architecture should contain these classes:
   - MedicalTextEncoder
   - LanguageVisionFusion
   - VL_SwinUNETR3D

8. MedicalTextEncoder should:
   - wrap a HuggingFace medical BERT model
   - output token embeddings
   - optionally freeze or unfreeze BERT parameters with a flag
   - Use microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext as the Medical BERT encoder.

9. VL_SwinUNETR3D should:
   - use SwinUNETR v2 as backbone
   - extract hierarchical encoder features
   - apply language fusion at multiple encoder stages
   - pass fused features into the decoder
   - output final segmentation logits

10. Important implementation details:
   - flatten 3D feature maps into tokens before cross-attention
   - restore fused tokens back to 3D shape after attention
   - keep code modular and readable
   - make the code suitable for research experiments
   - include shape comments for major tensors
   - Do not modify MONAI source code directly. Instead, create a custom model that reuses SwinUNETR internal modules and rewrites the forward pass for multi-stage language fusion.

11. The forward function should be:

   def forward(self, image, input_ids, attention_mask):
       """
       image: (B, C, H, W, D)
       input_ids: (B, L)
       attention_mask: (B, L)
       returns: segmentation logits
       """
       ...

12. Use only:
   - PyTorch
   - MONAI
   - HuggingFace Transformers

13. Please generate complete code with:
   - imports
   - class definitions
   - forward methods
   - tensor reshape logic
   - comments explaining each major step

Focus on a clean research prototype, not production optimization.
make sure adding git commit and referring commit messeage 