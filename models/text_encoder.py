"""
text_encoder.py
===============
Medical BERT text encoder that produces token-level embeddings
for use in the VL-SwinUNETR-3D vision-language model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


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
