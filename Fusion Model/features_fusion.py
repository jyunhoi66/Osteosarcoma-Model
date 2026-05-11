"""Multimodal transformer fusion model for binary classification.

This module contains the main model used in the paper code release. The model
combines global image features, radiomics features, and tumor-region image
features using a lightweight transformer encoder initialized from a ViT
checkpoint. The forward pass returns class logits and intentionally does not
apply softmax, so it can be used directly with ``CrossEntropyLoss``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from transformers import ViTModel

__all__ = ["RadiomicsTokenProjector", "BinaryClassifier"]


class RadiomicsTokenProjector(nn.Module):
    """Project tabular radiomics features into the image embedding space.

    Parameters
    ----------
    rad_dim:
        Number of radiomics features.
    embed_dim:
        Dimension of the image and transformer token embeddings.
    """

    def __init__(self, rad_dim: int, embed_dim: int = 768) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(rad_dim),
            nn.Linear(rad_dim, embed_dim, bias=False),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, radiomics_features: torch.Tensor) -> torch.Tensor:
        """Return a radiomics token with shape ``[batch_size, embed_dim]``."""
        return self.scale * self.projection(radiomics_features)


class BinaryClassifier(nn.Module):
    """Transformer-based fusion classifier for image, radiomics, and tumor features.

    Parameters
    ----------
    rad_dim:
        Number of radiomics features.
    tumor_dim:
        Dimension of the tumor-region feature vector. It must match ``img_dim``
        for token stacking.
    img_dim:
        Dimension of the global image feature vector.
    hidden_dims:
        Hidden layer sizes for the classification head.
    dropout:
        Dropout probability used in the classification head.
    pretrained_model_name_or_path:
        Hugging Face model name or local directory containing a ViT checkpoint.
    """

    def __init__(
        self,
        rad_dim: int,
        tumor_dim: int,
        img_dim: int = 768,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
        pretrained_model_name_or_path: str | Path = "google/vit-base-patch16-224-in21k",
    ) -> None:
        super().__init__()
        if tumor_dim != img_dim:
            raise ValueError(
                f"tumor_dim ({tumor_dim}) must match img_dim ({img_dim}) for token fusion."
            )
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must contain exactly two values.")

        self.radiomics_projector = RadiomicsTokenProjector(
            rad_dim=rad_dim,
            embed_dim=img_dim,
        )

        vit = ViTModel.from_pretrained(str(pretrained_model_name_or_path))
        self.token_encoder = vit.encoder
        self.refinement_encoder = ViTModel.from_pretrained(
            str(pretrained_model_name_or_path)
        ).encoder
        self.embedding_dim = vit.config.hidden_size

        if self.embedding_dim != img_dim:
            raise ValueError(
                f"ViT hidden size ({self.embedding_dim}) must match img_dim ({img_dim})."
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 2),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        radiomics_features: torch.Tensor,
        tumor_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return binary class logits with shape ``[batch_size, 2]``."""
        radiomics_token = self.radiomics_projector(radiomics_features)
        tokens = torch.stack(
            (image_features, radiomics_token, tumor_features),
            dim=1,
        )

        encoded_tokens = self.token_encoder(tokens).last_hidden_state
        fused_features = (
            encoded_tokens[:, 0, :]
            + encoded_tokens[:, 1, :]
            + encoded_tokens[:, 2, :]
        )
        refined_features = self.refinement_encoder(
            fused_features.unsqueeze(1)
        ).last_hidden_state.squeeze(1)

        return self.classifier(refined_features)
