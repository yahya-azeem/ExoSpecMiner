"""
Phase 1: Spectral Representation via Transformer Foundation Models.

Contains the Transformer foundation model architecture, Masked Autoencoder (MAE)
pre-training framework, and fine-tuning routines for molecular classification
and continuous parameter regression.
"""

from .transformer import SpectralTransformerEncoder, SpectralMaskedAutoencoder
from .finetuning import SpectralFinetuner, finetune_model

__all__ = [
    "SpectralTransformerEncoder",
    "SpectralMaskedAutoencoder",
    "SpectralFinetuner",
    "finetune_model",
]
