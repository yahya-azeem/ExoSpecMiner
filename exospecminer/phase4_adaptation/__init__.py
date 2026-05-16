"""
Phase 4: Systematics Mitigation and Domain Adaptation.

Contains Domain Adversarial Neural Network (DANN) architecture with Gradient Reversal Layer,
and Contrastive Unpaired Translation (CUT) for spectrum-to-spectrum domain translation.
"""

from .dann import GradientReversalLayer, DANNTransformer, train_dann
from .cut import ResNetGenerator1D, PatchDiscriminator1D, PatchSampleF, PatchNCELoss, CUTModel, train_cut

__all__ = [
    "GradientReversalLayer",
    "DANNTransformer",
    "train_dann",
    "ResNetGenerator1D",
    "PatchDiscriminator1D",
    "PatchSampleF",
    "PatchNCELoss",
    "CUTModel",
    "train_cut",
]
