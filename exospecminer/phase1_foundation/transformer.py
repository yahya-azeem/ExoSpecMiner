import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PositionalEncoding1D(nn.Module):
    """
    1D Sinusoidal Positional Encoding for spectral sequence tokens.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SpectralTransformerEncoder(nn.Module):
    """
    Transformer Encoder Foundation Model for 1D Exoplanet Spectra.
    Maps high-resolution spectral vectors into rich latent representations.
    """
    def __init__(
        self,
        spectral_len: int = 2048,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spectral_len = spectral_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = spectral_len // patch_size

        assert spectral_len % patch_size == 0, "spectral_len must be divisible by patch_size"

        # Patch embedding: linear projection of 1D spectral patches
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        
        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding1D(embed_dim, max_len=self.num_patches + 1)
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def patchify(self, spectra: torch.Tensor) -> torch.Tensor:
        """
        Splits 1D spectra into patches.
        Args:
            spectra: Tensor of shape (batch_size, spectral_len)
        Returns:
            Tensor of shape (batch_size, num_patches, patch_size)
        """
        batch_size = spectra.size(0)
        return spectra.view(batch_size, self.num_patches, self.patch_size)

    def forward_features(self, spectra: torch.Tensor, mask_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts features from spectra, optionally masking a subset of patches for MAE.
        Args:
            spectra: (batch_size, spectral_len)
            mask_indices: Optional boolean mask of shape (batch_size, num_patches). True means keep, False means mask.
        Returns:
            x: Encoder output tokens
            patches: Original patchified spectra
        """
        batch_size = spectra.size(0)
        patches = self.patchify(spectra)  # (B, N, P)
        x = self.patch_embed(patches)     # (B, N, D)

        # Apply positional embedding to patch tokens before masking
        # pos_embed has shape (1, N+1, D). Index 1: is for patches (0 is CLS)
        pos_embed_patches = self.pos_embed.pe[:, 1:self.num_patches + 1, :]
        x = x + pos_embed_patches

        if mask_indices is not None:
            # Gather unmasked patches
            # mask_indices shape: (B, num_keep) containing indices of patches to keep
            batch_indices = torch.arange(batch_size, device=spectra.device).unsqueeze(-1)
            x = x[batch_indices, mask_indices]

        # Append CLS token with its positional embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed.pe[:, :1, :]
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        # Pass through Transformer encoder blocks
        x = self.blocks(x)
        x = self.norm(x)

        return x, patches

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass for fine-tuning or feature extraction.
        Returns the latent representation corresponding to the CLS token.
        """
        x, _ = self.forward_features(spectra)
        return x[:, 0]  # Return CLS token representation


class SpectralMaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) Framework for Self-Supervised Pre-training on JWST Spectra.
    Reconstructs masked spectral patches to learn foundational atmospheric representations.
    """
    def __init__(
        self,
        encoder: SpectralTransformerEncoder,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.num_patches = encoder.num_patches
        self.patch_size = encoder.patch_size

        # Encoder to Decoder embedding projection
        self.enc_to_dec = nn.Linear(encoder.embed_dim, decoder_embed_dim)

        # MASK token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder positional embedding
        self.dec_pos_embed = PositionalEncoding1D(decoder_embed_dim, max_len=self.num_patches + 1)

        # Transformer Decoder blocks
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Reconstruction head: projects decoder embeddings back to patch space
        self.reconstruction_head = nn.Linear(decoder_embed_dim, self.patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.enc_to_dec.weight)
        nn.init.zeros_(self.enc_to_dec.bias)
        nn.init.xavier_uniform_(self.reconstruction_head.weight)
        nn.init.zeros_(self.reconstruction_head.bias)

    def random_masking(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates random mask for MAE pre-training.
        Returns:
            keep_indices: (B, num_keep) indices of unmasked patches
            mask_indices: (B, num_mask) indices of masked patches
            restore_indices: (B, N) indices to restore original patch order in decoder
        """
        num_keep = int(self.num_patches * (1 - self.mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(batch_size, self.num_patches, device=device)
        shuffle_indices = torch.argsort(noise, dim=1)
        restore_indices = torch.argsort(shuffle_indices, dim=1)

        keep_indices = shuffle_indices[:, :num_keep]
        mask_indices = shuffle_indices[:, num_keep:]

        return keep_indices, mask_indices, restore_indices

    def forward_encoder(self, spectra: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs encoder on unmasked patches.
        """
        batch_size = spectra.size(0)
        keep_indices, mask_indices, restore_indices = self.random_masking(batch_size, spectra.device)
        
        x, patches = self.encoder.forward_features(spectra, mask_indices=keep_indices)
        return x, patches, mask_indices, restore_indices

    def forward_decoder(self, x: torch.Tensor, restore_indices: torch.Tensor) -> torch.Tensor:
        """
        Runs decoder on full set of tokens (unmasked + mask tokens).
        """
        batch_size = x.size(0)
        
        # Project encoder features to decoder dimension
        x = self.enc_to_dec(x)

        # Separate CLS token from patch tokens
        cls_token = x[:, :1, :]
        keep_patches = x[:, 1:, :]

        # Create mask tokens for the masked positions
        num_mask = self.num_patches - keep_patches.size(1)
        mask_tokens = self.mask_token.expand(batch_size, num_mask, -1)

        # Concatenate unmasked patches and mask tokens, then unshuffle to restore original order
        full_patches = torch.cat([keep_patches, mask_tokens], dim=1)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1)
        full_patches = full_patches[batch_indices, restore_indices]

        # Re-attach CLS token and add decoder positional embedding
        x_dec = torch.cat([cls_token, full_patches], dim=1)
        x_dec = self.dec_pos_embed(x_dec)

        # Pass through decoder blocks
        x_dec = self.decoder_blocks(x_dec)
        x_dec = self.decoder_norm(x_dec)

        # Remove CLS token and project back to patch values
        reconstructed_patches = self.reconstruction_head(x_dec[:, 1:, :])
        return reconstructed_patches

    def forward(self, spectra: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full MAE forward pass.
        Args:
            spectra: (batch_size, spectral_len)
        Returns:
            loss: Mean Squared Error loss on masked patches
            reconstructed_spectra: (batch_size, spectral_len) full reconstructed spectra
        """
        x, patches, mask_indices, restore_indices = self.forward_encoder(spectra)
        reconstructed_patches = self.forward_decoder(x, restore_indices)

        # Calculate MSE loss only on masked patches
        batch_indices = torch.arange(spectra.size(0), device=spectra.device).unsqueeze(-1)
        target_masked = patches[batch_indices, mask_indices]
        pred_masked = reconstructed_patches[batch_indices, mask_indices]

        loss = torch.mean((pred_masked - target_masked) ** 2)

        # Flatten reconstructed patches back to 1D spectra shape for visualization/logging
        reconstructed_spectra = reconstructed_patches.view_as(spectra)

        return loss, reconstructed_spectra
