import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm


class ResNetBlock1D(nn.Module):
    """
    1D ResNet Block for Spectral Generator.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(dim, dim, kernel_size=3),
            nn.InstanceNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad1d(1),
            nn.Conv1d(dim, dim, kernel_size=3),
            nn.InstanceNorm1d(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResNetGenerator1D(nn.Module):
    """
    1D ResNet Generator for Spectrum-to-Spectrum Translation.
    Translates theoretical synthetic spectra into real JWST observational domains.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, ngf: int = 64, n_blocks: int = 6):
        super().__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm1d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm1d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResNetBlock1D(ngf * mult)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose1d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm1d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]

        model += [nn.ReflectionPad1d(3), nn.Conv1d(ngf, out_channels, kernel_size=7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        out = self.model(x)
        return out.squeeze(1)


class PatchDiscriminator1D(nn.Module):
    """
    1D Patch Discriminator for distinguishing real vs translated JWST spectra.
    """
    def __init__(self, in_channels: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv1d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm1d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm1d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.model(x)


class PatchSampleF(nn.Module):
    """
    2-layer MLP projection head for extracting patch features across layers for PatchNCE loss.
    """
    def __init__(self, in_channels: int = 64, netF_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, netF_dim),
            nn.ReLU(inplace=True),
            nn.Linear(netF_dim, netF_dim)
        )

    def forward(self, feats: torch.Tensor, patch_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # feats: (B, C, L)
        feats_perm = feats.permute(0, 2, 1)  # (B, L, C)
        batch_size, num_patches, channels = feats_perm.shape

        if patch_ids is None:
            # Randomly select patches
            num_sample = min(num_patches, 256)
            patch_ids = torch.randperm(num_patches, device=feats.device)[:num_sample]

        # Sample patches
        sample_feats = feats_perm[:, patch_ids, :]  # (B, num_sample, C)
        sample_feats = sample_feats.reshape(-1, channels)  # (B * num_sample, C)
        out = self.mlp(sample_feats)

        return out, patch_ids


class PatchNCELoss(nn.Module):
    """
    PatchNCE Loss enforcing mutual information maximization between corresponding patches
    of input synthetic spectra and translated JWST spectra.
    """
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        # feat_q: (B * num_patches, C) from translated spectrum
        # feat_k: (B * num_patches, C) from input spectrum
        batch_size_num_patches = feat_q.shape[0]

        # Normalize features
        feat_q = nn.functional.normalize(feat_q, dim=-1)
        feat_k = nn.functional.normalize(feat_k, dim=-1)

        # Positives: diagonal elements of similarity matrix
        l_pos = torch.bmm(feat_q.unsqueeze(1), feat_k.unsqueeze(2)).squeeze(2)  # (B*N, 1)

        # Negatives: off-diagonal elements
        # For simplicity and memory efficiency, compare against all k features in batch
        l_neg = torch.mm(feat_q, feat_k.T)  # (B*N, B*N)
        
        # Mask out diagonal for negatives
        mask = torch.eye(batch_size_num_patches, device=feat_q.device, dtype=torch.bool)
        l_neg.masked_fill_(mask, -10.0)

        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau

        # Targets: index 0 is positive
        targets = torch.zeros(batch_size_num_patches, dtype=torch.long, device=feat_q.device)

        return self.cross_entropy(logits, targets)


class CUTModel(nn.Module):
    """
    Complete Contrastive Unpaired Translation (CUT) Model Architecture.
    """
    def __init__(self, netG: ResNetGenerator1D, netD: PatchDiscriminator1D, netF: PatchSampleF):
        super().__init__()
        self.netG = netG
        self.netD = netD
        self.netF = netF


def train_cut(
    model: CUTModel,
    unpaired_loader: DataLoader,  # Yields (synthetic_spectra, real_spectra)
    num_epochs: int = 30,
    lr: float = 2e-4,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Executes Contrastive Unpaired Translation training loop.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer_G = torch.optim.AdamW(list(model.netG.parameters()) + list(model.netF.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.AdamW(model.netD.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_GAN = nn.MSELoss()
    criterion_NCE = PatchNCELoss()

    history = {"loss_G": [], "loss_D": [], "loss_NCE": []}

    for epoch in range(num_epochs):
        model.train()
        total_G, total_D, total_NCE = 0.0, 0.0, 0.0

        progress_bar = tqdm(unpaired_loader, desc=f"Epoch {epoch+1}/{num_epochs} [CUT Train]")
        for real_A, real_B in progress_bar:  # A: synthetic, B: real JWST
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # --- Train Generator G and Projection Head F ---
            optimizer_G.zero_grad()

            fake_B = model.netG(real_A)
            pred_fake = model.netD(fake_B)

            # GAN Loss: generator wants discriminator to predict 1 (real)
            target_real = torch.ones_like(pred_fake)
            loss_G_GAN = criterion_GAN(pred_fake, target_real)

            # PatchNCE Loss: mutual information between input A and translated B
            # For simplicity in this 1D architecture, we project the final layer embeddings
            feat_A = model.netG.model[:6](real_A.unsqueeze(1) if real_A.dim() == 2 else real_A)
            feat_fake_B = model.netG.model[:6](fake_B.unsqueeze(1) if fake_B.dim() == 2 else fake_B)

            sample_A, patch_ids = model.netF(feat_A)
            sample_fake_B, _ = model.netF(feat_fake_B, patch_ids=patch_ids)

            loss_NCE = criterion_NCE(sample_fake_B, sample_A)

            loss_G = loss_G_GAN + loss_NCE
            loss_G.backward()
            optimizer_G.step()

            # --- Train Discriminator D ---
            optimizer_D.zero_grad()

            pred_real = model.netD(real_B)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            pred_fake_detach = model.netD(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake_detach, torch.zeros_like(pred_fake_detach))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            total_G += loss_G.item()
            total_D += loss_D.item()
            total_NCE += loss_NCE.item()

            progress_bar.set_postfix({
                "G": f"{loss_G.item():.4f}",
                "D": f"{loss_D.item():.4f}",
                "NCE": f"{loss_NCE.item():.4f}"
            })

        num_batches = len(unpaired_loader)
        history["loss_G"].append(total_G / num_batches)
        history["loss_D"].append(total_D / num_batches)
        history["loss_NCE"].append(total_NCE / num_batches)

    return history
