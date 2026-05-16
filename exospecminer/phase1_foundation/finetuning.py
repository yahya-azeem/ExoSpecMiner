import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm


class SpectralFinetuner(nn.Module):
    """
    Fine-tuning architecture for ExoSpecMiner Foundation Model.
    Adapts the pre-trained SpectralTransformerEncoder for downstream tasks:
    1. Multi-label molecular classification (e.g., detecting H2O, CO2, CH4, CO, NH3)
    2. Continuous atmospheric parameter regression (e.g., log(g), T_eq, metallicity, C/O)
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_molecules: int = 7,
        num_continuous_params: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.embed_dim

        # Classification Head: predicts presence of molecular species
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_molecules)
        )

        # Regression Head: predicts continuous physical parameters
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_continuous_params)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, spectra: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectra: Tensor of shape (batch_size, spectral_len)
        Returns:
            logits: (batch_size, num_molecules) classification logits
            preds: (batch_size, num_continuous_params) regression predictions
        """
        # Extract latent representation from CLS token
        latent = self.encoder(spectra)

        # Downstream predictions
        logits = self.classifier(latent)
        preds = self.regressor(latent)

        return logits, preds


def finetune_model(
    model: SpectralFinetuner,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    cls_loss_weight: float = 1.0,
    reg_loss_weight: float = 10.0,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Executes aggressive fine-tuning of the pre-trained foundation model.
    Optimizes both molecular classification and continuous parameter regression simultaneously.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.to(device)

    # Loss criteria
    cls_criterion = nn.BCEWithLogitsLoss()
    reg_criterion = nn.HuberLoss()

    # Optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_loss": [], "train_cls_loss": [], "train_reg_loss": [],
        "val_loss": [], "val_cls_loss": [], "val_reg_loss": []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cls, total_reg = 0.0, 0.0, 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for spectra, target_cls, target_reg in progress_bar:
            spectra = spectra.to(device)
            target_cls = target_cls.to(device)
            target_reg = target_reg.to(device)

            optimizer.zero_grad()

            logits, preds = model(spectra)

            cls_loss = cls_criterion(logits, target_cls)
            reg_loss = reg_criterion(preds, target_reg)
            loss = cls_loss_weight * cls_loss + reg_loss_weight * reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_reg += reg_loss.item()

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Cls": f"{cls_loss.item():.4f}",
                "Reg": f"{reg_loss.item():.4f}"
            })

        scheduler.step()

        num_batches = len(train_loader)
        history["train_loss"].append(total_loss / num_batches)
        history["train_cls_loss"].append(total_cls / num_batches)
        history["train_reg_loss"].append(total_reg / num_batches)

        # Validation pass
        if val_loader is not None:
            model.eval()
            val_loss, val_cls, val_reg = 0.0, 0.0, 0.0
            with torch.no_grad():
                for spectra, target_cls, target_reg in val_loader:
                    spectra = spectra.to(device)
                    target_cls = target_cls.to(device)
                    target_reg = target_reg.to(device)

                    logits, preds = model(spectra)

                    cls_loss = cls_criterion(logits, target_cls)
                    reg_loss = reg_criterion(preds, target_reg)
                    loss = cls_loss_weight * cls_loss + reg_loss_weight * reg_loss

                    val_loss += loss.item()
                    val_cls += cls_loss.item()
                    val_reg += reg_loss.item()

            num_val_batches = len(val_loader)
            history["val_loss"].append(val_loss / num_val_batches)
            history["val_cls_loss"].append(val_cls / num_val_batches)
            history["val_reg_loss"].append(val_reg / num_val_batches)
            print(f"Epoch {epoch+1} Val Loss: {val_loss / num_val_batches:.4f} (Cls: {val_cls / num_val_batches:.4f}, Reg: {val_reg / num_val_batches:.4f})")

    return history
