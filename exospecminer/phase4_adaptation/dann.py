import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm


class GradientReversalFunction(torch.autograd.Function):
    """
    Autograd function for Gradient Reversal Layer (GRL).
    Acts as identity during forward pass, but multiplies gradients by -alpha during backward pass.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) module for Domain Adversarial Training.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class DANNTransformer(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) Transformer Architecture.
    Structural modification of the Transformer foundation model to learn domain-invariant representations
    across synthetic training spectra and real JWST observational domains.
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_continuous_params: int = 14,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.embed_dim

        # Primary Task Regressor: predicts atmospheric parameters
        self.task_regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_continuous_params)
        )

        # Secondary Domain Classifier with GRL: predicts synthetic (0) vs real JWST (1)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(alpha=alpha),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.task_regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.domain_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, spectra: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectra: Tensor of shape (batch_size, spectral_len)
        Returns:
            task_preds: (batch_size, num_params) atmospheric parameter predictions
            domain_logits: (batch_size, 1) domain classification logits
        """
        latent = self.encoder(spectra)

        task_preds = self.task_regressor(latent)
        domain_logits = self.domain_classifier(latent)

        return task_preds, domain_logits


def train_dann(
    model: DANNTransformer,
    train_loader: DataLoader,  # Yields (spectra, params, domain_labels)
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 25,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    task_loss_weight: float = 1.0,
    domain_loss_weight: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Executes domain adversarial training to align synthetic and real JWST spectral feature distributions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    task_criterion = nn.HuberLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "train_task_loss": [], "train_domain_loss": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_task, total_domain = 0.0, 0.0, 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [DANN Train]")
        for spectra, target_params, domain_labels in progress_bar:
            spectra = spectra.to(device)
            target_params = target_params.to(device)
            domain_labels = domain_labels.to(device).float().unsqueeze(-1)

            optimizer.zero_grad()

            task_preds, domain_logits = model(spectra)

            # Task loss is only computed for synthetic samples (where true atmospheric parameters are known, domain == 0)
            synthetic_mask = (domain_labels == 0.0).squeeze(-1)
            if synthetic_mask.any():
                task_loss = task_criterion(task_preds[synthetic_mask], target_params[synthetic_mask])
            else:
                task_loss = torch.tensor(0.0, device=device)

            # Domain loss computed across all samples (synthetic vs real)
            domain_loss = domain_criterion(domain_logits, domain_labels)

            loss = task_loss_weight * task_loss + domain_loss_weight * domain_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_task += task_loss.item()
            total_domain += domain_loss.item()

            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Task": f"{task_loss.item():.4f}",
                "Domain": f"{domain_loss.item():.4f}"
            })

        scheduler.step()
        num_batches = len(train_loader)
        history["train_loss"].append(total_loss / num_batches)
        history["train_task_loss"].append(total_task / num_batches)
        history["train_domain_loss"].append(total_domain / num_batches)

    return history
