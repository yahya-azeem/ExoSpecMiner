import torch
import torch.nn as nn
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from .cnf_ode import ContinuousNormalizingFlow, VectorFieldNet


class FMCPE(nn.Module):
    """
    Flow Matching Corrected Posterior Estimation (FMCPE) Architecture.
    Combines Transformer foundation model encoder with Continuous Normalizing Flows
    to resolve physical degeneracies and provide calibrated posterior distributions.
    """
    def __init__(self, encoder: nn.Module, cnf: ContinuousNormalizingFlow):
        super().__init__()
        self.encoder = encoder
        self.cnf = cnf

    def forward(self, spectra: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
        """
        Performs simulation-based inference (SBI) to estimate atmospheric posteriors.
        Args:
            spectra: Tensor of shape (batch_size, spectral_len) representing observed JWST spectra.
            num_samples: Number of posterior samples to draw per spectrum.
        Returns:
            posterior_samples: Tensor of shape (batch_size, num_samples, param_dim)
        """
        # Extract foundational context embedding from Transformer encoder
        context = self.encoder(spectra)

        # Draw posterior samples via Neural ODE integration
        posterior_samples = self.cnf.sample(num_samples, context)
        return posterior_samples


def train_fmcpe(
    model: FMCPE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Trains the Flow Matching Corrected Posterior Estimation model using conditional flow matching.
    Matches the neural vector field v_t(x) to the conditional linear velocity u_t(x|x1) = x1 - x0.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.cnf.vector_field.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [FMCPE Train]")
        for spectra, target_params in progress_bar:
            spectra = spectra.to(device)
            x1 = target_params.to(device)  # True atmospheric parameters
            batch_size = spectra.size(0)

            optimizer.zero_grad()

            # Extract spectral context embedding
            with torch.no_grad():
                context = model.encoder(spectra)

            # Sample base distribution x0 ~ N(0, I)
            x0 = torch.randn_like(x1)

            # Sample time t ~ U(0, 1)
            t = torch.rand(batch_size, 1, device=device)

            # Conditional linear interpolation x_t = t * x1 + (1 - t) * x0
            x_t = t * x1 + (1 - t) * x0

            # Conditional linear velocity u_t(x|x1) = x1 - x0
            u_t = x1 - x0

            # Predict vector field velocity v_t(x_t, context)
            v_t = model.cnf.vector_field(t, x_t, context)

            # Flow matching loss: MSE between predicted velocity and true linear velocity
            loss = criterion(v_t, u_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.cnf.vector_field.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        history["train_loss"].append(total_loss / len(train_loader))

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for spectra, target_params in val_loader:
                    spectra = spectra.to(device)
                    x1 = target_params.to(device)
                    batch_size = spectra.size(0)

                    context = model.encoder(spectra)
                    x0 = torch.randn_like(x1)
                    t = torch.rand(batch_size, 1, device=device)
                    x_t = t * x1 + (1 - t) * x0
                    u_t = x1 - x0

                    v_t = model.cnf.vector_field(t, x_t, context)
                    loss = criterion(v_t, u_t)
                    val_loss += loss.item()

            history["val_loss"].append(val_loss / len(val_loader))
            print(f"Epoch {epoch+1} Val Loss: {val_loss / len(val_loader):.4f}")

    return history


def apply_dynamic_flow_correction(
    model: FMCPE,
    calibration_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 5e-5,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Executes dynamic flow correction using a restricted set of real calibration observations.
    Bridges the sim-to-real gap by fine-tuning the vector field on real benchmark targets.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.cnf.vector_field.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    history = {"calibration_loss": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(calibration_loader, desc=f"Correction Epoch {epoch+1}/{num_epochs}")
        for real_spectra, real_params in progress_bar:
            real_spectra = real_spectra.to(device)
            x1 = real_params.to(device)
            batch_size = real_spectra.size(0)

            optimizer.zero_grad()

            with torch.no_grad():
                context = model.encoder(real_spectra)

            x0 = torch.randn_like(x1)
            t = torch.rand(batch_size, 1, device=device)
            x_t = t * x1 + (1 - t) * x0
            u_t = x1 - x0

            v_t = model.cnf.vector_field(t, x_t, context)
            loss = criterion(v_t, u_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.cnf.vector_field.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        history["calibration_loss"].append(total_loss / len(calibration_loader))

    return history
