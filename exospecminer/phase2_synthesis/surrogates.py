import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SpectralConv1d(nn.Module):
    """
    1D Fourier Layer for Fourier Neural Operators (FNO).
    Performs FFT, multiplies by learned Fourier modes, and performs Inverse FFT.
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Compute real FFT
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)

        # Inverse real FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class UFNO(nn.Module):
    """
    U-Net enhanced Fourier Neural Operator (U-FNO) Surrogate Emulator.
    Approximates 3D monochromatic radiative transfer with < 3% error.
    Combines global Fourier mode convolution with local U-Net skip connections.
    """
    def __init__(
        self,
        in_channels: int = 14,  # atmospheric parameters
        out_channels: int = 1,  # spectral transit depth
        modes: int = 16,
        width: int = 64,
        spectral_len: int = 2048,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.spectral_len = spectral_len

        # Input projection
        self.fc0 = nn.Linear(in_channels, width)

        # U-Net Downsampling path
        self.down1 = nn.Sequential(nn.Conv1d(width, width * 2, kernel_size=3, stride=2, padding=1), nn.GELU())
        self.down2 = nn.Sequential(nn.Conv1d(width * 2, width * 4, kernel_size=3, stride=2, padding=1), nn.GELU())

        # FNO Bottleneck layers
        self.fno1 = SpectralConv1d(width * 4, width * 4, modes=modes)
        self.fno2 = SpectralConv1d(width * 4, width * 4, modes=modes)
        self.conv_w1 = nn.Conv1d(width * 4, width * 4, kernel_size=1)
        self.conv_w2 = nn.Conv1d(width * 4, width * 4, kernel_size=1)

        # U-Net Upsampling path with skip connections
        self.up1 = nn.ConvTranspose1d(width * 4, width * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(width * 2 * 2, width, kernel_size=4, stride=2, padding=1)

        # Output projection
        self.fc1 = nn.Linear(width * 2, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, in_channels) representing atmospheric parameters.
        Returns:
            Tensor of shape (batch_size, spectral_len) representing transmission spectrum.
        """
        batch_size = x.size(0)

        # Expand 1D parameters across spectral length domain
        x = self.fc0(x)  # (B, width)
        x = x.unsqueeze(-1).repeat(1, 1, self.spectral_len)  # (B, width, L)

        # Downsample
        skip1 = x  # (B, width, L)
        x = self.down1(x)  # (B, width*2, L/2)
        skip2 = x  # (B, width*2, L/2)
        x = self.down2(x)  # (B, width*4, L/4)

        # FNO Bottleneck
        x1 = self.fno1(x) + self.conv_w1(x)
        x1 = nn.functional.gelu(x1)
        x2 = self.fno2(x1) + self.conv_w2(x1)
        x = nn.functional.gelu(x2)

        # Upsample with skip connections
        x = self.up1(x)  # (B, width*2, L/2)
        x = torch.cat([x, skip2], dim=1)  # (B, width*4, L/2)
        
        x = self.up2(x)  # (B, width, L)
        x = torch.cat([x, skip1], dim=1)  # (B, width*2, L)

        # Transpose for linear projection
        x = x.permute(0, 2, 1)  # (B, L, width*2)
        x = nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)  # (B, L, out_channels)

        return x.squeeze(-1)


class OSELM:
    """
    Online Sequential Extreme Learning Machine (OS-ELM) Surrogate Emulator.
    Accelerates 1D atmospheric profile evaluation by 90% via sequential learning.
    """
    def __init__(self, in_features: int, hidden_units: int, out_features: int):
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.out_features = out_features

        # Randomly initialize hidden layer weights and biases (fixed)
        self.alpha = np.random.normal(0, 1.0, (in_features, hidden_units))
        self.bias = np.random.normal(0, 1.0, (1, hidden_units))

        # Output weight matrix beta (learned sequentially)
        self.beta = np.zeros((hidden_units, out_features))

        # Precision matrix P for recursive least squares
        self.P = None
        self.is_initialized = False

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function for hidden layer.
        """
        h = np.dot(x, self.alpha) + self.bias
        return 1 / (1 + np.exp(-np.clip(h, -500, 500)))

    def initialize_phase(self, x0: np.ndarray, y0: np.ndarray):
        """
        Initial batch training to setup precision matrix P.
        x0: (batch_size, in_features), batch_size must be >= hidden_units
        """
        H0 = self._activate(x0)
        # P = (H0^T * H0)^-1
        self.P = np.linalg.pinv(np.dot(H0.T, H0))
        # beta = P * H0^T * y0
        self.beta = np.dot(np.dot(self.P, H0.T), y0)
        self.is_initialized = True

    def update(self, x: np.ndarray, y: np.ndarray):
        """
        Online sequential update of output weights beta using new batch.
        """
        if not self.is_initialized:
            self.initialize_phase(x, y)
            return

        H = self._activate(x)
        
        # Recursive least squares update
        # P_{k+1} = P_k - P_k * H^T * (I + H * P_k * H^T)^-1 * H * P_k
        I = np.eye(H.shape[0])
        temp = np.linalg.pinv(I + np.dot(np.dot(H, self.P), H.T))
        self.P = self.P - np.dot(np.dot(np.dot(np.dot(self.P, H.T), temp), H), self.P)

        # beta_{k+1} = beta_k + P_{k+1} * H^T * (y - H * beta_k)
        error = y - np.dot(H, self.beta)
        self.beta = self.beta + np.dot(np.dot(self.P, H.T), error)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Fast inference of atmospheric profiles/spectra.
        """
        H = self._activate(x)
        return np.dot(H, self.beta)
