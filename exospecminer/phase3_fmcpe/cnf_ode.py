import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


class VectorFieldNet(nn.Module):
    """
    Time-dependent Neural Vector Field v_t(x, c) for Continuous Normalizing Flows.
    Defines smooth trajectories in parameter space conditioned on JWST spectral context.
    """
    def __init__(
        self,
        param_dim: int = 14,
        context_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.context_dim = context_dim

        # Time embedding layer
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Input projection (parameters + context)
        self.input_proj = nn.Linear(param_dim + context_dim, hidden_dim)

        # ResNet blocks for vector field estimation
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        self.blocks = nn.ModuleList(layers)

        # Output projection to velocity vector
        self.output_proj = nn.Linear(hidden_dim, param_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero init output projection for identity flow initialization
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Scalar or tensor of shape (batch_size, 1) representing time in [0, 1].
            x: Tensor of shape (batch_size, param_dim) representing atmospheric parameters.
            context: Tensor of shape (batch_size, context_dim) representing spectral embedding.
        Returns:
            velocity: Tensor of shape (batch_size, param_dim) representing dx/dt.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.size(0), 1)
        elif t.size(0) != x.size(0):
            t = t.expand(x.size(0), 1)

        t_emb = self.time_embed(t)
        xc = torch.cat([x, context], dim=-1)
        h = nn.functional.gelu(self.input_proj(xc) + t_emb)

        for block in self.blocks:
            h = h + block(h)

        velocity = self.output_proj(h)
        return velocity


class ODEWrapper(nn.Module):
    """
    Wrapper to adapt VectorFieldNet for torchdiffeq/custom ODE solvers.
    """
    def __init__(self, net: VectorFieldNet, context: torch.Tensor):
        super().__init__()
        self.net = net
        self.context = context

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(t, x, self.context)


class ContinuousNormalizingFlow(nn.Module):
    """
    Continuous Normalizing Flow (CNF) using Neural ODE Solvers.
    Integrates the time-dependent vector field to transport base distribution samples to the posterior.
    """
    def __init__(self, vector_field: VectorFieldNet, solver: str = "rk4", num_steps: int = 50):
        super().__init__()
        self.vector_field = vector_field
        self.solver = solver
        self.num_steps = num_steps

    def _odeint_custom(self, odefunc: nn.Module, x0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """
        Custom RK4 ODE solver fallback.
        """
        dt = (t_span[1] - t_span[0]) / self.num_steps
        x = x0
        t = t_span[0]

        for _ in range(self.num_steps):
            k1 = odefunc(t, x)
            k2 = odefunc(t + dt / 2, x + k1 * dt / 2)
            k3 = odefunc(t + dt / 2, x + k2 * dt / 2)
            k4 = odefunc(t + dt, x + k3 * dt)

            x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt

        return x

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        """
        Draws posterior samples by integrating ODE from t=0 to t=1.
        Args:
            num_samples: Number of samples to draw.
            context: Tensor of shape (batch_size, context_dim) representing observed spectrum embedding.
        Returns:
            posterior_samples: Tensor of shape (batch_size, num_samples, param_dim)
        """
        batch_size = context.size(0)
        param_dim = self.vector_field.param_dim
        device = context.device

        # Expand context for num_samples
        # context: (B, context_dim) -> (B * num_samples, context_dim)
        expanded_context = context.repeat_interleave(num_samples, dim=0)

        # Base distribution x0 ~ N(0, I)
        x0 = torch.randn(batch_size * num_samples, param_dim, device=device)

        # ODE integration span t in [0, 1]
        t_span = torch.tensor([0.0, 1.0], device=device)

        odefunc = ODEWrapper(self.vector_field, expanded_context)

        if TORCHDIFFEQ_AVAILABLE and self.solver in ["rk4", "dopri5", "euler"]:
            # torchdiffeq returns shape (len(t_span), B*num_samples, param_dim)
            traj = odeint(odefunc, x0, t_span, method=self.solver, options={"step_size": 1.0 / self.num_steps})
            x1 = traj[-1]
        else:
            x1 = self._odeint_custom(odefunc, x0, t_span)

        posterior_samples = x1.view(batch_size, num_samples, param_dim)
        return posterior_samples
