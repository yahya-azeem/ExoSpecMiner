import torch
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from .exojax_forward import PriorSampler, ExoJAXForwardSimulator


class BayesianAdaptiveExplorer:
    """
    Bayesian Adaptive Exploration (BAE) Sampling Strategy.
    Transforms data synthesis into an active learning paradigm.
    Sequentially queries the forward simulator in high-uncertainty/degenerate regions of parameter space.
    """
    def __init__(
        self,
        forward_simulator: ExoJAXForwardSimulator,
        prior_sampler: PriorSampler,
        candidate_pool_size: int = 5000,
        acquisition_batch_size: int = 100,
    ):
        self.forward_simulator = forward_simulator
        self.prior_sampler = prior_sampler
        self.candidate_pool_size = candidate_pool_size
        self.acquisition_batch_size = acquisition_batch_size

    def evaluate_uncertainty(
        self,
        surrogate_model: torch.nn.Module,
        candidate_params: torch.Tensor,
        num_mc_dropout: int = 10,
    ) -> torch.Tensor:
        """
        Estimates epistemic uncertainty of the surrogate model using Monte Carlo Dropout.
        Args:
            surrogate_model: PyTorch model with dropout enabled.
            candidate_params: Tensor of shape (pool_size, num_params)
        Returns:
            uncertainty_scores: Tensor of shape (pool_size,) representing variance across MC dropouts.
        """
        surrogate_model.train()  # Enable dropout for MC sampling
        device = candidate_params.device

        preds = []
        with torch.no_grad():
            for _ in range(num_mc_dropout):
                pred = surrogate_model(candidate_params)  # (pool_size, spectral_len)
                preds.append(pred.unsqueeze(0))

        preds = torch.cat(preds, dim=0)  # (num_mc, pool_size, spectral_len)
        
        # Calculate variance across MC samples, then mean across spectral domain
        variance = torch.var(preds, dim=0)  # (pool_size, spectral_len)
        uncertainty_scores = torch.mean(variance, dim=1)  # (pool_size,)

        return uncertainty_scores

    def explore(
        self,
        surrogate_model: torch.nn.Module,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes one active learning acquisition iteration.
        1. Samples a large pool of candidate parameters from prior distributions.
        2. Evaluates surrogate uncertainty on candidates.
        3. Selects top candidates with highest uncertainty.
        4. Queries high-fidelity forward simulator for exact spectra.
        Returns:
            acquired_params: (batch_size, num_params)
            acquired_spectra: (batch_size, spectral_len)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Sample candidate pool from prior
        param_names = self.prior_sampler.get_parameter_names()
        raw_samples = self.prior_sampler.sample(self.candidate_pool_size)

        # Pack into 2D NumPy array
        candidate_matrix = np.column_stack([raw_samples[name] for name in param_names])
        candidate_tensor = torch.tensor(candidate_matrix, dtype=torch.float32, device=device)

        # 2. Evaluate epistemic uncertainty
        uncertainties = self.evaluate_uncertainty(surrogate_model, candidate_tensor)
        uncertainties_np = uncertainties.cpu().numpy()

        # 3. Select top acquisition_batch_size candidates
        top_indices = np.argsort(uncertainties_np)[::-1][:self.acquisition_batch_size]
        selected_candidates = candidate_matrix[top_indices]

        # 4. Query high-fidelity forward simulator
        acquired_spectra = []
        for i in range(self.acquisition_batch_size):
            param_dict = {name: selected_candidates[i, j] for j, name in enumerate(param_names)}
            spectrum = self.forward_simulator.simulate(param_dict)
            acquired_spectra.append(spectrum)

        acquired_spectra_np = np.array(acquired_spectra)

        return selected_candidates, acquired_spectra_np
