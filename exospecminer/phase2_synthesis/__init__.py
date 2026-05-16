"""
Phase 2: High-Fidelity Data Synthesis and Bayesian Adaptive Exploration.

Contains the ExoJAX 2 forward simulator wrapper, prior distributions (Table 1),
U-FNO and OS-ELM surrogate emulators, and Bayesian Adaptive Exploration (BAE)
sampling strategy.
"""

from .exojax_forward import PriorSampler, ExoJAXForwardSimulator
from .surrogates import UFNO, OSELM
from .bae_sampling import BayesianAdaptiveExplorer

__all__ = [
    "PriorSampler",
    "ExoJAXForwardSimulator",
    "UFNO",
    "OSELM",
    "BayesianAdaptiveExplorer",
]
