# ExoSpecMiner: Fast ML Retrieval for JWST Exoplanet Atmospheres

ExoSpecMiner is a modular, high-performance machine learning framework for fast, uncertainty-aware atmospheric retrieval of exoplanets from James Webb Space Telescope (JWST) transmission spectra.

## Key Architecture & Phases

1. **Phase 1: Spectral Representation (Foundation Models)**
   - 1D Vision Transformer Encoder with sinusoidal positional encoding.
   - Masked Autoencoder (MAE) self-supervised pre-training (75% masking).
   - Dual-head fine-tuning for molecular classification and physical parameter regression.

2. **Phase 2: High-Fidelity Data Synthesis & Active Learning**
   - Auto-differentiable forward modeling via ExoJAX 2 (JAX/NumPy backend).
   - Surrogate emulators: U-FNO (Fourier Neural Operator with U-Net skip connections) and OS-ELM.
   - Bayesian Adaptive Exploration (BAE) active learning for high-uncertainty query sampling.

3. **Phase 3: Flow Matching Corrected Posterior Estimation (FMCPE)**
   - Continuous Normalizing Flows (CNFs) using Neural ODE solvers.
   - Conditional flow matching matching neural vector fields to linear velocity trajectories.
   - Dynamic flow correction to bridge the simulation-to-reality gap using empirical calibration targets.

4. **Phase 4: Systematics Mitigation & Domain Adaptation**
   - Domain Adversarial Neural Networks (DANN) with Gradient Reversal Layers (GRL).
   - Contrastive Unpaired Translation (CUT) with PatchNCE loss for spectrum-to-spectrum translation.

5. **Phase 5: Distributed HPC Infrastructure**
   - Ray cluster management symmetrically deployed on top of SLURM schedulers for NVIDIA H200/A100 GPUs.

## Verification & Smoke Testing
- **Automated Benchmarks**: `python verify_exospecminer.py` (evaluates MAE, RMSE, R², 95% credible interval coverage, C2ST, and < 5s inference speed).
- **Local Smoke Test**: `python run_local_smoke_test.py` (executes the complete downscaling playbook: mini-Transformer overfitting, 32-bit JAX, BAE loops, mock CNF/DANN/CUT corrections, and Ray local debugging).
