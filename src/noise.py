"""Poisson-Gaussian low-dose simulation: y = Poisson(I0 * x) / I0 + N(0, sigma_e)."""
import numpy as np
import torch


def simulate_low_dose(x, dose, sigma_e=0.01, generator=None):
    """x: tensor in [0,1]; dose: photon count I0 (lower = noisier). Returns tensor in [0,1]."""
    counts = torch.poisson(x.clamp(0, 1) * dose, generator=generator)
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * sigma_e
    return (counts / dose + noise).clamp(0, 1)


def fixed_noisy(x, dose, sigma_e, index, seed=0):
    """Deterministic noisy copy for val/test: same image + dose -> same noise for every method."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + int(index) * 31 + int(dose))
    return simulate_low_dose(x, dose, sigma_e, g)


def random_dose(doses, rng=np.random):
    return float(rng.choice(doses))
