import torch

from src.noise import fixed_noisy, simulate_low_dose


def test_shape_and_range():
    x = torch.rand(2, 1, 64, 64)
    y = simulate_low_dose(x, 5000, 0.01)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_lower_dose_is_noisier():
    x = torch.full((1, 1, 128, 128), 0.5)
    g = lambda: torch.Generator().manual_seed(0)
    hi = (simulate_low_dose(x, 1e4, 0.01, g()) - x).var()
    lo = (simulate_low_dose(x, 2e3, 0.01, g()) - x).var()
    assert lo > hi


def test_fixed_noise_is_deterministic():
    x = torch.rand(1, 32, 32)
    assert torch.equal(fixed_noisy(x, 2000, 0.01, 7), fixed_noisy(x, 2000, 0.01, 7))
