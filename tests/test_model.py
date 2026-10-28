import torch

from src.losses import RestorationLoss, ssim
from src.model import ResidualUNet


def test_forward_shape_and_range():
    m = ResidualUNet().eval()
    y = torch.rand(2, 1, 256, 256)
    out = m(y)
    assert out.shape == y.shape and out.min() >= 0 and out.max() <= 1


def test_ssim_identity_and_loss():
    x = torch.rand(1, 1, 64, 64)
    assert abs(float(ssim(x, x)) - 1) < 1e-4
    loss = RestorationLoss(1.0, 0.2, 0.0)(x, x)
    assert float(loss) < 1e-4
