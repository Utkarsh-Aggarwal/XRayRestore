import cv2
import numpy as np


def clahe(noisy):
    """noisy: float [0,1] (H,W) -> float [0,1]."""
    u8 = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255.0


def bm3d_denoise(noisy):
    import bm3d
    from skimage.restoration import estimate_sigma
    sigma = float(estimate_sigma(noisy))
    return np.clip(bm3d.bm3d(noisy, sigma_psd=sigma), 0, 1).astype(np.float32)
