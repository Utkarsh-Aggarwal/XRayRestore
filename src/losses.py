"""L = w_l1 * L1 + w_ssim * (1 - SSIM) + w_perc * VGG16-relu2_2 perceptual."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim(x, y, window=11, c1=0.01 ** 2, c2=0.03 ** 2):
    pad = window // 2
    mu_x, mu_y = F.avg_pool2d(x, window, 1, pad), F.avg_pool2d(y, window, 1, pad)
    sxx = F.avg_pool2d(x * x, window, 1, pad) - mu_x ** 2
    syy = F.avg_pool2d(y * y, window, 1, pad) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean()


class Perceptual(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:9].eval()  # up to relu2_2
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        f = lambda t: self.vgg((t.repeat(1, 3, 1, 1) - self.mean) / self.std)
        return F.l1_loss(f(x), f(y))


class RestorationLoss(nn.Module):
    def __init__(self, l1=1.0, ssim=0.2, perceptual=0.05):
        super().__init__()
        self.w = (l1, ssim, perceptual)
        self.perc = Perceptual() if perceptual > 0 else None

    def forward(self, pred, target):
        loss = self.w[0] * F.l1_loss(pred, target)
        if self.w[1]:
            loss = loss + self.w[1] * (1 - ssim(pred, target))
        if self.perc is not None:
            loss = loss + self.w[2] * self.perc(pred, target)
        return loss
