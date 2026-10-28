"""Train the residual U-Net. Best checkpoint chosen by validation SSIM."""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import XrayDataset  # noqa: E402
from src.losses import RestorationLoss, ssim  # noqa: E402
from src.model import ResidualUNet  # noqa: E402
from src.noise import fixed_noisy, simulate_low_dose  # noqa: E402


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ablation", choices=["l1_only", "l1_ssim"])
    ap.add_argument("--out", default="checkpoints/best.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed = cfg["seed"]
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    lw, epochs = dict(cfg["loss"]), cfg["train"]["epochs"]
    if args.ablation:
        ab = dict(cfg["ablations"][args.ablation])
        epochs = ab.pop("epochs")
        lw = ab
    dev = device()
    tr = DataLoader(XrayDataset(cfg["splits"], "train", cfg["image_size"], augment=True),
                    batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"])
    va = XrayDataset(cfg["splits"], "val", cfg["image_size"])
    model = ResidualUNet(cfg["model"]["channels"], cfg["model"]["norm"]).to(dev)
    crit = RestorationLoss(**lw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr))
    best = -1.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        model.train()
        for x, _ in tqdm(tr, desc=f"epoch {ep + 1}/{epochs}", leave=False):
            dose = float(np.random.choice(cfg["doses"]))   # random dose per batch = augmentation
            y = simulate_low_dose(x, dose, cfg["sigma_e"])
            x, y = x.to(dev), y.to(dev)
            loss = crit(model(y), x)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(min(len(va), 300)):
                x, idx = va[i]
                y = fixed_noisy(x, cfg["doses"][-1], cfg["sigma_e"], idx, seed)
                scores.append(float(ssim(model(y[None].to(dev)), x[None].to(dev))))
        v = float(np.mean(scores))
        print(f"epoch {ep + 1}: loss {loss.item():.4f}  val SSIM@{cfg['doses'][-1]} {v:.4f}")
        if v > best:
            best = v
            torch.save({"model": model.state_dict(), "config": cfg, "loss_weights": lw, "val_ssim": v}, args.out)


if __name__ == "__main__":
    main()
