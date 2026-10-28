"""comparison.png, auroc_drop.png, psnr_vs_dose.png + results/summary.md from the CSVs."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import bm3d_denoise, clahe  # noqa: E402
from src.data import XrayDataset  # noqa: E402
from src.evaluate import load_unet  # noqa: E402
from src.noise import fixed_noisy  # noqa: E402

F = Path("results/figures")
F.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(open("configs/default.yaml"))
q = pd.read_csv("results/image_quality.csv")
a = pd.read_csv("results/diagnostic_fidelity.csv")

fig, ax = plt.subplots(figsize=(6, 4))
for m, g in q.groupby("method"):
    ax.plot(g.dose, g.psnr, "o-", label=m)
ax.set_xscale("log"), ax.set_xlabel("Dose I0 (photons, log)"), ax.set_ylabel("PSNR (dB)")
ax.legend(), ax.grid(alpha=.3), fig.tight_layout(), fig.savefig(F / "psnr_vs_dose.png", dpi=150)

low = a[a.dose.astype(str) == str(cfg["doses"][-1])]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(low.method, low.auroc_drop * 100, color=["#b0b0b0", "#e0a040", "#6a9fd8", "#3b8f5a"][:len(low)])
ax.axhline(0, color="k", lw=.8)
ax.set_ylabel("AUROC drop vs clean (points)"), ax.set_title(f"Diagnostic fidelity @ I0={cfg['doses'][-1]}")
fig.tight_layout(), fig.savefig(F / "auroc_drop.png", dpi=150)

ds = XrayDataset(cfg["splits"], "test", cfg["image_size"])
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
unet = load_unet("checkpoints/best.pt", cfg, dev)
fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
for r, i in enumerate([0, len(ds) // 2]):
    x, idx = ds[i]
    c = x[0].numpy()
    n = fixed_noisy(x, cfg["doses"][-1], cfg["sigma_e"], idx, cfg["seed"])[0].numpy()
    with torch.no_grad():
        u = unet(torch.from_numpy(n)[None, None].to(dev))[0, 0].cpu().numpy()
    for k, (t, im) in enumerate([("Clean", c), ("Noisy", n), ("CLAHE", clahe(n)), ("BM3D", bm3d_denoise(n)), ("U-Net", u)]):
        crop = im[64:192, 64:192]
        axes[r, k].imshow(crop, cmap="gray", vmin=0, vmax=1)
        axes[r, k].set_title(t if k == 0 else f"{t}  {psnr(c, im, data_range=1):.1f} dB")
        axes[r, k].axis("off")
fig.tight_layout(), fig.savefig(F / "comparison.png", dpi=130)

lines = ["# Summary (auto-generated from results/*.csv)\n", "## Image quality\n", q.round(3).to_markdown(index=False),
         "\n\n## Diagnostic fidelity\n", a.round(4).to_markdown(index=False)]
for tag in ["l1_only", "l1_ssim"]:
    p = Path(f"results/diagnostic_fidelity_{tag}.csv")
    if p.exists():
        lines += [f"\n\n## Ablation: {tag}\n", pd.read_csv(p).round(4).to_markdown(index=False)]
Path("results/summary.md").write_text("\n".join(lines) + "\n")
print("figures + summary written")
