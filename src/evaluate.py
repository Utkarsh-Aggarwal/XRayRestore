"""Image quality (PSNR/SSIM) + diagnostic fidelity (DenseNet-121 AUROC drop) per dose and method."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import bm3d_denoise, clahe  # noqa: E402
from src.data import XrayDataset  # noqa: E402
from src.model import ResidualUNet  # noqa: E402
from src.noise import fixed_noisy  # noqa: E402


def load_unet(path, cfg, dev):
    ck = torch.load(path, map_location=dev)
    m = ResidualUNet(cfg["model"]["channels"], cfg["model"]["norm"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m


class Classifier:
    """torchxrayvision NIH DenseNet-121 with its exact preprocessing."""

    def __init__(self, dev):
        import torchxrayvision as xrv
        self.xrv, self.dev = xrv, dev
        self.model = xrv.models.DenseNet(weights="densenet121-res224-nih").to(dev).eval()
        self.resize = xrv.datasets.XRayResizer(224, engine="cv2")

    @torch.no_grad()
    def __call__(self, imgs):
        batch = []
        for im in imgs:
            x = self.xrv.datasets.normalize(im * 255.0, 255)   # -> [-1024, 1024]
            batch.append(self.resize(x[None]))
        x = torch.from_numpy(np.stack(batch)).float().to(self.dev)
        return self.model(x).cpu().numpy()


def macro_auroc(probs, labels_str, pathologies, min_pos):
    aucs = {}
    for k, name in enumerate(pathologies):
        if not name:
            continue
        y = np.array([name in s.split("|") for s in labels_str]).astype(int)
        if y.sum() >= min_pos and y.sum() < len(y):
            aucs[name] = roc_auc_score(y, probs[:, k])
    return float(np.mean(list(aucs.values()))) if aucs else float("nan"), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--methods", nargs="*", default=["noisy", "clahe", "bm3d", "unet"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ds = XrayDataset(cfg["splits"], "test", cfg["image_size"])
    unet = load_unet(args.ckpt, cfg, dev) if "unet" in args.methods else None
    clf = Classifier(dev)
    labels = ds.df["Finding Labels"].tolist()
    clean = [ds[i][0][0].numpy() for i in range(len(ds))]
    p_clean = np.concatenate([clf(clean[i:i + 32]) for i in range(0, len(clean), 32)])
    auc_clean, _ = macro_auroc(p_clean, labels, clf.model.pathologies, cfg["eval"]["min_positives"])
    q_rows, a_rows = [], [{"dose": "-", "method": "clean", "n_images": len(ds), "macro_auroc": auc_clean, "auroc_drop": 0.0}]
    bm3d_n = min(len(ds), cfg["eval"]["bm3d_max_images"])
    for dose in cfg["doses"]:
        noisy = [fixed_noisy(torch.from_numpy(c)[None], dose, cfg["sigma_e"], i, cfg["seed"])[0].numpy()
                 for i, c in enumerate(clean)]
        outs = {"noisy": noisy}
        if "clahe" in args.methods:
            outs["clahe"] = [clahe(n) for n in noisy]
        if "bm3d" in args.methods:
            outs["bm3d"] = [bm3d_denoise(n) for n in tqdm(noisy[:bm3d_n], desc=f"BM3D@{dose}")]
        if unet is not None:
            with torch.no_grad():
                outs["unet"] = [unet(torch.from_numpy(n)[None, None].to(dev))[0, 0].cpu().numpy() for n in noisy]
        for m, imgs in outs.items():
            n = len(imgs)
            q_rows.append({"dose": dose, "method": m, "n_images": n,
                           "psnr": np.mean([psnr(clean[i], imgs[i], data_range=1) for i in range(n)]),
                           "ssim": np.mean([ssim(clean[i], imgs[i], data_range=1) for i in range(n)])})
            p = np.concatenate([clf(imgs[i:i + 32]) for i in range(0, n, 32)])
            sub_clean, _ = macro_auroc(p_clean[:n], labels[:n], clf.model.pathologies, cfg["eval"]["min_positives"])
            auc, _ = macro_auroc(p, labels[:n], clf.model.pathologies, cfg["eval"]["min_positives"])
            a_rows.append({"dose": dose, "method": m, "n_images": n, "macro_auroc": auc,
                           "auroc_drop": sub_clean - auc})
            print(q_rows[-1], a_rows[-1])
    Path("results").mkdir(exist_ok=True)
    suffix = "" if args.tag == "full" else f"_{args.tag}"
    pd.DataFrame(q_rows).round(4).to_csv(f"results/image_quality{suffix}.csv", index=False)
    pd.DataFrame(a_rows).round(4).to_csv(f"results/diagnostic_fidelity{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
