"""80/10/10 split by PATIENT ID (never by image), saved to data/splits.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
csv = next(p for p in [root / "sample_labels.csv", root / "sample" / "sample_labels.csv", root / "Data_Entry_2017.csv"] if p.exists())
df = pd.read_csv(csv)
imgs = {p.name: p for p in root.rglob("*.png")}
df = df[df["Image Index"].isin(imgs)].copy()
df["path"] = df["Image Index"].map(lambda n: str(imgs[n]))
patients = df["Patient ID"].unique()
rng = np.random.default_rng(0)
rng.shuffle(patients)
n = len(patients)
split = {p: "train" for p in patients[: int(.8 * n)]}
split.update({p: "val" for p in patients[int(.8 * n): int(.9 * n)]})
split.update({p: "test" for p in patients[int(.9 * n):]})
df["split"] = df["Patient ID"].map(split)
df[["Image Index", "Patient ID", "Finding Labels", "path", "split"]].to_csv(root / "splits.csv", index=False)
print(df.split.value_counts().to_string())
