#!/usr/bin/env bash
# train -> evaluate -> ablations -> figures, end to end. Run scripts/download_data.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
[ -f data/splits.csv ] || python scripts/make_splits.py data
python src/train.py
python src/evaluate.py
for ab in l1_only l1_ssim; do
  python src/train.py --ablation $ab --out checkpoints/$ab.pt
  python src/evaluate.py --ckpt checkpoints/$ab.pt --tag $ab --methods unet
done
python src/figures.py
