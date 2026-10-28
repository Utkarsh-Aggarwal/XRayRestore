#!/usr/bin/env bash
# Needs ~/.kaggle/kaggle.json placed by YOU (kaggle.com -> Settings -> API). Never paste keys anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
kaggle datasets download -d nih-chest-xrays/sample -p data --unzip
echo "Downloaded to data/. Expected: data/sample/images/*.png and data/sample_labels.csv"
