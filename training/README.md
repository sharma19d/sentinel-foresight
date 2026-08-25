# Training (Colab)

The world model trains here on a free Colab T4 GPU. Inference/demo runs on your
laptop afterwards — only training needs the GPU.

## Quickstart

```python
# 1. get the code
!git clone <YOUR_REPO_URL> foresight && cd foresight
!pip install -q torch scikit-learn pandas numpy

# 2. smoke test — synthetic attack traffic, ~10 seconds, proves the pipeline
!python training/train.py

# 3. (optional) mount Drive so checkpoints survive disconnects
from google.colab import drive; drive.mount('/content/drive')

# 4. real training on CIC-IDS-2018 (put CSVs in data/cicids/)
!python training/train.py \
    --data "data/cicids/*.csv" \
    --nrows 400000 \
    --epochs 30 \
    --out /content/drive/MyDrive/foresight_ckpt
```

`--nrows` caps rows per CSV so the run fits a free session; raise it if you have
GPU headroom. Use `--encoder lstm` to train the recurrent baseline variant.

## What you get
A portable `world_model.pt` (weights + scaler + feature list + config) and
`train_config.json`. Download `world_model.pt` to your laptop — the offline
Streamlit demo loads it with no retraining.

## Metrics printed each epoch
- `dyn_mse` — next-state prediction error (the world-model core)
- `stage_acc` — kill-chain stage accuracy
- `infil_F1` — infiltration detection F1 (stage ≥ Initial Access)
