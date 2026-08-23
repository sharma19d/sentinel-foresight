# Training (Colab)

World-model training scripts. Runs on a free Colab T4 GPU; checkpoints to Google
Drive so a session disconnect doesn't lose progress.

**Plan (to build):**
- `train_world_model.py` — supervised dynamics learning: predict S_t+1 from a
  window of S_t..S_t-w, on CIC-IDS-2018 / CTU-13 attack-timeline labels.
- `config.yaml` — reproducible training config (window size, K, model dims, lr).
- checkpoints + ONNX export for offline inference.
