# Demo (offline)

Streamlit app showing what the world model actually does: it rolls the learned
dynamics **K steps ahead** at every moment of a capture and plots the resulting
infiltration-probability timeline, alongside the predicted MITRE stage and a
plain-English attribution for any moment you inspect.

Runs fully offline — no cloud APIs, no network calls.

## Run

```bash
# one-time, from the repo root (CPU-only torch keeps the install small)
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install streamlit pandas numpy scikit-learn

.venv/bin/streamlit run demo/app.py
```

It expects a trained checkpoint at `checkpoints/world_model_best.pt` (produced
by `training/train.py`); the path is editable in the sidebar.

## Traffic sources

**Built-in synthetic attack (default).** Generates a capture that walks through
benign → reconnaissance → brute-force → C2 → benign. This is the default on
purpose: a live demo shouldn't depend on a 6.5 GB dataset being present on the
machine, and the scripted progression makes the forecast's behaviour legible in
a single screen.

**Upload a CICFlowMeter CSV.** Labels are optional — real traffic being scored
at inference time has no ground-truth column. If labels *are* present they're
overlaid on the timeline so the forecast can be read against what actually
happened.

## Reading the screen

- **Forecast timeline** — the model's K-step-ahead infiltration probability at
  each moment. The claim being demonstrated is that this rises *before* the
  attack fully develops, not that it labels an attack already in progress.
- **Predicted stage** — worst kill-chain stage anywhere in the forecast horizon.
- **Why this forecast?** — integrated-gradients feature attribution plus the
  transformer's temporal attention. If the attribution's completeness error is
  large the app says so, rather than presenting an unfaithful explanation as
  fact.

## Honest framing for questions

The shipped checkpoint is precision-first: **94.6% precision, 0.44% FPR, but
0.252 recall** on CIC-IDS-2018. It rarely cries wolf and it beats a persistence
baseline on next-state prediction by 37.7% — but it misses roughly three of
every four attack windows. Don't demo it as a complete detector; demo it as a
high-confidence early-warning signal. See `benchmark/results.json`.
