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

**Bundled real capture (default).** An 80-minute slice of CIC-IDS-2018,
2018-03-02 09:40–11:00 — 152k flows, 54% attack windows. Ships with the repo
(1.9 MB) so a live demo never depends on the 6.5 GB dataset being present.

Two choices worth stating plainly rather than leaving for someone to notice:
it is a **held-out** day the model never trained on, and it is the **Bot/C2**
day, where the model genuinely works (AUC 0.894) rather than an Infiltration
day, where it is at or below chance (AUC 0.466). Measured on this sample:
AUC 0.731, precision 0.94 at the 0.9 threshold.

**Synthetic generator (third option).** Walks benign → recon → brute-force →
C2 → benign. Predates the real data and its feature scales sit dozens of σ outside the
training distribution, so the model saturates near 100% on every window of it,
benign included. Kept as a test fixture, and the app raises a drift warning on
it — do not demo with this.

**Upload a raw PCAP** (`.pcap` / `.pcapng`). Packets are reassembled into flows
locally with scapy — nothing leaves the machine. The flow features follow
CICFlowMeter's definitions (payload bytes not frame length, bidirectional
5-tuple keying, 120 s flow timeout) so the model sees features that mean what
they meant in training; the app runs the drift check on the result and warns if
they don't land in-distribution.

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

The shipped checkpoint (15 s windows × 24) ranks attacks well — **AUC 0.876**
against 0.825 for the best single raw feature — and beats a persistence baseline
on next-state prediction by 24.7%. Its operating point depends heavily on the
threshold: at 0.5 it catches 94% of attack windows but fires on 47% of benign
ones; tuned for FPR ≤ 1% on train it gives precision 0.771, recall 0.279. Don't
demo it as a complete detector; demo it as an early-warning signal whose
threshold you pick for the deployment. See `benchmark/results_ws15.json`.
