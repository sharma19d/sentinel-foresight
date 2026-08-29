# SENTINEL Foresight
### A World Model for Predictive Cyber Defence — SIH 2026 · PS 26153

Most intrusion detectors classify each network flow in isolation as benign or
malicious, discarding the *temporal, causal structure* of an attack. **Foresight**
instead learns a **world model** of the network — the state-transition dynamics
`P(S_t+1 | S_t)` — and rolls it forward K steps to forecast attacker progression
**before the kill chain completes.**

> **Read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for the full picture** — the
> concept, architecture, measured results, known limitations, and how to pick the
> work back up. This README is the short version.

---

## Status — what is actually built

| Component | State |
|---|---|
| Feature pipeline + 22-feature network state `S_t` | ✅ built |
| CIC-IDS-2018 loader (hardened against real-world quirks) | ✅ built |
| World model — Transformer/LSTM, next-state + stage heads | ✅ built |
| K-step rollout / forecast timeline | ✅ built |
| Training pipeline (trained on all 10 days, 4M flows) | ✅ trained |
| Explainability — integrated gradients + attention | ✅ built |
| Benchmark vs persistence / logistic-regression / majority | ✅ built |
| Offline Streamlit demo (bundled real capture) | ✅ built |
| Out-of-distribution input guard | ✅ built |
| PCAP ingest (scapy → flows, CICFlowMeter-compatible) | ✅ built |
| Regression tests (`tests/`) | ✅ built |
| Slides / demo video / screenshots | ❌ not started |

---

## Measured results

Trained on 4M flows from CIC-IDS-2018; validation is the **last 3 capture days**,
held out temporally (never a random split). Raw numbers: `benchmark/results.json`.

**Dynamics — did it learn `P(S_t+1 | S_t)`?**

| | next-state MSE |
|---|---|
| Persistence baseline (`S_t+1 = S_t`) | 0.902 |
| **World model** | **0.562 (−37.7%)** |

This is the result that earns the name "world model": it predicts the network's
next state materially better than assuming nothing changes.

**Detection — infiltration windows (23.4% prevalence)**

| model | precision | recall | F1 | FPR | AUC |
|---|---|---|---|---|---|
| Majority (always benign) | 0.000 | 0.000 | 0.000 | 0.0000 | n/a |
| Best single raw feature (`n_flows`) | 0.571 | 0.149 | 0.236 | 0.0343 | 0.811 |
| Logistic regression (same features) | 0.420 | 0.018 | 0.034 | 0.0075 | 0.737 |
| **World model** | **0.946** | **0.252** | **0.398** | **0.0044** | **0.839** |

The world model wins on every metric, but **be careful which margin you quote.**
Against logistic regression the F1 gap looks enormous (12×) — that flatters us,
because that baseline scores poorly at a fixed 0.5 threshold. The honest
comparison is the strongest trivial baseline, one raw feature: there the AUC
margin is modest (**0.839 vs 0.811**). Where the model wins decisively is
operational: **8× lower false-positive rate (0.44% vs 3.4%) at 1.7× the
precision** — which for an appliance that auto-blocks is the number that
matters.

### ⚠ The caveat that must travel with those numbers

Aggregate F1 hides a large capability split. Forecast AUC per held-out day:

| day | attack type | model AUC | best raw feature |
|---|---|---|---|
| 2018-03-02 | **Bot / C2** | 0.895 | 0.970 (`ack`) |
| 2018-02-28 | Infiltration | 0.735 | 0.715 (`n_unique_dst_ports`) |
| 2018-03-01 | Infiltration | **0.460** | 0.690 (`n_flows`) |

Two things to be straight about:

**Infiltration is barely detected.** On 03-01 the model is *below chance* —
attack windows score lower risk than benign ones. Infiltration is the
acknowledged-hardest CIC-IDS-2018 class (largely normal-looking traffic from an
already-trusted host), but the aggregate number is carried by the Bot day.

**Within a single day, a raw feature can beat the model** (two of three days
above). Pooled across days the model wins (0.839 vs 0.811) — its scores stay
comparable across days, whereas a raw feature's scale shifts between them. That
cross-day stability is a real property worth claiming; per-day dominance is not.

Overall recall is also low (0.252): this is a **high-precision, low-noise
early-warning signal**, not a complete detector. Present it that way.

*(Checked and ruled out: the per-day numbers are not a temporal-alignment
artefact. Re-scoring the forecast against future truth (t+1…t+K) rather than
present truth moved 03-01 only 0.466 → 0.460.)*

---

## Quick start

```bash
git clone <this repo> && cd sentinel-foresight

# CPU-only torch keeps the install small
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install streamlit pandas numpy scikit-learn scapy

.venv/bin/streamlit run demo/app.py          # demo
.venv/bin/python tests/test_pcap_ingest.py   # tests (no pytest needed)
.venv/bin/python tests/test_demo_smoke.py
```

The demo accepts a **CICFlowMeter CSV or a raw PCAP** (reassembled to flows
locally with scapy, following CICFlowMeter's feature definitions so the model
sees what it was trained on).

The demo bundles a real 80-minute slice of held-out CIC-IDS-2018 traffic
(`demo/sample_capture.csv.gz`, 1.9 MB), so it runs with **no dataset download**.
It needs a checkpoint at `checkpoints/world_model_best.pt` — see
`training/README.md` to train one, or `PROJECT_CONTEXT.md` for the exact command.

---

## Layout

```
foresight/
  features/   flow + packet feature extraction  (ported from SENTINEL)
  data/       state.py (S_t definition) · cicids.py (CSV) · pcap.py (PCAP) · drift.py (OOD guard) · synth.py
  model/      world_model.py — P(S_t+1 | S_t), next-state + stage heads
  rollout/    K-step forward simulation, batched
  explain/    integrated gradients + temporal attention
  mitre.py    kill-chain stage vocabulary + dataset-label mapping
training/     train.py — CLI trainer (best-val-F1 checkpointing)
benchmark/    baseline.py + results.json
demo/         app.py + bundled real sample capture
tests/        PCAP ingest + CSV/PCAP equivalence + demo smoke tests
checkpoints/  trained weights (gitignored)
```

## Relationship to SENTINEL

Foresight reuses SENTINEL's battle-tested feature-extraction code (flow parsing,
Kitsune-style timing stats, TCP-flag decoding, MITRE vocabulary) but is a
**distinct system**: SENTINEL is a real-time detection-and-response appliance;
Foresight is a predictive world model trained offline on datasets. Kept separate
so neither story muddies the other.
