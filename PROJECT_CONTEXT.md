# SENTINEL Foresight — Complete Project Context

> **Purpose of this file:** the single source of truth for the project. Read this
> and you have everything — the mission, the concept, the architecture, what is
> built, what remains, how to run it, and how to present it. Written so a person
> with **zero prior context** (future-you or a new teammate) can continue to a
> winning submission. Paste it at the start of any AI session for full context.

---

## 0. One-liner

**Foresight is a *world model* for cyber defence: it learns how a network's state
evolves over time and rolls that forward to forecast an attacker's next move —
predicting infiltration *before the kill chain completes*, with an explanation
for every prediction.**

Built for **Smart India Hackathon 2026, Problem Statement 26153** (World Models
for predictive cyber defence).

---

## 1. The mission (PS 26153)

Traditional IDS classifiers label each network flow in isolation as benign or
malicious. This throws away the **temporal, causal structure** of an attack — the
order in which ports are probed, SYN floods preceding ACK floods, the timing of
recon before lateral movement. **An infiltration is a process unfolding over
time, not a single anomalous packet.**

PS 26153 asks for a **world model**: an AI that learns the transition dynamics
`P(S_t+1 | S_t)` — given the current network state, the probability distribution
over the *next* state — enabling **forward simulation**: roll K steps ahead and
see whether the trajectory converges to compromise before the attacker finishes.

### Required deliverables (verbatim intent)
1. Represent network state as feature vectors or graphs.
2. Learn state-transition dynamics with a sequence model (LSTM/Transformer/GNN).
3. Forecast future states + estimate probability of attacker progression.
4. Map predicted behaviour to MITRE ATT&CK stages.
5. Explain predictions (attention / SHAP / feature attribution).
6. Offline demo (Streamlit/Flask/CLI) accepting PCAP or CSV.
7. Benchmark vs a logistic-regression baseline (F1, precision, recall, FPR).

Datasets: **CIC-IDS-2018** and/or **CTU-13** (open, labelled, with attack timelines).

---

## 2. The core idea — world model vs classifier (READ THIS)

This distinction is the whole project. Judges *will* probe "is this really a world
model, or a classifier in disguise?" Our answer must be airtight:

| | Classifier (what NOT to build) | World model (what we build) |
|---|---|---|
| Learns | `f(flow) → {benign, malicious}` | `P(S_t+1 | S_t)` — how state evolves |
| Output | a label for the present | a **predicted next state** + its stage |
| Time | ignores it | **is** the model |
| Superpower | none | **forward simulation** — forecast the future |

**Our model predicts the full next state vector `S_t+1` (a regression over 22
features), then reads its kill-chain stage off that predicted future.** Because it
predicts *state*, we can feed predictions back in and roll K steps ahead. That is
the defensible, non-negotiable core. The stage label is a *readout of a predicted
future state*, never a direct classification of the present.

---

## 3. System architecture (data flow)

```
 PCAP / CSV                                                     ┌────────────┐
 (CIC-IDS-2018,   ─►  FEATURE PIPELINE  ─►  STATE SERIES   ─►   │ WORLD MODEL│
  CTU-13, live)      flow + packet feats     S_0..S_T           │  P(S_t+1   │
                     (foresight/features)    (22-dim vectors    │   | S_t)   │
                                              per time window)  └─────┬──────┘
                                                                      │
        ┌─────────────────────────────────────────────────────────────┤
        ▼                          ▼                          ▼        ▼
  K-STEP ROLLOUT           STAGE READOUT              ATTENTION / SHAP  (train:
  forecast_series()        MITRE kill chain           explainability    next-state
  infiltration timeline    (Recon→…→Exfil)            top features      MSE + stage CE)
        │                          │                          │
        └──────────────► OFFLINE STREAMLIT DEMO ◄─────────────┘
                 upload → forecast timeline + stage + why + benchmark
```

**State `S_t`** = one 22-dim vector aggregating all flows in a time window Δ:
volume, flag distribution (SYN/ACK/FIN/RST/PSH/URG), SYN/ACK ratio, unique-port
and port-scan score, timing (IAT mean/var), protocol mix, fan-in/out. Defined in
`foresight/data/state.py :: STATE_FEATURES` (order is frozen — model + explainer
index by it).

---

## 4. Repository structure (file-by-file)

```
sentinel-foresight/
├── PROJECT_CONTEXT.md          ← THIS FILE (source of truth)
├── README.md                   project overview mapped to PS 26153
├── requirements.txt            torch, sklearn, shap, streamlit, scapy, pandas…
├── foresight/
│   ├── mitre.py                ✅ kill-chain stages + dataset-label → stage map
│   ├── features/
│   │   ├── traffic_models.py   ✅ ported from SENTINEL — TCP flags, flow schema
│   │   └── packet_features.py  ✅ ported from SENTINEL — Kitsune IAT/jitter stats
│   ├── data/
│   │   ├── state.py            ✅ S_t definition (22 feats) + windowing + sequences
│   │   ├── cicids.py           ✅ CIC-IDS-2018 CSV → canonical flow DataFrame
│   │   └── synth.py            ✅ synthetic attack-progression generator (testing)
│   ├── model/
│   │   └── world_model.py      ✅ Transformer/LSTM world model + heads + loss
│   ├── rollout/
│   │   └── rollout.py          ✅ K-step forward simulation + infiltration timeline
│   └── explain/                ⏳ TODO — SHAP + attention attribution
├── training/
│   ├── train.py                ✅ Colab-ready trainer (synthetic + CIC path)
│   └── README.md               ✅ Colab quickstart
├── benchmark/                  ⏳ TODO — logistic-regression baseline + metrics
├── demo/                       ⏳ TODO — offline Streamlit upload-and-forecast app
└── data/                       (datasets — gitignored)
```

Legend: ✅ built & validated · ⏳ not yet built.

---

## 5. Build roadmap — from here to project-ready

Each phase is a self-contained, demoable increment. Status as of this writing:

| Phase | What | Status |
|---|---|---|
| **0** | Scaffold + port SENTINEL feature code | ✅ done |
| **1** | Data foundation — state.py, CIC loader, synthetic gen | ✅ done |
| **2** | World model + K-step rollout + Colab training script | ✅ done (code) |
| **3** | **Run on Colab** — smoke test, then train on CIC-IDS-2018 | ⏳ **NEXT** |
| **4** | Explainability — SHAP + attention viz (`foresight/explain/`) | ⏳ |
| **5** | Benchmark — logistic-regression baseline, metrics table | ⏳ |
| **6** | Offline Streamlit demo — upload → forecast timeline + stage + why | ⏳ |
| **7** | Polish — theming, recorded demo video, slides, README figures | ⏳ |
| **S** | STRETCH — GNN encoder, CTU-13 second dataset, live SENTINEL bridge | optional |

**Critical path to a submittable prototype:** 3 → 5 → 6 (train, benchmark, demo).
Phase 4 (explainability) is required by the PS and is a judge favourite — do not skip.

### Phase 3 detail (do this next)
1. Push repo to GitHub.
2. Colab: `!git clone …`, `pip install torch scikit-learn pandas`, `!python training/train.py` (10 s synthetic smoke test — confirms the brain works).
3. Download **CIC-IDS-2018** CSVs (Kaggle hosts them) into `data/cicids/`.
4. `!python training/train.py --data "data/cicids/*.csv" --nrows 400000 --epochs 30 --out /content/drive/MyDrive/foresight_ckpt`
5. Download `world_model.pt` → laptop for the offline demo.

---

## 6. How to run

### Local development (no GPU, no torch needed for data work)
```bash
cd sentinel-foresight
PYTHONPATH=$PWD python -c "from foresight.data.synth import make_synthetic_flows; \
  from foresight.data.state import flows_to_state_windows; \
  X,y,ts = flows_to_state_windows(make_synthetic_flows()); print(X.shape, set(y))"
```
Feature/data code runs on numpy+pandas. **Torch is only needed to train** (Colab)
and to run inference for the demo (light — CPU laptop is fine).

### Train — on Colab (see training/README.md)
Free T4 GPU. Produces a portable `world_model.pt` (weights + scaler + feature
list + config). Inference runs offline on the laptop from that file.

### Demo — on the laptop (Phase 6, offline)
`streamlit run demo/app.py` → upload a PCAP/CSV → forecast timeline + stage + why.

---

## 7. Design decisions & rationale (defend these)

- **State = windowed feature vector (representation A), not a graph.** Satisfies
  the entire core deliverable, trains on free Colab, Transformer-friendly. GNN is
  a stretch differentiator, not the foundation.
- **Predict the next *state*, not a label.** This is what earns the "world model"
  claim. The stage is a readout of the predicted future state.
- **Temporal train/val split** (first 70% time / last 30%), never random — random
  splitting leaks the future into training and inflates metrics. Judges check this.
- **Leakage-safe standardization** — scaler fit on training windows only.
- **Attention (Transformer) for explainability**, SHAP as feature-attribution
  backup. PS accepts either; we do both for strength.
- **Separate repo from SENTINEL.** SENTINEL = real-time detect & respond appliance;
  Foresight = offline predictive world model. Shared feature code, separate stories.
- **Synthetic generator** so we develop and smoke-test with zero dataset/compute,
  respecting the "no strong local hardware" constraint.

---

## 8. SIH deliverables → where they live

| PS requirement | Implementation |
|---|---|
| Flow + packet features → normalised matrix | `foresight/features/*`, `foresight/data/state.py` |
| State-transition dynamics model | `foresight/model/world_model.py` |
| Trained on labelled open dataset, reproducible | `training/train.py` (+ portable ckpt) |
| K-step forecast + infiltration probability | `foresight/rollout/rollout.py` |
| MITRE ATT&CK stage mapping | `foresight/mitre.py` + stage head |
| Explainability (attention / SHAP) | attention in model; `foresight/explain/` (TODO) |
| Offline demo (Streamlit) accepting PCAP/CSV | `demo/` (TODO) |
| Benchmark vs logistic-regression baseline | `benchmark/` (TODO) |

---

## 9. Demo & presentation narrative

Mirror SENTINEL's live-attack arc, but the payoff is **prediction, not reaction**:

> "Here's a network capture with an attack hidden inside. [upload] Watch the
> infiltration probability — flat during recon… now it climbs sharply, and the
> model **forecast this ~15 seconds before the attacker completed initial
> access.** Here's *why* — SYN-heavy scans across 20 ports. And here's proof it
> beats a standard classifier by X% F1."

The three things that win: (1) **forecast-before-compromise** timeline,
(2) **explanation** for each call, (3) **benchmark** beating the baseline.

---

## 10. Environment & constraints

- **No strong local GPU.** Training → **Google Colab** (free T4). Inference/demo →
  laptop CPU (light). Torch is *not* installed locally by choice.
- **Datasets are large** (CIC-IDS-2018 ~16M flows). Use `--nrows` subsets / specific
  attack days to fit free Colab. Report as a prototype under stated constraints.
- **Ported code provenance:** `foresight/features/*` are copied unchanged from
  SENTINEL (see file headers) — attribute honestly.
- Python 3.11+; deps in `requirements.txt`.

---

## 11. Glossary

- **S_t** — network state at time window t (22-dim vector).
- **World model** — model of `P(S_t+1 | S_t)`; the transition dynamics.
- **Rollout** — feeding predictions back to simulate K steps into the future.
- **Kill chain / stage** — Recon → Initial Access → Lateral Movement → C2 → Exfil.
- **Infiltration probability** — P(state is at/after Initial Access).
