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
│   └── explain/                ✅ integrated gradients + attention (SHAP optional)
├── training/
│   ├── train.py                ✅ Colab-ready trainer (synthetic + CIC path)
│   └── README.md               ✅ Colab quickstart
├── benchmark/                  ✅ persistence / logreg / majority baselines + results.json
├── demo/                       ✅ offline Streamlit app + bundled real sample (CSV **and PCAP**)
├── tests/                      ✅ PCAP ingest · CSV/PCAP equivalence · demo smoke
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
| **3** | **Run on Colab** — smoke test, then train on CIC-IDS-2018 | ✅ done (3 runs, best-epoch ckpt) |
| **4** | Explainability — attribution + attention (`foresight/explain/`) | ✅ done |
| **5** | Benchmark — baselines + metrics table | ✅ done — beats persistence & logreg |
| **6** | Offline Streamlit demo — forecast timeline + stage + why | ✅ done (CSV **and PCAP**) |
| **7** | Polish — PRESENTING.md ✅ · demo video + slides still ⏳ |
| **S** | STRETCH — GNN encoder, CTU-13 second dataset, live SENTINEL bridge | optional |

**Critical path to a submittable prototype:** 3 → 5 → 6 (train, benchmark, demo).
Phase 4 (explainability) is required by the PS and is a judge favourite — do not skip.

### Measured results (trained on all 10 days of CIC-IDS-2018, 4M flows)

Validation split = the last 3 capture days, held out temporally (never randomly).
Reproduce with `benchmark/baseline.py`; raw numbers in `benchmark/results.json`.

**Dynamics — did it actually learn P(S_t+1 | S_t)?**

| | next-state MSE |
|---|---|
| Persistence baseline (`S_t+1 = S_t`) | 0.902 |
| **World model** | **0.562** (**−37.7%**) |

This is the result that justifies the name: the model predicts the network's
next state materially better than assuming nothing changes. Without clearing
this bar, "world model" would have been an overclaim.

**Detection — infiltration (stage ≥ Initial Access), 23.4% prevalence**

| model | precision | recall | F1 | FPR | AUC |
|---|---|---|---|---|---|
| Majority (always benign) | 0.000 | 0.000 | 0.000 | 0.0000 | n/a |
| Best single raw feature (`n_flows`) | 0.826 | 0.124 | 0.216 | 0.0105 | 0.825 |
| Logistic regression (same features) | 0.295 | 0.272 | 0.283 | 0.2596 | 0.585 |
| **World model** | 0.444 | 0.936 | **0.602** | 0.4693 | **0.876** |

**Read that FPR before quoting the F1.** At the default 0.5 threshold the model
is tuned for recall and fires on 47% of benign windows — fine for triage,
unusable for auto-blocking. Thresholds should be picked for the deployment, on
training data. Picking for FPR ≤ 1% on train gives, on val:
**precision 0.771, recall 0.279, F1 0.410, FPR 3.3%.**

Against the strongest trivial baseline (one raw feature) the honest margin is
**AUC 0.876 vs 0.825**. Against logistic regression the F1 gap looks enormous;
that flatters us, because that baseline sits badly against a fixed threshold.

**Per-attack-type breakdown — the most important caveat.** The aggregate F1
above hides a large split in capability. Forecast AUC against ground truth, per
held-out day:

| day | attack type | 360 s context (shipped) | 16 s context (previous) |
|---|---|---|---|
| 2018-03-02 | Bot / C2 | **0.970** | 0.894 |
| 2018-02-28 | Infiltration | **0.872** | 0.715 |
| 2018-03-01 | Infiltration | **0.818** | **0.466** |
| *pooled* | | **0.857** | 0.800 |

**The Infiltration failure was diagnosed and fixed.** The earlier model was
*below chance* on 03-01 — attack windows scored lower risk than benign ones.
Per-feature analysis showed the signal was there but needed a coarser window
(single-feature separability rose from 0.723 at 1 s to 0.781 at 60 s):
Infiltration is slow, low-volume traffic from an already-trusted host, and a
one-second window cannot distinguish it. Widening the state window from 16 s to
**6 minutes** raised Infiltration AUC from 0.466 to 0.818, and improved every
other day too.

**Be careful what you claim from this.** At *matched* operating points (FPR ≈ 1%)
the two models score almost the same pooled F1 (0.410 vs 0.408). The real,
defensible gain is in **ranking and coverage** — better AUC on all three days,
and an entire attack class going from undetectable to detected — not a jump in
the headline F1, which moved mostly because the fixed 0.5 threshold sits at a
different place on the new model's curve.

The model genuinely detects **Bot/C2 traffic** and is **at or below chance on
Infiltration** — on 03-01 attack windows score *lower* risk than benign ones.
Infiltration is the acknowledged-hardest CIC-IDS-2018 class (it is largely
normal-looking traffic from an already-trusted host), but the honest reading is
that the headline number is carried by the Bot day. Do not present the aggregate
F1 without this table; if a judge tests it on Infiltration it will fail, and
having already stated the limitation is far better than being caught by it.

**Known weakness — state it honestly, don't hide it:** recall is 0.252. The
model is a high-confidence, low-noise detector (94.6% precision, 0.44% FPR)
that still misses ~3 of every 4 attack windows. For an appliance that
auto-blocks, precision-first is a defensible trade — but recall is the
obvious target for the next iteration (longer windows, focal loss, or
per-stage thresholds rather than a fixed 0.5).

**Training note:** validation F1 swings violently between epochs (0.01→0.40 on
this task), so `train.py` keeps the *best-val-F1* epoch, not the last one.
The shipped checkpoint is epoch 3. Saving the final epoch instead scored
0.161 — 2.5× worse for the same run.

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
| Explainability (attention / attribution) | `foresight/explain/` ✅ |
| Offline demo (Streamlit) accepting PCAP/CSV | `demo/` ✅ both |
| Benchmark vs logistic-regression baseline | `benchmark/` ✅ (see results.json) |

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

- **No strong local GPU.** Training → **Google Colab** (free T4), driven from
  the terminal via `google-colab-cli` (see §12). Inference/demo → laptop CPU;
  CPU-only torch **is** now installed locally in `.venv/` and the demo runs
  offline there.
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

---

## 12. RESUME HERE — state as of 2026-08-28

Everything through Phase 6 is built, trained, benchmarked, and **verified by
actually running it**. This section is what you need to pick the work back up
cold, months later, with no memory of the session.

### Environment setup (one-time, ~10 min)

```bash
cd sentinel-foresight
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install streamlit pandas numpy scikit-learn
.venv/bin/streamlit run demo/app.py        # works offline, bundled sample
```

`.venv/` and `checkpoints/` are gitignored — the trained weights are **not in
git**. If `checkpoints/world_model_best.pt` is missing, retrain (below).

### Training on Colab — driven from the terminal, no browser

`google-colab-cli` lets an agent run Colab directly, which removed all
copy-paste from the workflow. Setup:

```bash
pipx install google-colab-cli
# jupyter_kernel_client 1.x breaks `colab exec`; pin the last working release:
/path/to/pipx/venvs/google-colab-cli/bin/python -m pip install "jupyter_kernel_client==0.15.0"

# gcloud ADC needs all four scopes or the keep-alive 403s:
gcloud auth application-default login \
  --scopes="openid,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/colaboratory"
```

Then (note `--auth=adc` is required on **every** call — the installed CLI
defaults to oauth2 despite its docs):

```bash
colab --auth=adc new -s foresight --gpu T4
colab --auth=adc exec -s foresight --timeout 60   # pipe python via stdin, or -f script.py
colab --auth=adc download -s foresight /content/... ./local
colab --auth=adc stop -s foresight                # ALWAYS — idle VMs burn quota
```

Gotchas learned the hard way:
- **Only one GPU session per account.** A leftover browser runtime causes
  `TooManyAssignmentsError` (412); disconnect it in the browser first.
- **`colab exec` has a 30 s default timeout** and killing the CLI does *not*
  kill the remote process. Launch long jobs with `nohup ... > log 2>&1 &` via
  `subprocess.Popen`, then poll the log.
- **VMs get reclaimed mid-run.** Anything not downloaded is lost — a run was
  lost this way. Download checkpoints as soon as they are written.

Full training run (~11 min on a T4: ~4 min load, ~13 s/epoch):

```bash
python3 training/train.py --data "data/cicids/*.csv" --nrows 400000 \
  --window 16 --k 4 --encoder transformer --epochs 30 --batch 256 --lr 1e-3 \
  --out /content/foresight_ckpt
```

Dataset: `kaggle datasets download -d solarmainframe/ids-intrusion-csv --unzip`
(6.5 GB, 10 days). Kaggle now issues a single `KGAT_...` token → write it to
`~/.kaggle/access_token` (chmod 600), **not** the old `kaggle.json`.

### Bugs already fixed — do not reintroduce

Each of these was silent, and several only appeared when the code ran on a
second machine. They are the reason the pipeline is trustworthy now:

1. **`pd.to_datetime().astype("int64") / 1e9`** — only correct at nanosecond
   resolution. pandas ≥ 2.0 infers the unit and may return `datetime64[us]`,
   making every timestamp 1000× too small and collapsing a multi-hour capture
   into seconds. Colab had ns, the laptop had µs — so it worked in training and
   broke everywhere else. Always force `.astype("datetime64[ns]")` first.
2. **Repeated header rows** mid-file in several CIC CSVs (`Label == "Label"`)
   became fabricated all-zero benign flows.
3. **Corrupted timestamps** (a 1970 row in a 2018 file) inflated the apparent
   span to ~1.5 billion seconds; `fill_gaps` then tried to allocate one row per
   elapsed second and OOM-killed the process. Filtered by distance from median;
   `state.py` also hard-fails above 20M windows instead of exhausting RAM.
4. **Per-day windowing** — concatenating all 10 days into one timeline made
   sequences span overnight gaps, teaching transitions that never happened.
5. **Memory** — loading all 10 raw frames at once (each ~80 columns when only
   ~16 are used) OOM-killed free-tier Colab. Now: `usecols`, one file at a time,
   `gc.collect()` + `malloc_trim(0)`.
6. **Last-epoch checkpointing** — val F1 swings 0.01→0.40 between epochs, so
   saving epoch 30 saved noise. `train.py` now keeps the best-val-F1 epoch (the
   shipped checkpoint is **epoch 3**; epoch 30 scored 2.5× worse).
7. **Out-of-distribution saturation** — the synthetic fixture sits ~53σ from the
   training distribution and made the model report ~100% risk on *every* window,
   benign included. `foresight/data/drift.py` now warns instead of rendering a
   confident-looking lie.

### What to do next, in order

1. ~~**PCAP ingest**~~ — **done.** `foresight/data/pcap.py` reassembles packets
   into the canonical flow schema following CICFlowMeter's definitions (see its
   docstring for why each choice matters: payload-not-frame bytes, bidirectional
   keying, 120 s flow timeout). `tests/test_pcap_ingest.py` proves both ingest
   paths produce identical canonical features for identical traffic — the guard
   against unit-conversion asymmetry, since CICFlowMeter reports µs and PCAP
   gives seconds. **Still unverified:** distribution match against real
   CICFlowMeter output, which needs a labelled PCAP *and* its CSV counterpart.
   Run `check_drift()` (the demo does automatically) before trusting a
   PCAP-derived forecast.
2. ~~**Improve Infiltration recall**~~ — **largely done.** Diagnosed as a
   windowing problem, not an architecture one: single-feature separability on
   Infiltration rises with window size (0.723 at 1 s → 0.781 at 60 s), because
   the attack is slow and low-volume. Retrained at `--window-seconds 15
   --window 24` (6 min context); Infiltration AUC 0.466 → 0.818 and every other
   day improved. That checkpoint is now shipped. **Remaining:** at matched FPR
   the pooled F1 is unchanged, so the win is ranking/coverage rather than the
   headline metric — a genuine further gain likely needs the host-level
   fan-in/fan-out features the public CSVs drop (CTU-13 has the IPs), or focal
   loss, or per-stage thresholds.
3. **Improve precision at usable thresholds** — the real weakness (AUC 0.466). Ideas, in
   rough order of expected value: focal loss instead of flat class weights; a
   longer `--window` (Infiltration unfolds slowly); per-stage thresholds rather
   than a fixed 0.5; adding host-level fan-in/fan-out features, which need the
   IP columns the public CSVs drop (CTU-13 has them).
4. **Phase 7 polish** — `PRESENTING.md` is written; a recorded demo video
   and slides remain.
5. *(Stretch)* CTU-13 as a second dataset; GNN encoder; live SENTINEL bridge.

### Reproducing the numbers

```bash
python3 benchmark/baseline.py --data "data/cicids/*.csv" --nrows 400000 \
  --ckpt checkpoints/world_model_best.pt
```

`benchmark/results.json` holds the committed results, including the per-day
breakdown. The demo sample is a real held-out slice (2018-03-02 09:40–11:00,
54% attack windows) — deliberately **not** a training day, and deliberately Bot
traffic, which is where the model actually works. Both facts are stated in
`demo/README.md` so the choice is transparent rather than quiet cherry-picking.
