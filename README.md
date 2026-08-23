# SENTINEL Foresight
### A World Model for Predictive Cyber Defence — SIH 2026 · PS 26153

Most intrusion detectors classify each network flow in isolation as benign or
malicious, discarding the *temporal, causal structure* of an attack. **Foresight**
instead learns a **world model** of the network — the state-transition dynamics
`P(S_t+1 | S_t)` — and rolls it forward K steps to forecast attacker progression
**before the kill chain completes.**

## What it does
- Ingests **flow-level** (NetFlow/CSV) and **packet-level** (PCAP) telemetry and
  builds a timestamped, normalised network-state feature matrix.
- Learns state-transition dynamics with a **temporal sequence model**
  (LSTM / Transformer) trained on labelled open datasets (CIC-IDS-2018, CTU-13).
- **Forward-simulates** K steps: outputs an infiltration-probability timeline and
  the predicted **MITRE ATT&CK stage** (Recon → Initial Access → Lateral Movement
  → C2 → Exfiltration).
- Explains every prediction (attention weights / SHAP) — which flags, ports, and
  flow statistics drive it.
- Benchmarks against a logistic-regression baseline (F1 / precision / recall / FPR).
- Offline Streamlit demo: drop in a PCAP/CSV, see the forecast. No cloud APIs.

## Layout
```
foresight/
  features/   flow + packet feature extraction   (ported from SENTINEL)
    traffic_models.py   TCP flags, protocol, flow schema
    packet_features.py  damped IAT / jitter stats per 5-tuple
  data/       dataset loaders (CIC-IDS-2018, CTU-13)      [to build]
  model/      the world model — P(S_t+1 | S_t)            [to build]
  rollout/    K-step forward simulation + stage readout   [to build]
  explain/    attention / SHAP attribution                [to build]
  mitre.py    kill-chain stage vocabulary
training/     Colab training scripts + configs            [to build]
benchmark/    world model vs logistic-regression baseline [to build]
demo/         offline Streamlit upload-and-forecast app   [to build]
```

## Relationship to SENTINEL
Foresight reuses SENTINEL's battle-tested feature-extraction code (flow parsing,
Kitsune-style timing stats, TCP-flag decoding, MITRE vocabulary) but is a
**distinct system**: SENTINEL is a real-time detection-and-response appliance;
Foresight is a predictive world model trained offline on datasets. Kept separate
so neither story muddies the other.

## Status
Scaffold + ported feature pipeline. World model, training, and demo to follow.
