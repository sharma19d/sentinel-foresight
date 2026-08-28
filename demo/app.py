"""
SENTINEL Foresight — offline demo.

Runs fully offline on a laptop: loads the trained checkpoint, turns a capture
into the world model's network-state sequence, rolls the learned dynamics K
steps ahead at every moment, and plots the resulting infiltration-probability
timeline — the forecast, not a post-hoc classification.

    streamlit run demo/app.py

Deliberately defaults to built-in synthetic attack traffic rather than an
uploaded file: a live demo shouldn't depend on a 6.5 GB dataset being present,
and the synthetic capture walks cleanly through benign → recon → brute-force →
C2 → benign, which makes the forecast behaviour legible in one screen.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresight.data.state import flows_to_state_windows, STATE_FEATURES
from foresight.data.synth import make_synthetic_flows
from foresight.explain import explain
from foresight.mitre import AttackStage
from foresight.model.world_model import WorldModel, WorldModelConfig, INFILTRATION_STAGE
from foresight.rollout.rollout import forecast_series, rollout, sliding_windows

DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "world_model_best.pt",
)

STAGE_COLOUR = {
    AttackStage.BENIGN: "#2e7d32",
    AttackStage.RECONNAISSANCE: "#f9a825",
    AttackStage.INITIAL_ACCESS: "#ef6c00",
    AttackStage.LATERAL_MOVEMENT: "#d84315",
    AttackStage.COMMAND_AND_CONTROL: "#c62828",
    AttackStage.EXFILTRATION: "#6a1b9a",
}


def stage_name(s: int) -> str:
    return AttackStage(int(s)).name.replace("_", " ").title()


@st.cache_resource
def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WorldModel(WorldModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    # The checkpoint carries the scaler it was fit with. Re-fitting a scaler on
    # the uploaded capture instead would silently shift every feature into a
    # different space than the model was trained in, producing confident
    # nonsense — so the training-time statistics travel with the weights.
    return model, mean, scale, ckpt


@st.cache_data(show_spinner=False)
def flows_from_upload(raw_bytes: bytes, nrows: int) -> pd.DataFrame:
    from foresight.data.cicids import load_cicids_csv
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        fh.write(raw_bytes)
        tmp = fh.name
    try:
        # Unlabelled captures are the realistic inference case, so labels are
        # optional here; if the file does carry them we show them as ground truth.
        return load_cicids_csv(tmp, nrows=nrows, require_label=False)
    finally:
        os.unlink(tmp)


@st.cache_data(show_spinner=False)
def flows_synthetic(minutes: float, seed: int) -> pd.DataFrame:
    return make_synthetic_flows(seed=seed, minutes=minutes)


def main():
    st.set_page_config(page_title="SENTINEL Foresight", page_icon="🛰", layout="wide")
    st.title("🛰 SENTINEL Foresight")
    st.caption(
        "Predictive cyber defence with a learned world model — forecasting where "
        "the network is heading, not just classifying where it has been."
    )

    with st.sidebar:
        st.header("Model")
        ckpt_path = st.text_input("Checkpoint", DEFAULT_CKPT)
        if not os.path.exists(ckpt_path):
            st.error("Checkpoint not found. Train one, or point at a .pt file.")
            st.stop()
        model, mean, scale, ckpt = load_checkpoint(ckpt_path)
        st.success(f"{ckpt['config']['encoder']} · window {ckpt['window']}"
                   + (f" · best epoch {ckpt['best_epoch']}" if ckpt.get("best_epoch") else ""))

        st.header("Forecast")
        k_steps = st.slider("Horizon K (windows ahead)", 1, 20, int(ckpt.get("k", 4)))
        threshold = st.slider("Alert threshold", 0.05, 0.95, 0.5, 0.05)
        stride = st.select_slider("Resolution (every Nth window)", [1, 2, 5, 10, 25], value=5)

        st.header("Traffic")
        source = st.radio("Source", ["Built-in synthetic attack", "Upload CIC-IDS CSV"])
        if source.startswith("Built-in"):
            minutes = st.slider("Capture length (minutes)", 4.0, 30.0, 12.0, 1.0)
            seed = st.number_input("Seed", 0, 9999, 0)
        else:
            upload = st.file_uploader("CICFlowMeter CSV", type=["csv"])
            nrows = st.number_input("Max rows", 10_000, 1_000_000, 200_000, 10_000)

    # ── Build the state sequence ────────────────────────────────────
    if source.startswith("Built-in"):
        flows = flows_synthetic(minutes, int(seed))
    else:
        if not upload:
            st.info("⬅ Upload a CICFlowMeter CSV, or switch to the built-in capture.")
            st.stop()
        flows = flows_from_upload(upload.getvalue(), int(nrows))

    X, y_true, _ts = flows_to_state_windows(flows, window_seconds=ckpt["window_seconds"])
    if len(X) <= ckpt["window"]:
        st.error(f"Only {len(X)} windows — need more than {ckpt['window']}.")
        st.stop()

    Xs = ((X.values.astype(np.float32) - mean) / scale).astype(np.float32)
    has_labels = bool(flows["label"].astype(str).str.len().max())

    with st.spinner(f"Rolling the world model {k_steps} steps ahead…"):
        idx, hprob, hstage = forecast_series(
            model, Xs, window=ckpt["window"], k_steps=k_steps, stride=int(stride),
        )

    # ── Headline metrics ────────────────────────────────────────────
    alerts = hprob >= threshold
    peak_i = int(np.argmax(hprob))
    peak_stage = int(hstage[peak_i])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Windows analysed", f"{len(X):,}")
    c2.metric("Peak infiltration risk", f"{hprob[peak_i]:.0%}")
    c3.metric("Forecast stage at peak", stage_name(peak_stage))
    c4.metric(f"Windows over {threshold:.0%}", f"{int(alerts.sum()):,}")

    if alerts.any():
        first = int(np.argmax(alerts))
        st.error(
            f"⚠ Forecast crosses the alert threshold at window **{idx[first]}**, "
            f"predicting **{stage_name(int(hstage[first]))}** up to {k_steps} "
            f"windows before it fully develops."
        )
    else:
        st.success("✓ No forecast window crosses the alert threshold.")

    # ── The timeline ────────────────────────────────────────────────
    st.subheader("Infiltration probability — forecast timeline")
    chart = pd.DataFrame({"window": idx, "forecast risk": hprob}).set_index("window")
    if has_labels:
        # Ground truth, when the capture happens to carry labels, so the
        # forecast can be read against what actually happened.
        truth = (y_true[idx] >= INFILTRATION_STAGE).astype(float)
        chart["actual attack (ground truth)"] = truth
    st.line_chart(chart, height=320)

    st.subheader("Predicted kill-chain stage over time")
    stage_df = pd.DataFrame({
        "window": idx,
        "stage": [stage_name(s) for s in hstage],
        "value": hstage,
    })
    st.bar_chart(stage_df.set_index("window")["value"], height=180)
    st.caption(" · ".join(f"{int(s)} = {stage_name(s)}" for s in sorted(set(hstage))))

    # ── Why? ────────────────────────────────────────────────────────
    st.subheader("Why this forecast?")
    pos = st.slider("Inspect window", int(idx[0]), int(idx[-1]), int(idx[peak_i]))
    seeds = sliding_windows(Xs, ckpt["window"])
    seed_i = min(max(pos - (ckpt["window"] - 1), 0), len(seeds) - 1)

    e = explain(model, seeds[seed_i], steps=32)
    fc = rollout(model, torch.from_numpy(np.ascontiguousarray(seeds[seed_i])), k_steps=k_steps)

    st.markdown(f"**{e.summary()}**")
    if e.completeness_error > 0.05:
        # Surfaced rather than hidden: a large completeness residual means the
        # attribution is not faithful and should not be argued from.
        st.warning(f"Attribution completeness error {e.completeness_error:.3f} — "
                   "treat this explanation as unreliable.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Feature contributions**")
        feats = pd.DataFrame(e.top_features(8), columns=["feature", "contribution"])
        st.bar_chart(feats.set_index("feature"), height=280)
    with right:
        st.markdown("**Forecast trajectory from here**")
        st.dataframe(
            pd.DataFrame({
                "step ahead": np.arange(1, len(fc.infiltration_prob) + 1),
                "infiltration prob": fc.infiltration_prob.round(3),
                "predicted stage": [stage_name(s) for s in fc.stages],
            }),
            hide_index=True, use_container_width=True, height=280,
        )

    if e.attention is not None:
        st.markdown("**Temporal attention** — how much the forecast leaned on each "
                    "of the preceding windows (rightmost = most recent)")
        st.bar_chart(pd.DataFrame({"attention": e.attention}), height=160)

    with st.expander("Model & data details"):
        st.write({
            "encoder": ckpt["config"]["encoder"],
            "input window": ckpt["window"],
            "window seconds": ckpt["window_seconds"],
            "features": len(STATE_FEATURES),
            "flows loaded": len(flows),
            "state windows": len(X),
            "labels present": has_labels,
        })


if __name__ == "__main__":
    main()
