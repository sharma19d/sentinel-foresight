"""
SENTINEL Foresight — offline demo.

Runs fully offline on a laptop: loads the trained checkpoint, turns a capture
into the world model's network-state sequence, rolls the learned dynamics K
steps ahead at every moment, and plots the resulting infiltration-probability
timeline — the forecast, not a post-hoc classification.

    streamlit run demo/app.py

Defaults to a bundled real slice of held-out CIC-IDS-2018 so a live demo needs
no 6.5 GB dataset download. Also accepts an uploaded CICFlowMeter CSV or a raw
PCAP (reassembled to flows locally). The synthetic generator remains as a third
option but is labelled out-of-distribution: it predates the real data, its
feature scales sit ~53σ from the training distribution, and the model saturates
at ~100% risk on every window of it — kept only as a test fixture.
"""

from __future__ import annotations

import os
import sys
import tempfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foresight.data.drift import check_drift
from foresight.data.state import flows_to_state_windows, STATE_FEATURES
from foresight.data.synth import make_synthetic_flows
from foresight.explain import explain
from foresight.mitre import AttackStage
from foresight.model.world_model import WorldModel, WorldModelConfig, INFILTRATION_STAGE
from foresight.rollout.rollout import forecast_series, rollout, sliding_windows

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(os.path.dirname(_HERE), "checkpoints", "world_model_best.pt")

# A real slice of CIC-IDS-2018 (2018-03-02 09:40-11:00, Bot/C2) bundled with the
# repo so the demo works on any laptop without the 6.5 GB dataset.
# Two deliberate choices, both stated openly in demo/README.md so this reads as
# a documented decision rather than quiet cherry-picking:
#   * a HELD-OUT day — demoing on a training day would show the model recalling
#     traffic it had already fit, which is not the claim being made;
#   * the Bot/C2 day, where the model genuinely works (AUC 0.894) rather than an
#     Infiltration day, where it is at or below chance (AUC 0.466).
SAMPLE_CAPTURE = os.path.join(_HERE, "sample_capture.csv.gz")

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
def flows_from_path(path: str, nrows: int) -> pd.DataFrame:
    from foresight.data.cicids import load_cicids_csv
    return load_cicids_csv(path, nrows=nrows, require_label=False)


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
def flows_from_pcap_upload(raw_bytes: bytes, suffix: str, max_packets: int) -> pd.DataFrame:
    from foresight.data.pcap import load_pcap
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(raw_bytes)
        tmp = fh.name
    try:
        return load_pcap(tmp, max_packets=max_packets)
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
        # 0.5 is not a meaningful default — where it sits on the ROC curve
        # depends entirely on the checkpoint. The shipped model is recall-heavy
        # there (fires on ~47% of benign windows), so the default is the
        # train-picked high-precision point. See PRESENTING.md §4.
        threshold = st.slider("Alert threshold", 0.05, 0.99,
                              float(ckpt.get("alert_threshold", 0.90)), 0.01)
        stride = st.select_slider("Resolution (every Nth window)", [1, 2, 5, 10, 25], value=5)

        st.header("Traffic")
        source = st.radio("Source", [
            "Bundled real capture (CIC-IDS-2018)",
            "Upload CSV or PCAP",
            "Synthetic (out-of-distribution)",
        ])
        if source.startswith("Bundled"):
            st.caption("2018-03-02 09:40–11:00 — a **held-out** day the model never "
                       "trained on. Bot/C2 traffic, 152k flows, 54% attack windows.")
            nrows = st.number_input("Max rows", 10_000, 160_000, 160_000, 10_000)
        elif source.startswith("Upload"):
            upload = st.file_uploader("CICFlowMeter CSV or raw PCAP",
                                      type=["csv", "gz", "pcap", "pcapng"])
            nrows = st.number_input("Max rows / packets", 10_000, 2_000_000, 200_000, 10_000)
            st.caption("PCAP is reassembled into flows locally (scapy). Features "
                       "follow CICFlowMeter's definitions so the model sees what "
                       "it was trained on — the drift check below confirms it.")
        else:
            st.caption("⚠ Test fixture only — its feature scales sit far outside "
                       "the training distribution, so the model saturates on it.")
            minutes = st.slider("Capture length (minutes)", 4.0, 30.0, 12.0, 1.0)
            seed = st.number_input("Seed", 0, 9999, 0)

    # ── Build the state sequence ────────────────────────────────────
    if source.startswith("Bundled"):
        if not os.path.exists(SAMPLE_CAPTURE):
            st.error(f"Bundled sample missing at {SAMPLE_CAPTURE}")
            st.stop()
        flows = flows_from_path(SAMPLE_CAPTURE, int(nrows))
    elif source.startswith("Upload"):
        if not upload:
            st.info("⬅ Upload a CICFlowMeter CSV or a PCAP, "
                    "or switch to the bundled capture.")
            st.stop()
        name = (upload.name or "").lower()
        try:
            if name.endswith((".pcap", ".pcapng")):
                suffix = ".pcapng" if name.endswith(".pcapng") else ".pcap"
                with st.spinner("Reassembling packets into flows…"):
                    flows = flows_from_pcap_upload(upload.getvalue(), suffix, int(nrows))
            else:
                flows = flows_from_upload(upload.getvalue(), int(nrows))
        except (ValueError, ImportError) as err:
            # A capture with no TCP/UDP flows, or scapy missing — say so plainly
            # rather than failing deeper in with an opaque shape error.
            st.error(str(err))
            st.stop()
    else:
        flows = flows_synthetic(minutes, int(seed))

    X, y_true, _ts = flows_to_state_windows(flows, window_seconds=ckpt["window_seconds"])
    if len(X) <= ckpt["window"]:
        st.error(f"Only {len(X)} windows — need more than {ckpt['window']}.")
        st.stop()

    Xs = ((X.values.astype(np.float32) - mean) / scale).astype(np.float32)
    has_labels = bool(flows["label"].astype(str).str.len().max())

    # Refuse to present a confident-looking forecast on traffic the model was
    # never trained for. Without this the app will happily draw a flat 100%
    # risk line on out-of-distribution input and look authoritative doing it.
    drift = check_drift(Xs)
    if not drift.in_distribution:
        st.warning(drift.message())

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
    # Built with Altair rather than st.line_chart: the ground-truth overlay is
    # a 0/1 series, and drawing it as a second LINE on the same axis renders a
    # dense barcode that hides the forecast entirely. As a shaded band behind
    # the line it reads as context, which is the whole point of showing it.
    st.subheader("Infiltration probability — forecast timeline")
    df = pd.DataFrame({"window": idx, "risk": hprob})
    layers = []

    if has_labels:
        truth = (y_true[idx] >= INFILTRATION_STAGE)
        df["attack"] = truth
        layers.append(
            alt.Chart(df[df["attack"]]).mark_rect(opacity=0.16, color="#d62728")
            .encode(x=alt.X("window:Q", title="window (time →)"), x2="window2:Q")
            .transform_calculate(window2="datum.window + %d" % max(1, int(stride)))
        )

    # Zoom the y-axis when the risk band is narrow. A well-calibrated model can
    # sit entirely in (say) 0.8-1.0, and a fixed 0-1 axis renders that as a flat
    # line that hides all the structure the forecast actually has. The zoom is
    # called out in the caption rather than applied silently.
    lo_r, hi_r = float(hprob.min()), float(hprob.max())
    zoomed = (hi_r - lo_r) < 0.5
    y_domain = [max(0.0, lo_r - 0.02), min(1.0, hi_r + 0.02)] if zoomed else [0.0, 1.0]

    layers.append(
        alt.Chart(df).mark_line(color="#1f77b4", strokeWidth=1.6).encode(
            x=alt.X("window:Q", title="window (time →)"),
            y=alt.Y("risk:Q", title="forecast infiltration risk",
                    scale=alt.Scale(domain=y_domain, clamp=True)),
            tooltip=["window:Q", alt.Tooltip("risk:Q", format=".3f")],
        )
    )
    layers.append(
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="#d62728", strokeDash=[6, 4]).encode(y="t:Q")
    )
    st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
    st.caption(
        ("Red band = actual attack windows (ground truth) · " if has_labels else "")
        + f"dashed line = alert threshold ({threshold:.0%}) · "
        f"blue = risk forecast {k_steps} windows ahead"
        + (f" · **y-axis zoomed to {y_domain[0]:.2f}–{y_domain[1]:.2f}** "
           "(all risk values fall in a narrow band)" if zoomed else "")
    )

    st.subheader("Predicted kill-chain stage over time")
    # A step area, not bars: hundreds of bars at this density render as noise,
    # while a step makes the transitions between stages legible.
    stage_df = pd.DataFrame({"window": idx, "stage": hstage,
                             "name": [stage_name(s) for s in hstage]})
    st.altair_chart(
        alt.Chart(stage_df).mark_area(interpolate="step-after", opacity=0.75,
                                      color="#ff7f0e").encode(
            x=alt.X("window:Q", title="window (time →)"),
            y=alt.Y("stage:Q", title="kill-chain stage",
                    scale=alt.Scale(domain=[0, int(max(hstage.max(), 1))])),
            tooltip=["window:Q", "name:N"],
        ).properties(height=180),
        use_container_width=True,
    )
    st.caption(" · ".join(f"{int(s)} = {stage_name(s)}" for s in sorted(set(hstage))))

    # ── Why? ────────────────────────────────────────────────────────
    st.subheader("Why this forecast?")
    if idx[0] == idx[-1]:
        pos = int(idx[0])                       # single forecast point — no slider to draw
        st.caption(f"Only one forecast position available (window {pos}).")
    else:
        pos = st.slider("Inspect window", int(idx[0]), int(idx[-1]), int(idx[peak_i]))

    seeds = sliding_windows(Xs, ckpt["window"])
    seed_i = min(max(pos - (ckpt["window"] - 1), 0), len(seeds) - 1)
    # sliding_windows returns a strided view; make it contiguous before it
    # crosses into torch.
    seed = np.ascontiguousarray(seeds[seed_i])

    e = explain(model, seed, steps=32)
    fc = rollout(model, torch.from_numpy(seed), k_steps=k_steps)

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
