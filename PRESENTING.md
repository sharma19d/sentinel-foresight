# Presenting SENTINEL Foresight

Everything needed to demo this to a panel, a professor, or a judge — including
the questions that will be asked and the honest answers to them.

---

## 1. Before you present (do this the day before, not on the day)

```bash
cd sentinel-foresight
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install streamlit pandas numpy scikit-learn scapy altair

.venv/bin/python tests/test_pcap_ingest.py      # both must print PASSED
.venv/bin/python tests/test_demo_smoke.py
.venv/bin/streamlit run demo/app.py             # opens on http://localhost:8501
```

**Checklist**
- [ ] `checkpoints/world_model_best.pt` exists (it is **gitignored** — if you
      cloned fresh onto a new laptop it will NOT be there; copy it across or
      retrain per `PROJECT_CONTEXT.md` §12).
- [ ] The demo loads and the timeline chart draws within ~15 s.
- [ ] Both test scripts print PASSED.
- [ ] You have run through the script below out loud at least once.

**It runs fully offline.** No cloud APIs, no dataset download — the real capture
is bundled (1.9 MB). Venue Wi-Fi cannot break the demo.

---

## 2. The 4-minute demo script

### Frame the problem (30 s, before touching the screen)

> "A normal IDS classifies each network flow in isolation — benign or
> malicious. That throws away the thing that actually defines an attack: the
> *order* things happen in. An intrusion is a process unfolding over time, not
> one bad packet.
>
> So we didn't build a classifier. We built a **world model** — it learns how
> the network's *state* evolves, `P(S_t+1 | S_t)`, and rolls that forward to
> forecast where the network is heading."

### Show the forecast (90 s)

Point at the timeline. Say what it is:

> "This is a real capture from CIC-IDS-2018 — a day the model has **never seen**,
> held out from training. The blue line isn't a classification of what's
> happening now; it's the model's forecast for **four windows into the future**.
> The red band is the actual attack, so you can see the forecast against ground
> truth."

Then move the **Horizon K** slider and let them watch the line change. That is
the moment the "world model, not classifier" claim becomes visible rather than
asserted — the forecast genuinely extends further ahead.

### Show *why* (60 s)

Scroll to **Why this forecast?**:

> "Every prediction is attributable. Here the model is reacting to
> `n_flows` and `ACK` volume, and the attention shows it's leaning on traffic
> from a few windows back — not the instant. And this isn't a hand-wavy
> saliency map: integrated gradients satisfies a completeness axiom, and we
> check it at runtime. If the attribution isn't faithful, the app says so
> instead of showing you a confident-looking picture."

### Show the honesty guardrail (30 s) — do not skip this

Switch the traffic source to **Synthetic (out-of-distribution)**. A warning
appears.

> "One more thing. Our synthetic test traffic sits about 53 standard deviations
> outside the training distribution, and the model happily reported 100% risk on
> *every* window of it — including benign ones. A security tool that is
> confidently wrong is worse than useless, so it now detects that and refuses to
> present the forecast as trustworthy."

This lands better than any accuracy number. It shows you tested your own system
adversarially and found something.

### Close on the benchmark (30 s)

> "Against a persistence baseline — 'assume nothing changes' — it predicts the
> next network state **37.7% better**. That's what earns the name *world model*.
> On detection it beats logistic regression and the best single raw feature, at
> an **8× lower false-positive rate**."

---

## 3. Questions you will be asked, and the honest answers

**"Is this really a world model, or a classifier with extra steps?"**
> It regresses the full 22-dimensional next state, then reads the kill-chain
> stage off that *predicted* state. Because it predicts state, predictions feed
> back in and roll K steps forward. A classifier cannot do that. And it beats a
> persistence baseline on next-state MSE by 37.7% — if it couldn't, I'd agree the
> name was an overclaim.

**"What's your accuracy?"** — do not answer with accuracy.
> Accuracy is misleading here: always predicting "benign" scores ~76% on this
> data. Use AUC 0.876 and the FPR instead, and say why.

**"Where does it fail?"** — the most important question. Answer it fully.
> Recall is the weak point. It's a high-precision, low-noise early-warning
> signal, not a complete detector. It was also *below chance* on Infiltration
> traffic until we diagnosed the cause — one-second windows are too fine-grained
> for an attack that unfolds slowly — and widening the window to six minutes took
> that from 0.466 to 0.818 AUC.

**"Why should I trust the numbers?"**
> Validation is the last three capture days, split temporally, never randomly —
> a random split would leak future traffic into training. The scaler is fit on
> train only and travels inside the checkpoint. And the benchmark includes the
> strongest trivial baseline we could build, not just a weak one.

**"Does it work on my traffic?"**
> Drop a `.pcap` into the uploader. It reassembles flows locally using
> CICFlowMeter's feature definitions. If it's unlike the training data, the
> drift guard will tell you rather than inventing a number.

---

## 4. Things to be careful about

**Do not quote "12× better than logistic regression."** It's technically true
and it flatters us — that baseline scores badly at a fixed 0.5 threshold.
Against the best single raw feature the AUC margin is modest (0.876 vs 0.825).
Lead with the false-positive rate instead; it's the honest and more impressive
number.

**Do not demo with the synthetic source** except to show the drift guard.

**Do not claim per-day dominance.** Within a single day a raw feature can beat
the model. What survives scrutiny is that the model's scores stay comparable
*across* days, where a raw feature's scale shifts.

**Know your threshold.** The default 0.5 is not sacred — the shipped model's
operating point moves a lot with it (at 0.5 it favours recall; raise it for
precision). If asked "why 0.5", the answer is that thresholds should be picked
on train data for a target FPR, and `benchmark/results_ws15.json` shows the
trade-off curve.

---

## 5. If something breaks live

| Symptom | Fix |
|---|---|
| "Checkpoint not found" | Path box in the sidebar — point it at the `.pt` file. |
| Demo slow / laggy | Raise **Resolution (every Nth window)** to 10 or 25. |
| Upload rejected | Only `.csv`, `.csv.gz`, `.pcap`, `.pcapng` are accepted. |
| PCAP gives "no TCP/UDP flows" | The capture has no IP-based TCP/UDP traffic. |
| Everything is broken | `git stash && git checkout main` — the bundled sample
  path is the one exercised by the smoke tests. |

**Fallback:** if the laptop dies entirely, `benchmark/results.json` and the
tables in `README.md` carry every number, and the architecture diagram is in
`PROJECT_CONTEXT.md` §3. You can present the whole thing from the repo.
