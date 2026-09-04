# Decision Log

Every non-obvious design choice gets an entry here, with the reasoning, dated. When a past decision
turns out to be wrong, add a new entry reversing it — never edit history.

---

## 2026-09-03 — Project framing: capital preservation, not return maximization

**Decision:** The success criterion is "beat cash over a full market cycle with max drawdown under
15%," not "maximize return."

**Why:** Stated goal is reliability and not losing money. Optimizing for return directly leads to
leverage and concentration, which are the two things that actually wipe out beginner accounts.

---

## 2026-09-03 — No ML before Phases 1-3 are complete

**Decision:** Phases 0-3 contain zero machine learning. A rules-based strategy is the baseline.

**Why:** The rules baseline is the control group, not a warm-up. ML on price data overfits
aggressively and silently. If a model cannot beat a two-parameter rule out-of-sample, it has
learned noise.

---

## 2026-09-03 — Write our own backtester rather than using a library

**Decision:** Custom ~300-line backtest engine, cross-checked against `vectorbt`.

**Why:** Off-the-shelf engines hide their assumptions about costs, fills, and timing. For a
monthly-rebalance, few-asset universe the engine is simple enough to fully understand. Understanding
every line matters more than features here. Cross-checking against an established library guards
against our own bugs.

---

## 2026-09-03 — ETFs only, no individual stocks

**Decision:** Universe restricted to broad-market ETFs.

**Why:** Single-name risk is unbounded and unpredictable — a company can gap down 40% on news no
model anticipated. Broad indices cannot. Removes an entire category of loss.

---

## 2026-09-03 — No leverage, margin, options, shorting, or crypto

**Decision:** Cash accounts and long-only positions in liquid ETFs. Permanently.

**Why:** These instruments are the proximate cause of most catastrophic retail losses. Excluding
them by policy removes the possibility rather than relying on discipline.

---

## 2026-09-03 — REVERSES the "no ML before Phase 4" decision

**Decision:** The AI is built in Phase 1, not Phase 4.

**Why:** The original objection to early ML was that models trained on real price history overfit
silently — with one copy of history and no ground truth, a model that learned something and a model
that memorized noise are indistinguishable. Developing inside a synthetic market removes that
objection entirely: we generate the market, so we know what is in it and can check the model's
answers. The constraint was never "ML is dangerous"; it was "ML is unfalsifiable on real data."
Building the sandbox first makes it falsifiable, so the AI can start immediately.

The caution from the original decision survives in a better form: the rules baseline is no longer a
phase to get through, but a permanent control arm reported alongside the AI in every experiment.

---

## 2026-09-03 — Synthetic market simulator is the centerpiece, built first

**Decision:** Phase 0 is the market generator. Everything else depends on it.

**Why:** It converts the project's central unanswerable question ("did the AI learn anything?") into
a checkable one. It also enables the Null Test as a permanent CI check: if the AI ever profits on
pure randomness, the build fails. That single automated guard covers the most expensive and most
common bug class in the field, on every commit, forever.

---

## 2026-09-03 — Parametric generators, not learned ones (no GANs)

**Decision:** Synthetic markets come from parametric models (GBM, AR(1), Ornstein-Uhlenbeck,
regime-switching, jump-diffusion, Heston), not from GANs or diffusion models.

**Why:** A learned generator produces data whose true statistical properties are unknown, which
destroys the entire reason for simulating. The value here is exact ground truth, and only
parametric models provide it. Revisit as a late stretch goal, never as a validation substrate.

---

## 2026-09-03 — All market sources share one interface

**Decision:** Synthetic, historical, and live feeds are interchangeable behind a single interface.
The AI cannot tell which it is trading.

**Why:** Makes each rung of the Reality Ladder attributable — moving up changes the data source and
nothing else, so any behavior change is caused by reality rather than by a diverging code path.
Backtest/live code divergence is where undetected bugs live in most retail systems.

---

## 2026-09-03 — LightGBM before deep learning; RL deferred but viable

**Decision:** Gradient-boosted trees are the primary model. Deep learning and RL are Phase 5
experiments.

**Why:** On tabular data at this volume trees genuinely outperform neural nets and train in seconds
rather than hours, while reporting feature importances that keep the system interpretable. RL is
normally impractical for retail because it is enormously data-hungry — but an unlimited simulator is
exactly the condition RL needs, so it becomes viable here once the simulator has been validated
against reality. Its specific risk is learning to exploit simulator quirks rather than market
structure, which is why it comes after rung 2.
