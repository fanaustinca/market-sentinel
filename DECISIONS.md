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

---

## 2026-09-03 — Ground truth is a separate object from market data

**Decision:** `MarketData` holds prices and a neutral name, nothing else. Generator parameters,
signal presence, and regime labels live in a separate `GroundTruth` object; `Scenario` pairs them.
Modelling code accepts `MarketData`, never `Scenario`.

**Why:** The plan requires that the AI cannot tell which rung of the Reality Ladder it is on. Making
that a matter of discipline would eventually fail. With the answer key in a different object there
is no field to leak it through — passing prices to a model cannot accidentally pass the answer too.

---

## 2026-09-03 — Validate the generator by rejection rate, not by single paths

**Decision:** The generator is verified by running ~1000 independent null markets and checking that
each statistical test rejects at exactly its 5% significance level, rather than by asserting that
any individual path passes.

**Why:** Seed 42 produces a market our tests call "trending" — from a generator built to contain no
signal at all. That is not a bug; at a 5% level, one path in twenty is flagged by chance. Asserting
that a single path passes would produce a test that fails randomly one run in twenty, and would
prove nothing when it passed.

Inverting the question fixes both problems. Too many rejections means the generator leaks structure;
too few means the paths are suspiciously well behaved, which random data never is. The band is
three binomial standard errors.

This is the same reasoning the Null Test will use in Phase 1: the question is never "did the AI lose
money on noise?" but "did the AI's results fall inside the distribution chance alone produces?"

---

## 2026-09-03 — Hand-write Ljung-Box and the variance ratio test rather than use statsmodels

**Decision:** `sentinel/stats/randomwalk.py` implements the tests directly on numpy and scipy.

**Why:** These two tests are the measuring instrument the entire project depends on; a wrong or
misunderstood one invalidates every result downstream. They are about thirty lines each, and the
cost of understanding them fully is far below the cost of trusting them blindly. It also drops a
heavy dependency. Cross-check against `statsmodels` if any result ever looks surprising.

---

## 2026-09-04 — Circuit breaker resets its reference peak on re-entry

**Decision:** When the drawdown breaker's cooldown expires, the peak it measures
against is reset to the current equity value.

**Why:** Found by inspection of a backtest that looked wrong — buy-and-hold showed
6.3% volatility on a 16% market. The breaker was measuring against an all-time
high the strategy might never reach again, so it re-tripped on the very next bar
and the system sat in cash permanently after one bad stretch. Each re-entry now
gets its own drawdown budget. `drawdown_stop=None` (and the `UNLIMITED` preset)
disables the breaker for measurement runs, where it would otherwise mask the
behaviour being measured.

---

## 2026-09-04 — Lookahead is detected by truncation, not by code review

**Decision:** `check_causality` feeds a strategy progressively longer histories
and asserts past decisions never change. Applied to every strategy, including
ones not yet written.

**Why:** Reviewing code for lookahead does not work reliably — it hides in a shift
of the wrong sign, a centred rolling window, a full-sample mean used to
standardise. The truncation property is mechanical and needs no understanding of
how a strategy works.

**Correction, same day:** the first version excluded each truncation's final row
as "unsettled". That was wrong and dangerous: a strategy peeking exactly one day
ahead differs from an honest one *only* in that row, so the exclusion blinded the
detector to the most important case. Caught only because `tests/test_no_lookahead.py`
contains deliberately-broken strategies that must always be detected. Those cheat
strategies are load-bearing — without them the flaw would have shipped and every
causality claim in the project would have been worthless. Do not delete them.

---

## 2026-09-04 — Null markets must have zero drift

**Decision:** The Null Test uses generators configured with `mu=0`.

**Why:** With positive drift, any strategy holding assets makes money — from beta,
not skill. The Null Test would then measure exposure rather than the thing it
exists to detect. At `mu=0` the expected return of any fixed allocation is zero,
so profit can only come from timing, and there is nothing to time.

`run_null_test` additionally refuses any generator declaring
`has_exploitable_signal=True`, so the control can never accidentally be run on a
market that contains a signal.
