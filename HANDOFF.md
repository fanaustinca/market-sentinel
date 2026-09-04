# Handoff

State of the project as of **2026-09-04**, written for whoever picks it up next.

Read `plan.md` first for *why* the project is shaped this way. This file covers
what exists, what is verified, what is not, and what to do next.

---

## TL;DR

Phase 0 is **complete and verified**. Phase 1 is **built but not yet validated** --
the code is all there and the pieces work individually, but the experiment that
decides whether the AI is trustworthy has not been run at scale. That run is the
single next task.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                                   # 121 tests, ~23s, all passing
python experiments/validate_sandbox.py   # Phase 0 evidence
```

---

## What is done and verified

### Phase 0 -- the synthetic market laboratory ✅

Six generators, each producing markets with known statistical properties,
validated against theory rather than eyeballed.

| Generator | Contains | Verified against |
|---|---|---|
| `GBMGenerator` | nothing (the clean null) | rejection rates at exactly 5%, parameter recovery |
| `AR1Generator` | momentum, tunable strength | phi recovered to 5 s.e.; volatility invariant to phi |
| `OUGenerator` | mean reversion | stationary spread = sigma/sqrt(2·theta); VR < 1 |
| `RegimeSwitchingGenerator` | calm/stressed states + **true labels** | run lengths 100/33 days; per-state volatility |
| `JumpDiffusionGenerator` | nothing, but fat tails | kurtosis 24 vs GBM's ~0; no autocorrelation |
| `HestonGenerator` | forecastable *risk*, not direction | absolute-return autocorrelation with none in raw returns |

Two findings from this phase worth carrying forward:

**Seed 42 produces a "trending" market from a generator containing no signal.**
At a 5% significance level, one path in twenty is flagged by chance. This is why
the generator is validated by *rejection rate across 1000 paths*, never by single
paths — and it is the same logic the Null Test uses one level up.

**The `-sigma^2/2` volatility drag is worth 1.28%/year.** Omit it and the
generator silently hands out return nobody requested. `test_drift_includes_volatility_drag`
fails loudly if it ever goes missing.

### Phase 1 infrastructure ✅

- `sentinel/features/build.py` — causal features only; no centred windows, no
  full-sample standardisation
- `sentinel/engine/backtest.py` — explicit forward loop with costs, weight drift,
  and a drawdown circuit breaker
- `sentinel/engine/metrics.py` — CAGR, Sharpe, Sortino, drawdown depth *and duration*
- `sentinel/strategies/baseline.py` — the permanent control arm
- `sentinel/evaluation/causality.py` — automated lookahead detection

### The lookahead detector ✅ — read this before touching anything

`tests/test_no_lookahead.py` is the most important file in the repo. It works by
truncation: feed a strategy history up to day k, then feed it more, and check that
day k's decision did not move.

**It contains two deliberately-broken strategies** (`TomorrowPeeker`,
`FullSampleNormaliser`) that must always be caught. Do not delete them. During
development the peeker exposed a real blind spot in the detector — an earlier
version excluded each truncation's final row, which is *exactly* where a one-day
peek shows up. Without those cheat strategies that flaw would have shipped, and
every causality claim in the project would have been worthless.

---

## What is built but NOT verified ⚠️

**This is the honest part. Do not treat the AI as working.**

### The AI (`sentinel/ai/model.py`)

`WalkForwardModel` and `WalkForwardClassifier` are written, causal (tested), and
smoke-tested on exactly **two** markets:

| Market | Sharpe |
|---|---|
| GBM, mu=0 (null) | −0.43 |
| AR(1), phi=0.30 (strong planted signal) | +1.35 |

That is the right *shape* — loses on noise, finds a strong signal — but two runs
is not evidence. It is the same single-path error the whole project is built to
avoid, and it must not be quoted as a result.

### The Null Test (`sentinel/evaluation/null_test.py`)

The module is written and works. **It has never been run at scale.** The full
sweep across strategies × null markets was interrupted before completing.

**Running it is the Phase 1 exit criterion and the next task.**

One preliminary data point, from 100 markets on the *baseline* strategy
(`AbsoluteMomentum`, 1512 steps), which passed with mean Sharpe −0.03 (t = −0.64):

> **noise floor p95 = 0.688** — absolute momentum reaches Sharpe 0.69 on markets
> containing nothing, one time in twenty, by luck alone.

If that survives a fuller run it has an uncomfortable implication worth raising
explicitly: **`plan.md` §2 sets a target of Sharpe > 0.7, which is essentially at
the noise floor.** The target may need raising, or restating as "must exceed the
measured null floor" rather than a fixed number. Do not quietly ignore this.

---

## Next steps, in order

### 1. Run the full Null Test — the Phase 1 gate

Sweep `{BuyAndHold, AbsoluteMomentum, DualMomentum, WalkForwardModel,
WalkForwardClassifier}` × `{GBM mu=0, JumpDiffusion mu=0, Heston mu=0}` at
`n_markets=200, n_steps=2520`.

All three markets have `has_exploitable_signal=False`, and `run_null_test`
refuses any generator that does not — so the control cannot be run on a market
that secretly contains a signal.

- **Pass:** mean Sharpe not significantly positive (t < 3) on every combination.
- **Fail:** hunt for the leak before doing anything else. The most likely
  location is the label alignment in `WalkForwardModel.compute_weights` — a model
  used on day `t` may train on labels up to `y[t-1]` and no further.
- **Deliverable:** the noise floor per strategy, written to `reports/`. Every
  later result gets compared against these numbers.
- Then add it to `.github/workflows/tests.yml` (there is a comment marking the
  spot) so the build fails if the AI ever profits from noise.

Note `run_null_test` parallelises over processes; it took ~0.2s for 100 baseline
markets on 24 cores, and the AI is ~2s per market single-threaded.

### 2. Phase 2 — the Recovery Test

Sweep `AR1Generator` phi from 0 to ~0.3 and find where the AI stops detecting the
signal. That threshold is the AI's sensitivity.

Expect this to be discouraging, and report it honestly if so: real daily equity
autocorrelation is roughly 0.01–0.05 and unstable. If the AI needs phi = 0.15,
it cannot work on real markets. **Phase 2 is explicitly allowed to kill the
project** — cheaply, on fake money, with a clear reason.

Also worth testing here: `OUGenerator`, whose signal lives in the price *level*
rather than in returns. A returns-only model should be blind to it, and
confirming that blindness is a real finding about the feature set.

### 3. Not yet built

- **Regime classifier** (`sentinel/ai/regime/`) — the plan's centrepiece. The
  labels already exist: `RegimeSwitchingGenerator` ships `truth.regimes`, one per
  return. Score classification *accuracy* and **detection lag** (how many days
  after a switch before the model notices), not just profit. Impossible on real
  data; this is what the sandbox is for.
- **Anomaly detector** — must only ever *reduce* risk, never increase it.
- **Phase 3, adversarial** — `experiments/adversarial.py` does not exist. Crashes,
  correlation breakdown, regimes never seen in training.
- **Phase 4, real data** — no adapter yet. Same `MarketData` interface, so the AI
  cannot tell it apart from synthetic. Use total-return prices; raw prices bias
  every backtest downward by several percent a year.
- **Phases 5–6** — 6 months paper trading, then real money. Calendar-bound; no
  code can compress them.

---

## Conventions that must not be broken

1. **`MarketData` carries prices and nothing else.** `GroundTruth` is a *separate*
   object precisely so the answer key cannot leak into a model through a shared
   field. Never add a `truth` attribute to `MarketData`, and never let a strategy
   accept a `Scenario`.
2. **Timing:** `compute_weights` row `t` = weights held from `t` to `t+1`, using
   data up to and including `t`. The engine applies row `t` to the return from `t`
   to `t+1`. `test_engine_applies_yesterdays_decision_to_todays_return` pins this
   against a hand-computed answer.
3. **Any new strategy gets a causality test.** One line: `check_causality(strategy, data)`.
4. **The rules baseline runs in every experiment, permanently.** Not a phase to
   get past — the control arm. An AI that matches a two-parameter rule has learned
   nothing and is strictly worse, because it has far more ways to fail silently.
5. **Null markets use `mu=0`.** With positive drift, any strategy that holds
   assets makes money from beta rather than skill, and the Null Test would measure
   the wrong thing.
6. **`UNLIMITED` limits for measurement runs.** The risk layer would otherwise
   mask the behaviour being measured.
7. **Seeds fixed in every test.** A statistical test that redraws its own data
   fails ~1 run in 20 for no reason, and a suite that cries wolf gets ignored.
8. **Log decisions in `DECISIONS.md`,** including reversals. Never edit history —
   add a new dated entry.

---

## Environment notes

Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `lightgbm`, `pytest`.
Developed on x86-64 WSL2 / Python 3.12, CI on ubuntu-latest.

**On a DGX Spark (aarch64):** nothing here is GPU-bound and nothing needs CUDA.
The workload is many small LightGBM fits, which is CPU- and core-count-bound —
`run_null_test` parallelises across processes and will benefit directly from more
cores. If the `lightgbm` wheel does not resolve on ARM64, `scikit-learn`'s
`HistGradientBoostingRegressor` is the same algorithm and a drop-in replacement in
`sentinel/ai/model.py::_make_regressor`; adjust the hyperparameter names and
nothing else changes.

A GPU only becomes relevant at the Phase 5 experiments (sequence models,
reinforcement learning). RL is worth flagging as genuinely viable here despite
being impractical for most retail projects — it is enormously data-hungry, and an
unlimited simulator is exactly the condition it needs. Its specific risk is
learning to exploit the *simulator's* quirks rather than market structure, which
is why the plan puts it after the sandbox has been validated against real data.

---

## Standing reminders from `plan.md` §10

Predicted in advance so they can be checked rather than rationalised later:

1. The AI will "pass" the Null Test by making money the first time it runs. It
   will be a lookahead bug. It is always a lookahead bug.
2. The Phase 2 sensitivity threshold will likely be discouraging. That is
   information, obtained for free, on fake money.
3. Real markets will break something the simulator never did. That gap is the
   most interesting thing this project will produce.
