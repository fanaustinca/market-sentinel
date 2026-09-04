# Handoff

State of the project as of **2026-09-04**, written for whoever picks it up next.

Read `plan.md` for *why* the project is shaped this way, and `DECISIONS.md` for what changed and
why. This file covers what exists, what is verified, what is not, and what to do next.

---

## TL;DR

Phases 0 through 3 are complete and validated, and the system has been run against 33 years of
real market data. The project has produced five findings that are worth more than the code:

1. **Nothing here profits from noise.** 27 strategy × market cells, all pass. The Phase 1 gate now
   runs on every commit.
2. **A Sharpe number without an evaluation window attached is meaningless.** The noise floor is
   `1.645/√years` plus cost drag: 1.60 at one year, 0.49 at ten.
3. **The return-forecasting AI is dead, and a two-parameter rule beat it.** It needs a signal three
   times stronger than real markets offer, and detects less than a 5-day moving average does.
4. **The simulator taught the regime strategy something false.** In the sandbox, high volatility
   means falling prices — because the generator was built that way. On real SPY the opposite holds.
   Correcting two numbers in the generator's defaults moved its rank correlation with reality from
   **−0.143 to +0.750**: it had been worse than useless. This is the single most valuable thing the
   project has produced and the reason the Reality Ladder exists.
5. **Every piece of machinery added has made real results worse.** The real-SPY ranking is almost
   perfectly inverse to complexity, and the control arm caught it every time it was asked.

6. **Against a floor that gives no credit for being invested, nothing here is significant.** The
   demeaned floors flatter every strategy; reshuffled *with drift preserved*, `absolute_momentum`'s
   timing edge is p = 0.13 and `regime_aware`'s is negative.

The best candidate is **`absolute_momentum`** — two parameters, checkable by hand. What survives
scrutiny is not its return but its **drawdown**: −27.8% against buy-and-hold's −51.0%, which follows
mechanically from stepping aside after a sustained decline and does not depend on the timing edge
being real. Do not carry the Sharpe number forward without the p-value attached.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest                                   # 231 tests
python experiments/null_test.py          # the Phase 1 gate
```

---

## What is done and verified

### Phase 0 — the synthetic market laboratory ✅

Six generators with known statistical properties, validated against theory. Unchanged from the
previous handoff; see `reports/2026-09-03-sandbox-validation.txt`. A seventh has since been added:

| Generator | Contains | Verified against |
|---|---|---|
| `BootstrapGenerator` | real returns, reshuffled | fat tails preserved, autocorrelation destroyed |

The bootstrap is the bridge to rung 2. Resampling one day at a time keeps SPY's real return
distribution — excess kurtosis around 11 — while removing everything predictable, so a real-data
result can be judged against a floor measured on the right distribution. Blocks longer than a day
preserve serial structure, so `has_exploitable_signal` is `True` there and `run_null_test` refuses
it. That refusal is load-bearing: a floor inflated by preserved structure would excuse a broken
strategy rather than catch it.

### Phase 1 gate — the Null Test, run at scale ✅

`reports/2026-09-04-null-test.txt`. Every strategy × {GBM, jump-diffusion, Heston} at `mu=0`, 200
markets of 2520 days. **All 27 cells pass.**

`plan.md` §10 predicted the AI would appear profitable on the first run and that it would be a
lookahead bug. It was not — the AI loses on noise at t between −11 and −14. The causality tests had
already caught the leaks during development, which is where that prediction was actually spent.

Now wired into CI at a reduced size (60 markets × 1260 days), which detects a leak comfortably: a
peeking strategy scores t > 20 against a threshold of 3.

### The noise floor scales with track length ✅

`reports/2026-09-04-noise-floor-scaling.txt`. This resolved the open question the previous handoff
flagged — a floor of 0.688 in one run and 0.506 in another. Both were right; the first was a
six-year window, the second ten.

    floor ≈ (cost drag) + 1.645/√years

Measured against predicted across three strategies and seven track lengths: mean deviation 0.048
Sharpe, worst 0.179. **`plan.md` §2's "Sharpe > 0.7" is now read as "> 0.7 over at least ten years,
and above the measured floor for that window."**

This also reversed a claim the codebase carried: busier strategies do *not* have wider null
distributions. Spread is set by track length alone; turnover shifts the centre down via costs. So a
busy strategy has a *lower* floor, which is a trap — it is lower only because the strategy is
already losing. Every result is reported against both the floor and zero.

### Phase 2 — the Recovery Test ✅

`reports/2026-09-04-recovery-test.txt`. Detection thresholds on AR(1) markets:

| Strategy | φ @ 50% detection |
|---|---|
| buy and hold | never (the calibration check — it has no timing ability) |
| absolute momentum, 252d | never — **blind** at any strength |
| short momentum, 5d | **0.124** |
| walk-forward AI (regression) | 0.155 |
| walk-forward AI (classifier) | 0.159 |

Real daily equity autocorrelation is 0.01–0.05, unstable, and not reliably signed. The AI needs
about 3× the top of that range. **It cannot work on real markets**, and a two-parameter rule detects
better than it does. Both models are retired as candidates and kept only as control arms.

### The regime classifier ✅ measured, ⚠️ does not transfer

`reports/2026-09-04-regime.txt`. On synthetic regime markets it is genuinely good: 89.8% balanced
accuracy, AUC 0.948, calibration gap 0.019, **median detection lag 3.2 days**, 92.8% of switches
caught. The sharpness sweep lands exactly on its calibration check — at zero volatility difference
between states, balanced accuracy is 50.3%.

On real SPY it produces Sharpe 0.504 against buy-and-hold's 0.654. It clears its noise floor and
loses to the index. **See the next section for why — the reason is not that the classifier is bad.**

---

## The most important finding: what the simulator taught the strategy

`experiments/simulator_gap.py`, `reports/simulator_gap.json`.

`RegimeSwitchingGenerator` is built with `mu = (0.12, -0.15)` and `sigma = (0.12, 0.32)`. Calm means
rising and quiet; stressed means falling and volatile. **The two properties are welded together by
construction.** So at rung 1, a strategy that flees volatility is automatically fleeing losses, and
every experiment rewards it for doing so.

Real markets do not honour that coupling:

| Classifier's state | sandbox annual return | real SPY annual return |
|---|---|---|
| calm | +9.2% | +9.0% |
| stressed | **−20.9%** | **+17.1%** |

The stressed state has a *higher* forward return on real data. Sorting by trailing realised
volatility instead — no model at all — gives the same answer: SPY's highest-volatility quintile has
its highest next-day return. It is the volatility risk premium, and selling into it means selling
exactly the periods you are paid to hold.

Every rung-1 result was correct and none of it transferred, because all of it was measured inside a
market whose author had already decided volatility and loss were the same thing. This is `plan.md`
§10 prediction 3 arriving on schedule, and it is attributable to one named assumption precisely
because nothing else changed between rungs.

**The response is not to abandon the classifier.** It is accurate; the mapping from state to
position was wrong. Volatility should *scale* exposure, not switch it off — which is §4's
"uncertainty shrinks positions" applied to the quantity that actually varies.
`sentinel/strategies/volatility.py` implements it and is the current live hypothesis.

---

## What is built but NOT verified ⚠️

- **Nothing is verified for real money.** Everything below rung 3 is a backtest, and the honest
  status of every number in `reports/` is "measured on data whose outcome was already known".
- **`RegimeVolatilityTarget`** — works, and loses to the plain `VolatilityTarget` by 0.15 of Sharpe
  on real SPY. A fitted two-state HMM forecasts volatility *worse* than a 21-day rolling standard
  deviation. Kept as a control arm; not a candidate.
- **The corrected sandbox is not out-of-sample.** `equity_like()` was calibrated on the same SPY
  history the +0.750 rank correlation was measured against. It shows the correction is
  self-consistent, not that it will hold on data nobody has seen.
- **`RegimeRotation`** (multi-asset) — causal and tested mechanically. Its premise, that treasuries
  rise when equities fall, is a fact about 2000, 2008 and 2020 and was false in 2022. The crisis
  table in the real-data report exists to keep that visible.

### A lookahead bug caught during this session, worth reading

`RegimeVolatilityTarget.forecast_volatility` was first written to read
`classifier.last_parameters` — the parameters from the most recent refit — and apply them to every
row. `check_causality` reported `LOOKAHEAD DETECTED at row 756, drift 2.67e-02` within seconds.

Nothing about the code looked wrong. `last_parameters` reads like an accessor, and the volatility
forecast it produced was entirely plausible. This is the third time the truncation detector has
caught something code review would not have, and it is why every new strategy gets
`check_causality(strategy, data)` as a one-line test.

---

## Next steps, in order

### 1. Get more independent paths

The single most valuable thing available, and the shuffle experiment says why: it has already
extracted what SPY 1993-2025 can tell us, and the answer was p = 0.13. More backtesting on the same
path cannot improve that number — only more paths can.

Cheapest sources, in order: pre-1993 US index data (from a total-return series rather than SPY,
which did not exist), other developed indices via the same yfinance adapter, and other asset classes.
Each is an approximately independent draw of the same question. Note that they are not fully
independent — global equity markets are correlated, and 2008 happened everywhere at once — so treat
five indices as worth rather less than five times one.

### 2. Find the next thing the simulator is missing

The corrected sandbox ranks `absolute_momentum` third; real SPY ranks it first. The regime generator
has no long-horizon trend structure for a 252-day rule to work with, so it cannot represent whatever
real markets are offering there.

Two possibilities, and they call for opposite actions. Either there is a genuine trend premium the
sandbox cannot express — in which case add it and re-measure everything — or 0.798 over 33 years is
luck, and the floor for that window is +0.29. The way to tell them apart is a generator that *can*
express long-horizon momentum, then checking whether the real result sits inside what that generator
produces by chance. This is the same question the Null Test answers, one level up.

### 3. Re-measure the multi-asset results against correlation breakdown

Every rung-1 multi-asset number was produced on a fixed-correlation market and understates drawdown
by roughly 17 points. `CorrelationBreakdownGenerator` now exists; the numbers have not been redone.

### 4. Phase 5 — paper trading

Six months of calendar time, no code can compress it, and it is the only genuinely out-of-sample
test available. Needs an Alpaca adapter behind the same `MarketData` interface, which does not
exist. Carry `absolute_momentum` and `volatility_target`; carry the baselines beside them as always.

Before starting, write down the expected range for each strategy from the numbers in `reports/`.
Divergence from a prediction made in advance is a bug to find; divergence from a number recalled
afterwards is a story to tell, and the difference matters.

### 5. Not yet built

- **Anomaly detector** — must only ever *reduce* risk. Never started.
- **Position sizing for gap risk.** Phase 3 showed the drawdown breaker is worse than useless
  against an overnight move. Nothing in the system currently addresses that, and no breaker
  parameter can.

---

## Conventions that must not be broken

1. **`MarketData` carries prices and nothing else.** `GroundTruth` is a separate object so the
   answer key cannot leak through a shared field. Never add a `truth` attribute to `MarketData`;
   never let a strategy accept a `Scenario`. The one exception is `sentinel/evaluation/oracle.py`,
   which is handed the answer key deliberately to measure ceilings, lives outside `strategies/`, and
   must never be quoted as achievable.
2. **Timing:** `compute_weights` row `t` = weights held from `t` to `t+1`, using data up to and
   including `t`.
3. **Any new strategy gets a causality test.** One line: `check_causality(strategy, data)`. It has
   now caught three real bugs that review missed.
4. **The rules baselines run in every experiment, permanently.** They are what revealed the AI was
   worse than a moving average.
5. **Null markets use `mu=0`.** With drift, holding assets earns from beta rather than skill.
6. **`UNLIMITED` limits for measurement runs**, real limits for rung-2 runs meant to represent what
   would actually have been traded.
7. **Seeds fixed in every test.**
8. **Every result carries its evaluation window and its noise floor.** A Sharpe alone is not a
   result.
9. **Real-data results carry a data fingerprint.** Yahoo restates adjusted history; the price cache
   is committed so a fingerprint can actually be checked.
10. **Log decisions in `DECISIONS.md`,** including reversals. Never edit history — add a dated entry.

---

## Environment notes

Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `lightgbm`, `yfinance`, `pyarrow`,
`pytest`. Developed and validated on **aarch64 (DGX Spark), Python 3.12**; CI on ubuntu-latest.
The previous handoff worried the `lightgbm` wheel might not resolve on ARM64 — it does, 4.7.0, and
the `HistGradientBoostingRegressor` fallback was not needed.

Nothing is GPU-bound. The workload is many small model fits, which is core-count-bound;
`sweep_markets` parallelises across processes. Runtimes on 20 cores: full Null Test ~470s, Recovery
Test ~800s, regime experiment ~460s, real data ~15 minutes (dominated by the AI on 8287-day
bootstrap markets).

Adding an import means adding a line to `requirements.txt` **and** `pyproject.toml`.

---

## Standing reminders from `plan.md` §10 — scored

1. *"The AI will pass the Null Test by making money the first time. It will be a lookahead bug."*
   **Did not happen.** The causality tests caught the leaks earlier, which is where the prediction
   was really spent.
2. *"The Phase 2 sensitivity threshold will be discouraging."* **Correct.** 3× what real markets
   offer, and beaten by a two-parameter rule.
3. *"Real markets will break something the simulator never did. That gap is the most interesting
   thing this project will produce."* **Correct, and it was.** See the section above.
4. *"The model will be overconfident before calibration."* **Wrong, for this model.** The HMM's
   probabilities came out well calibrated with no post-hoc correction — mean gap 0.019. Worth
   revisiting if a model with more capacity is ever used.
