# Handoff

State of the project as of **2026-09-04**, written for whoever picks it up next.

Read `plan.md` for *why* the project is shaped this way, and `DECISIONS.md` for what changed and
why. This file covers what exists, what is verified, what is not, and what to do next.

---

## TL;DR

Phases 0, 1 and 2 are complete and validated, and the system has reached real market data. The
project has produced four findings that are worth more than the code:

1. **Nothing here profits from noise.** 27 strategy × market cells, all pass. The Phase 1 gate now
   runs on every commit.
2. **A Sharpe number without an evaluation window attached is meaningless.** The noise floor is
   `1.645/√years` plus cost drag: 1.60 at one year, 0.49 at ten.
3. **The return-forecasting AI is dead, and a two-parameter rule beat it.** It needs a signal three
   times stronger than real markets offer, and detects less than a 5-day moving average does.
4. **The simulator taught the regime strategy something false.** In the sandbox, high volatility
   means falling prices — because the generator was built that way. On real SPY the opposite holds.
   This is the single most valuable thing the project has produced, and it is the reason the
   Reality Ladder exists.

The current best real-data result is **`absolute_momentum`, Sharpe 0.798 against buy-and-hold's
0.654** over 32.9 years of SPY, clearing a bootstrap noise floor of 0.344. That is a two-parameter
rule, not a model, and it is the only thing here that has beaten the index on real data.

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

- **`VolatilityTarget` / `RegimeVolatilityTarget`** — written, causal, unit-tested, and they pass
  the Null Test on both GBM and Heston. They have **not** yet been run through the full real-data
  comparison, which is the next task. Do not quote them as working.
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

### 1. Re-run the real-data comparison with the volatility strategies

`python experiments/real_data.py`. They are already in the strategy list. The question is whether
sizing for volatility beats both the bootstrap floor and buy-and-hold on 32.9 years of SPY — and
whether the regime-based volatility forecast beats the plain 21-day realised one. If it does not,
the classifier has bought nothing over a rolling standard deviation and the simpler version wins.

### 2. Fix the generator the finding exposed

`RegimeSwitchingGenerator` cannot express "volatile but rising", which is the state real equity
markets are in most often. Decoupling `mu` from `sigma` — or adding a third state — makes the
sandbox able to represent reality rather than a caricature of it. Every regime result should then be
re-measured against the corrected generator; some of them will get worse, and that is the point.

### 3. Phase 3 — adversarial markets

`experiments/adversarial.py` does not exist. Crashes, correlation breakdown, regimes never seen in
training. Verify the circuit breaker fires and the anomaly detector notices. Note that the breaker
is already firing a lot on real data — 506 days for buy-and-hold over 33 years — and nobody has
checked whether that is helping or hurting.

### 4. Not yet built

- **Anomaly detector** — must only ever *reduce* risk. Never started.
- **Phase 5, paper trading** — six months of calendar time, no code can compress it. Needs the
  Alpaca adapter, which does not exist.

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
