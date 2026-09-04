# market-sentinel

An AI trading system developed inside a **synthetic market laboratory** — where the ground truth is
known — before it is ever pointed at a real market.

**Status:** Phases 0–2 complete and validated. The system has reached real market data (rung 2).
The return-forecasting AI has been **measured, found wanting, and retired**; the regime classifier
survives. See [HANDOFF.md](HANDOFF.md) for exactly what is verified and what is not.

## The idea

You cannot tell whether a model trained on real market history learned something or memorized
noise, because there is only one copy of history and no answer key.

So this project builds the markets first. If you generate a market yourself, you know exactly what
is in it — whether it holds a pattern, what kind, and how strong. Two experiments follow:

**The Null Test.** Generate a market of pure randomness, with nothing to find. Run the AI on it.
*The AI must fail.* If it reports profit on noise, that is not a strategy — it is a bug, caught on
day one on fake money instead of eighteen months later on real money. Run it hundreds of times and
you get a **noise floor**: what your method scores on markets containing nothing. Any real result
below that line is a coincidence, not a discovery.

**The Recovery Test.** Plant a pattern of known strength, then weaken it until the AI loses the
scent. That threshold is the AI's sensitivity. Real market signals are very weak — so if the
threshold sits above what real markets plausibly offer, you know the approach cannot work, from a
measurement rather than a hunch.

## What the measurements said

Every number below is reproducible from `experiments/`, with the raw output committed in
`reports/`.

### Phase 1 — nothing here profits from noise

Nine strategies × three signal-free market types × 200 markets of ten years each. **All 27 cells
pass.** The AI does not merely fail to profit on randomness; it loses significantly (mean Sharpe
−0.23 to −0.30, t between −11 and −14), which is correct — trading costs money and there is nothing
to pay for it.

`plan.md` §10 predicted the AI would appear to make money on the first Null Test run and that it
would turn out to be a lookahead bug. It did not happen, and the credit belongs to the causality
tests that caught the leaks during development instead.

### The noise floor depends on how long you measured for

The single most useful number the project has produced, because it makes every other result
interpretable:

    noise floor ≈ (cost drag) + 1.645 / √years

| Track record | 1y | 2y | 3y | 5y | 10y | 20y |
|---|---|---|---|---|---|---|
| Sharpe reachable by luck alone | +1.60 | +1.12 | +0.96 | +0.77 | +0.49 | +0.35 |

A Sharpe of 0.7 is a demanding result over twenty years, roughly a coin flip over six, and cannot
be evidence of anything over two. Most retail backtests are quoted over two to five years —
precisely the range where the number carries almost no information.

This also corrected a claim the project had been carrying: a busier strategy does **not** have a
wider null distribution. Spread is set by track length and nothing else; turnover shifts the
*centre* down by the cost it pays. So a busy strategy has a *lower* floor — which is a trap, since
it is lower only because the strategy is already losing money. Clearing your floor is necessary,
not sufficient.

### Phase 2 — how weak a signal can each strategy find?

AR(1) momentum planted at known strength, swept down. Detection is measured against each strategy's
own noise floor.

| Strategy | φ needed for 50% detection | Verdict |
|---|---|---|
| buy and hold | never | correct — it has no timing ability (the calibration check) |
| absolute momentum, 252d | never | **blind**, at any strength, to a one-day signal |
| short momentum, 5d | 0.124 | 2× the top of the plausible real range |
| walk-forward AI (regression) | 0.155 | 3× |
| walk-forward AI (classifier) | 0.159 | 3× |

Real daily equity autocorrelation is roughly 0.01–0.05, unstable, and not reliably of one sign. The
AI needs about three times the top of that range, so **it cannot work on real markets** — the
outcome `plan.md` §10 called most likely, obtained for free on fake money.

And the finding underneath it: **a two-parameter moving-average rule detects the signal better than
the gradient-boosted model with sixteen features and two hundred trees.** The AI learned nothing the
rule did not already have, costs more to run, and has vastly more ways to fail silently. That is the
case for deleting a model, and the permanent control arm is what made it visible.

### The regime classifier — the framing that survived

Not a better model of the same question. A different question: *which state is the market in*, rather
than *what is the price tomorrow*. Measured on 200 regime-switching markets against the generator's
true labels — impossible on real data, and the reason the sandbox exists.

| | |
|---|---|
| Balanced accuracy | 89.8% |
| AUC | 0.948 |
| Calibration gap | 0.019 — so the probability can size positions directly |
| **Median detection lag** | **3.2 days** (p90 6.2) |
| Switches caught | 92.8% |
| False alarms | 4.2% of calm days |

Detection lag is the metric that decides whether a classifier is usable, and it is almost never
reported: a model that is 95% accurate but three weeks late is right about the past and silent about
the present, and its accuracy score looks superb throughout. Measured against an oracle that is
perfect but *late*, being 32 days slow removes the entire edge.

Value, measured against a ceiling rather than in the abstract:

| | Sharpe | CAGR | max drawdown |
|---|---|---|---|
| buy and hold | +0.330 | +4.61% | −44.4% |
| **the real classifier** | **+0.548** | +4.73% | **−19.2%** |
| perfect, 4 days late | +0.683 | +7.56% | −22.3% |
| perfect, instant *(impossible)* | +0.847 | +8.69% | −18.6% |

The oracles are there so a weak result can be *attributed* rather than blamed — "the model is poor"
and "this market held little to find" look identical from a single backtest and call for opposite
responses.

## The Reality Ladder

Each rung adds exactly one element of reality, so any failure is attributable.

| Rung | Market | Money | Proves | Status |
|---|---|---|---|---|
| 1. Sandbox | Synthetic | Fake | The AI finds real signals — and correctly finds nothing in noise | ✅ |
| 2. Backtest | Real history | Fake | It survives real messiness: fat tails, gaps, costs, crashes | ✅ reached |
| 3. Paper | Real, live | Fake | The plumbing works in real time | not started |
| 4. Live | Real, live | Real | It works when it matters | not started |

The AI cannot tell which rung it is on — all three market sources share one interface, and there is
no branch anywhere that says "if this is real".

Real-data results are judged against a floor computed by **resampling that market's own returns**,
one day at a time. SPY's daily returns have an excess kurtosis around 11; a floor measured on
Gaussian simulations is the wrong floor for judging them.

## What this is not

- **Not financial advice.** Personal tool.
- **Not a guarantee against losses.** No such system exists; anything that can gain can lose.
- **Not a get-rich system.** It must beat a savings account, and it may fail to. The plan explicitly
  allows the conclusion "this doesn't work — buy an index fund", and Phase 2 has already reached
  that conclusion about one of the two approaches tried.

Cost through paper trading: **$0**.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

pytest                                     # 231 tests

python experiments/validate_sandbox.py     # Phase 0 evidence
python experiments/null_test.py            # Phase 1 gate — must pass, runs in CI
python experiments/noise_floor_scaling.py  # what a Sharpe number is worth at each length
python experiments/recovery_test.py        # Phase 2 sensitivity curves
python experiments/regime_test.py          # classification quality, lag, and the oracle ladder
python experiments/real_data.py            # rung 2: real ETF history
```

Every experiment takes `--quick` for a smoke test. The Null Test refuses to write a report from
fewer than 100 markets per cell, because a 95th percentile estimated from fewer is too noisy to
publish and every later phase compares against these numbers.

## Progress

- [x] **Phase 0** — six generators, validated against theory (rejection rates, parameter recovery)
- [x] **Phase 1** — features, engine, baselines, lookahead detector, the AI
- [x] **Phase 1 gate** — the full Null Test. 27 cells, all pass. Now runs on every commit.
- [x] **Phase 2** — the Recovery Test. Killed the return-forecasting branch; the regime branch lives.
- [x] **Regime classifier** — accuracy, calibration, detection lag, and value against an oracle ceiling
- [x] **Rung 2** — real ETF history, judged against a bootstrap floor from the same returns
- [ ] Phase 3 — adversarial markets
- [ ] Phase 5 — six months of paper trading (calendar time; no code can compress it)

New here? Start with [HANDOFF.md](HANDOFF.md).

## License

MIT — see [LICENSE](LICENSE).
