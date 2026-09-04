# market-sentinel

An AI trading system developed inside a **synthetic market laboratory** — where the ground truth is
known — before it is ever pointed at a real market.

**Status:** Phases 0–3 complete and validated, tested against 391 market-years of real data across
eight countries. The headline result is a negative one, arrived at honestly:

> **Trend following is a risk-reduction tool, not a return-generation tool.** Its drawdown benefit
> replicates in 8 of 8 markets (p = 0.0039). Its return edge does not (p = 0.145). Nothing in this
> project is ready for real money, and the plan anticipated that and permits it.

The return-forecasting AI was measured, found wanting, and retired. The simulator was caught
teaching a false lesson, corrected, and re-verified. See [HANDOFF.md](HANDOFF.md) for exactly what
is verified and what is not.

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

### The simulator was teaching a false lesson

Then the classifier met real data and lost to buy-and-hold. The reason turned out not to be the
classifier:

| Classifier's state | sandbox | real SPY |
|---|---|---|
| calm | +9.2%/yr | +9.0%/yr |
| **stressed** | **−20.9%/yr** | **+17.1%/yr** |

`RegimeSwitchingGenerator` was built with `mu=(0.12, −0.15)` and `sigma=(0.12, 0.32)` — calm means
rising *and* quiet, stressed means falling *and* volatile. The two are welded together by
construction, so inside the sandbox a strategy that flees volatility is automatically fleeing
losses. Real equity volatility is *compensated*; sorting by trailing realised volatility with no
model at all gives the same answer.

Every rung-1 result was correct, and none of it transferred. This is `plan.md` §10's third
prediction — *"real markets will break something the simulator never did; that gap is the most
interesting thing this project will produce"* — arriving on schedule, and it is attributable to one
line of generator configuration precisely because the strategies, engine, costs and metrics are
identical code at both rungs.

**How much it mattered**, measured by whether each sandbox ranks strategies the way reality does:

| | rank correlation with real SPY |
|---|---|
| classic sandbox (`mu = +0.12 / −0.15`) | **−0.143** |
| corrected sandbox (`mu = +0.09 / +0.17`) | **+0.750** |

The original was not merely uninformative — it was *negatively* correlated with reality. Acting on
its rankings was worse than choosing at random. The fix changed no strategy and no engine code; it
changed two numbers in a generator's default arguments.

### Phase 3 — adversarial markets

Every crash scenario run twice, with the risk layer on and off, so the breaker's value is a number
rather than a reassurance:

| Scenario | Drawdown saved | Return given up |
|---|---|---|
| 35% over 60 days | +6.1% | 0.72%/yr |
| 50% over 20 days | +12.5% | 2.33%/yr |
| **35% over 1 day** | **−1.5%** | **−1.09%/yr** |

Against an overnight gap the breaker is **actively harmful**: it cannot act until the damage is
done, then sells the bottom and sits in cash through the rebound. No parameter choice removes that —
the mechanism is trailing by construction. The honest response is position sizing that survives a
gap, not a better breaker, and the failure is pinned in the test suite so it cannot be rediscovered
expensively.

Correlation breakdown was also measured, and it invalidates nothing so much as it adds a caveat.
The same four-asset equally weighted portfolio: −15.4% max drawdown when correlations stay at 0.2,
−32.8% when they rush to 0.95 during stress. Every fixed-correlation result in this project
understates risk by roughly that much.

### Rung 2 — 33 years of real SPY

Judged against each strategy's own bootstrap floor, computed by resampling SPY's own returns:

| Strategy | Sharpe | floor | CAGR | max DD | complexity |
|---|---|---|---|---|---|
| **absolute_momentum** | **+0.798** | +0.344 | +9.84% | −27.8% | two parameters |
| **volatility_target** | **+0.731** | +0.384 | +7.93% | −39.4% | one parameter |
| buy_and_hold | +0.654 | +0.414 | +9.30% | −51.0% | none |
| regime_volatility_target | +0.583 | +0.372 | +5.88% | −34.0% | an HMM |
| regime_aware | +0.504 | +0.387 | +4.10% | −40.0% | an HMM |
| ai_walkforward | +0.055 | +0.079 | +0.15% | −30.3% | LightGBM, 16 features |

**The ordering is almost perfectly inverse to complexity.** The multi-asset ladder says the same
thing independently: rotating into cash (0.761) beats rotating into treasuries (0.724), which beats
adding momentum-based asset selection (0.394). A fitted two-state HMM forecasts volatility *worse*
than a 21-day rolling standard deviation.

That is not an argument that models never work. It is exactly what the permanent control arm was put
there to detect, and it detected it every time it was asked.

### Does it replicate? — 8 countries, 391 market-years

`is_it_timing.py` (below) reached p = 0.13 on SPY and said the next step had to be more independent
paths: more backtesting on one path cannot improve a number the shuffle has already extracted. So
the same test was run on eight national indices — including the Nikkei, which spent thirty years
going down.

| | `absolute_momentum` timing edge | percentile |
|---|---|---|
| US S&P 500 (1927) | +0.172 | 94% |
| Japan Nikkei 225 (1965) | +0.223 | 96% |
| Canada TSX (1979) | −0.075 | 32% |
| UK FTSE 100 (1984) | +0.040 | 60% |
| Hong Kong Hang Seng (1986) | +0.054 | 64% |
| Germany DAX (1987) | +0.036 | 59% |
| France CAC 40 (1990) | +0.027 | 56% |
| Australia ASX 200 (1992) | −0.129 | 22% |
| **positive in 6 of 8** | median +0.038 | **sign test p = 0.145** |

**The return edge does not replicate.** But the drawdown does, in **8 of 8 markets**, by a median of
17.6 points — sign test **p = 0.0039**, the only significant result in the project.

The two differ for a reason that is the whole point. The drawdown benefit does not require the rule
to predict anything: stepping aside after a sustained decline mechanically truncates a long fall,
whether or not forecastable structure exists. It is a property of the rule's *shape*, which is
exactly why it survives when the return edge does not.

It is not free — a median 1.96% a year across these markets, plus sitting in cash through part of
every recovery. Whether roughly halving the worst drawdown is worth roughly two points of annual
return is a question about the person holding it, not about the data.

### The harder question underneath it: is it timing, or just being invested?

The floors above are measured on **demeaned** returns, the convention every null market here uses.
Applied to a real-data result that bar means less than it looks. Buy-and-hold scores 0.654 against a
demeaned floor of 0.414 — it "beats its noise floor" while doing nothing but being invested in a
market that went up. If buy-and-hold clears a bar, clearing it is not evidence of skill.

So the returns were reshuffled again with **the drift left in**. Ordering destroyed, so there is
nothing to time; market still rises, so exposure is still paid for. Whatever a strategy scores there
is what its average exposure earns, and the gap to its real score is the part attributable to using
the order of returns.

| Strategy | real | shuffled | timing edge | percentile |
|---|---|---|---|---|
| buy_and_hold | +0.637 | +0.636 | +0.001 | 52% ← calibration check |
| **absolute_momentum** | +0.769 | +0.526 | **+0.243** | **87%** |
| volatility_target | +0.742 | +0.599 | +0.143 | 79% |
| regime_aware | +0.453 | +0.607 | −0.154 | 18% |
| short_momentum | +0.040 | +0.299 | −0.259 | 10% |

**Nothing here reaches significance.** The best candidate sits at p = 0.13. Two strategies did
*worse* than shuffling their own market — `regime_aware`'s apparent floor-clearing was pure
exposure, and its timing actively cost 0.15 of Sharpe.

What this does *not* reduce is the drawdown result: −27.8% against buy-and-hold's −51.0%. Roughly
halving the worst loss follows mechanically from stepping aside after a sustained decline. It holds
whether or not the timing edge is real, and for a project whose stated goal is capital preservation
rather than return maximisation, it may be the more relevant number.

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

pytest                                     # 267 tests

python experiments/validate_sandbox.py     # Phase 0 evidence
python experiments/null_test.py            # Phase 1 gate — must pass, runs in CI
python experiments/noise_floor_scaling.py  # what a Sharpe number is worth at each length
python experiments/recovery_test.py        # Phase 2 sensitivity curves
python experiments/regime_test.py          # classification quality, lag, and the oracle ladder
python experiments/real_data.py            # rung 2: real ETF history
python experiments/simulator_gap.py        # where reality diverged from the simulator
python experiments/corrected_sandbox.py    # does the fixed sandbox predict reality?
python experiments/adversarial.py          # Phase 3: crashes and correlation breakdown
python experiments/is_it_timing.py         # the harder floor: timing, or just exposure?
python experiments/more_paths.py           # does it replicate? 8 countries, 391 market-years
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
- [x] **Rung 2 diagnosis** — found the simulator's false lesson, corrected it, re-verified
- [x] **Phase 3** — adversarial markets: crashes, gap risk, correlation breakdown
- [x] **Replication** — 8 countries, 391 market-years. Return edge fails, drawdown holds.
- [ ] Phase 5 — six months of paper trading (calendar time; no code can compress it)

## What would actually be traded, if anything

On the evidence: **nothing, for return.** No strategy here clears a noise floor that declines to
credit it for simply being invested in a market that went up.

For *risk*, one thing does: **absolute momentum** — hold while the trailing 12-month return is
positive, else cash. Two parameters, checkable by hand in a spreadsheet, and a rule a beginner could
write in an afternoon. Across eight countries it roughly halved the worst drawdown, every time, for
a median cost of about two points of annual return.

That is a real finding and it is not the one the project set out to get. The plan said it must be
able to reach the conclusion *"this doesn't work — buy an index fund"*, and on returns it has. What
it also produced is a set of instruments — the Null Test, the noise floor with its window attached,
the truncation detector, the drift-preserving shuffle, the oracle ladder — that caught every false
positive this project generated, including several of its own. Most of those false positives looked
convincing first.

The next honest test is paper trading, where the answer is not yet known to anybody.

New here? Start with [HANDOFF.md](HANDOFF.md).

## License

MIT — see [LICENSE](LICENSE).
