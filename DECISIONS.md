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

---

## 2026-09-04 — Phase 1 gate passed: nothing in the project profits from noise

**Decision:** The Null Test has been run at scale and Phase 1 is closed. Results in
`reports/2026-09-04-null-test.txt` and `reports/noise_floor.json`.

**What was run:** every strategy × {GBM, jump-diffusion, Heston}, all at `mu=0`, 200 markets
of 2520 days each. 18 cells, all PASS at t < 3.

**Why it matters that this was boring:** `plan.md` §10 predicted the AI would appear to make
money on the first Null Test run and that it would turn out to be a lookahead bug. It did not
happen. The AI's mean Sharpe on noise is −0.23 to −0.30 with t between −11 and −14 — it loses,
consistently and significantly, which is the correct behaviour. The prediction was reasonable and
the outcome is better; the credit belongs to the causality tests, which caught the lookahead
during development rather than here.

**The number worth carrying forward:** the AI's loss on noise is not zero but −0.25 Sharpe, and
that is trading cost, not bad luck. It retrains and re-sizes daily, so it turns over far more than
any baseline. **Before the AI produces its first unit of value it has to earn back a quarter of a
Sharpe point.** Any future comparison against a monthly-rebalanced baseline must account for that
handicap explicitly rather than treating both as starting from zero.

---

## 2026-09-04 — REVERSES the claim that a busier strategy has a wider null distribution

**Decision:** The noise floor's *spread* is set by the length of the track record and essentially
nothing else. Turnover moves the distribution's *centre*, not its width. The docstring in
`sentinel/evaluation/null_test.py` claimed the opposite and has been corrected.

**Why:** Measured, not reasoned about. Across constant-weight strategies at exposures from 0.1 to
1.0, and momentum rules turning over between 0.8 and 84 times a year, the standard deviation of
null Sharpe stayed at 0.31 ± 0.01 on ten-year markets — against a theoretical `1/sqrt(years)` =
0.316. Exposure scales mean and volatility together and cancels out of a ratio.

What turnover does is pay costs on markets that offer nothing back: 0.8 turns a year costs 0.04
Sharpe, 84 turns costs 0.375. Since the floor is `mean + 1.645·sd`, a busier strategy ends up with
a **lower** floor.

**Why the correction is load-bearing rather than pedantic:** the original claim implies "busy
strategy, higher bar", which is reassuring and wrong. The truth is a trap — a busy strategy clears
its own floor *more easily*, purely because it starts out losing money to costs. Clearing the noise
floor is therefore necessary but not sufficient, and every result must be reported against both the
floor and zero. Acting on the old claim would have set the AI's bar too high and the baselines' too
low, which is exactly backwards.

---

## 2026-09-04 — A Sharpe target without an evaluation window is not a target

**Decision:** `plan.md` §2's "Sharpe > 0.7" is restated as "> 0.7 measured over at least ten years,
and above the measured null floor for that same window." Evidence in
`reports/2026-09-04-noise-floor-scaling.txt`.

**Why:** This resolves the open question flagged in `HANDOFF.md`, which recorded a floor of 0.688
from one run and 0.506 from another and could not reconcile them. Both are right. The first was
measured on six-year markets, the second on ten-year markets, and

    floor ≈ (cost drag) + 1.645/sqrt(years)

gives 0.67 at six years and 0.52 at ten. The discrepancy was never a discrepancy.

The consequence is sharper than it first appears. A Sharpe of 0.7 is a demanding result over twenty
years, roughly a coin flip over six, and cannot constitute evidence of anything over two — the
one-year floor is 1.64. Most retail backtests are quoted over two to five years, which is precisely
the range where the number carries almost no information. Every result this project reports now
carries its window length, and is compared against the floor for that window rather than a
remembered constant.

---

## 2026-09-04 — The control arm needs a short-horizon rule as well as a long one

**Decision:** `ShortHorizonMomentum` (5-day lookback, daily action) joins the permanent control arm
beside the conventional 252-day `AbsoluteMomentum`.

**Why:** A rule can only see signals that live on its own timescale, and the effect is not a matter
of degree. Against a planted AR(1) signal at phi = 0.3 — an order of magnitude stronger than
anything real markets offer — detection rates were:

    lookback 252, monthly    14%   (the null rate: blind)
    lookback  60, monthly    14%
    lookback  20, weekly     16%
    lookback   5, daily     100%
    lookback   2, daily     100%

The twelve-month rule is not weak here. It is blind, at the chance rate, at any strength. AR(1)
momentum acts at a one-day lag and a 252-day average cannot represent it.

Running only the conventional horizon in the Recovery Test would have produced a flat curve for
every strategy and the conclusion "this signal cannot be detected" — when the truth was "nothing we
ran was capable of detecting it". That error reads as evidence to kill the project, which makes it
considerably more expensive than a missing feature.

**The cost, stated so it is not forgotten:** the short rule turns over ~50 times a year, worth about
0.24 of Sharpe. It must find something real merely to break even, and its null floor is
correspondingly lower.

---

## 2026-09-04 — Phase 2 result: the return-forecasting AI is dead. The regime framing is not.

**Decision:** `WalkForwardModel` and `WalkForwardClassifier` are retired as candidates for real
money. They stay in the repository as permanent control arms and as the subject of the Null Test,
but no further work goes into making them better. Evidence in
`reports/2026-09-04-recovery-test.txt`.

**Why:** The Recovery Test measured the signal strength each strategy needs before it detects
anything more than half the time, on AR(1) markets where the signal strength is known exactly:

    strategy                     phi @ 50% detection
    buy_and_hold                 never detects (the calibration check — it has no timing ability)
    absolute_momentum (252d)     never detects (blind to a one-day signal at any strength)
    short_momentum (5d)          0.124
    ai_walkforward               0.155
    ai_walkforward_classifier    0.159

Real daily equity autocorrelation is roughly 0.01–0.05, unstable, and not reliably of one sign. The
AI needs about **three times the top of that range**. This is the outcome `plan.md` §10 predicted as
most likely, it arrived for free on fake money, and it is a real finding rather than a
disappointment.

**The finding underneath the finding:** a two-parameter moving-average rule detects the signal at
phi = 0.124 — **better than the gradient-boosted model with sixteen features and two hundred
trees**. The AI has not learned anything the rule does not already have, it costs more to run, and it
has vastly more ways to fail silently. This is exactly the case `plan.md` §4 describes as grounds for
deleting the model, and the control arm is what made it visible. Without a baseline in every
experiment, "the AI detects at phi = 0.155" would have read as a capability.

**What survives, and why this does not kill the project:** the regime classifier is a different
question, not a better model of the same question. It does not forecast returns; it estimates which
state the market is in. On regime-switching markets it reaches 90% causal accuracy with a four-day
median detection lag and captures roughly two-thirds of what a perfect oracle would earn — measured
in `reports/2026-09-04-regime.txt`. `plan.md` §4 argued in advance that "is this a calm market or a
stressed one" is far more tractable than "what is the price tomorrow". That is now measured rather
than asserted, and the project continues down the regime branch only.

---

## 2026-09-04 — CORRECTS the handoff note claiming the feature set is blind to mean reversion

**Decision:** The claim in `HANDOFF.md` that "a returns-only model should be blind to" the
Ornstein-Uhlenbeck generator is wrong, and the reason is worth keeping.

**Why:** The prediction was that OU's signal lives in the price *level* while the features are built
from returns, so the model could not see it. Measured, the momentum rules are indeed completely blind
— detection stays at or below 5% at every mean-reversion speed, and their Sharpe gets steadily
*worse* as theta rises, because a momentum rule on a mean-reverting market is systematically wrong
rather than merely uninformed. That part held.

But the AI does detect it, at theta = 5.29. The feature set is not returns-only:
`distance_ma_{window}d` is exactly a price-level feature — where today's price sits relative to its
own trailing average, in units of trailing volatility — and that is precisely the statistic an OU
process is predictable from.

The threshold is still hopeless in absolute terms (theta = 5.29 is a 33-day half-life, roughly 106×
anything real markets offer), so this changes no conclusion. It is logged because the reasoning was
wrong for a checkable reason: the feature set was described from memory rather than read. A
prediction written down in advance and then measured is how that gets caught.

---

## 2026-09-04 — THE SIMULATOR TAUGHT THE STRATEGY SOMETHING FALSE

**Decision:** The regime strategies' core assumption — that a high-volatility market is one to sit
out — is wrong on real equities, and every strategy built on it is reclassified as unverified.
Evidence in `experiments/simulator_gap.py` and `reports/simulator_gap.json`.

**What was measured:** the classifier's own states, profiled by forward return.

| Classifier's state | sandbox regime market | real SPY 1993–2025 |
|---|---|---|
| calm | +9.2%/yr, 14.0% vol | +9.0%/yr, 13.4% vol |
| stressed | **−20.9%/yr**, 29.8% vol | **+17.1%/yr**, 27.8% vol |

The state the classifier calls stressed has a *higher* forward return than the calm state on real
data. Sorting by trailing 21-day realised volatility instead — no model involved, nothing to fit —
gives the same answer: SPY's highest-volatility quintile has its highest next-day return. So it is a
property of the market, not an artefact of the classifier.

**Why the sandbox said otherwise:** `RegimeSwitchingGenerator` is constructed with
`mu = (0.12, -0.15)` and `sigma = (0.12, 0.32)`. Calm means rising *and* quiet; stressed means
falling *and* volatile. The two properties are welded together by construction, so inside the
sandbox a strategy that flees volatility is automatically fleeing losses, and every rung-1
experiment rewarded it for doing so.

None of those rung-1 results were wrong. The classifier really does reach 89.8% balanced accuracy
with a 3.2-day median lag; the oracle ladder really does show it capturing a large share of the
available edge. All of it was measured inside a market whose author had already decided that
volatility and loss are the same thing, and none of it transferred.

**Why this is the project's most valuable output so far:** `plan.md` §10 predicted "real markets will
break something the simulator never did — that gap is the most interesting thing this project will
produce". It is, and the gap is attributable to one named line of generator configuration precisely
because nothing else changed between rungs. The strategies, engine, cost model, risk layer and
metrics are the identical code at both. That attributability is the entire argument for the Reality
Ladder, and it just paid for itself.

**What follows.** Two things, and neither is "abandon the classifier":

1. **Change the mapping from state to position.** Volatility should *scale* exposure, not switch it
   off. On real SPY the calm state runs 13.4% vol for 9.0% return and the stressed state 27.8% for
   17.1% — Sharpe 0.67 and 0.62, near enough identical. The market is not offering a better deal in
   either state; it is offering the same deal at two different sizes. `VolatilityTarget` and
   `RegimeVolatilityTarget` in `sentinel/strategies/volatility.py` implement that. Both are causal,
   both pass the Null Test on GBM and on Heston, and neither is verified on real data yet.
2. **Fix the generator.** A regime model that cannot express "volatile but rising" cannot represent
   the state real equity markets are in most often. Decoupling `mu` from `sigma`, or adding a third
   state, makes the sandbox able to represent reality rather than a caricature. Every regime result
   should then be re-measured; some will get worse, and that is the point of doing it.

---

## 2026-09-04 — Two bugs the mechanical checks caught that review would not have

Logged because both were invisible on inspection, and both would have produced plausible,
wrong numbers rather than a crash.

**A lookahead in `RegimeVolatilityTarget`.** The volatility forecast was first written to read
`classifier.last_parameters` — the parameters from the most recent refit — and apply them to every
row, including rows from years earlier. `check_causality` reported
`LOOKAHEAD DETECTED at row 756, drift 2.67e-02` within seconds of it being written.

Nothing about the code looked wrong. `last_parameters` reads like an accessor and the volatility
forecast it produced was entirely sensible. The fix was `WalkForwardRegimeClassifier.forecast_variance`,
which records the per-state variances *in force at each row* during the walk-forward loop. This is
the third real bug the truncation detector has caught, and the reason convention 3 — every new
strategy gets a one-line causality test — is not negotiable.

**Bootstrapped markets carried synthetic ticker names.** `BootstrapGenerator` inherited `SYN0, SYN1`
from the base generator, which is correct for purely synthetic markets and part of keeping a model
unable to tell which rung it is on. But a bootstrap of SPY and IEF *is* those assets reshuffled, so
a multi-asset strategy asked to trade `SPY` could not run on it — which meant it silently could not
be given a noise floor at all.

It surfaced as a crash, which is the good case. Had the strategies addressed columns by position
rather than by name, the floors would have been computed against whichever assets happened to line
up and the numbers would have looked entirely reasonable. `Generator.default_tickers` now lets a
generator built from a real market carry that market's names.

A third, smaller one in the same family: four `RegimeRotation` variants shared one class-level
`name`, so the results dictionary keyed by strategy name collapsed them into a single row and the
last one silently won. No crash — just a comparison table that looked fine and reported one strategy
three times under three different sets of numbers. Names are now derived from configuration.

---

## 2026-09-04 — The corrected sandbox predicts reality; the original one predicted the opposite

**Decision:** `RegimeSwitchingGenerator.equity_like()` becomes the default preset for any experiment
whose purpose is to rank strategies. The original parameters stay available and are documented as
what they are. Evidence in `reports/2026-09-04-corrected-sandbox.txt`.

**Why:** After finding that the generator's defaults welded high volatility to negative drift, the
obvious question was whether fixing the parameters actually fixed anything. The check does not need
years of waiting: run the same strategies in both sandboxes and on real SPY, and see which sandbox
*orders* them the way reality does. Rank agreement is the right test — no simulator will reproduce
SPY's exact Sharpe, and it does not need to. What a sandbox is for is deciding which of two
strategies to pursue.

    rank correlation with real SPY
      classic sandbox    (mu = +0.12 / −0.15)    −0.143
      corrected sandbox  (mu = +0.09 / +0.17)    +0.750

The classic sandbox was not merely uninformative. It was **negatively correlated with reality** — a
strategy it ranked highly was, on average, one that did worse on real data. Following its advice was
worse than choosing at random. Specifically, it put `regime_aware` and `regime_gate` at the top and
`buy_and_hold` near the bottom; on real SPY that ordering is inverted.

The correction changed no strategy, no engine code, and no metric. It changed two numbers in a
generator's default arguments.

**The caveat that must travel with this result:** `equity_like()` was calibrated on the same SPY
history it is being checked against, so +0.750 is not out-of-sample. It shows the correction is
self-consistent, not that it will hold on data nobody has seen. The genuinely out-of-sample test is
Phase 5 paper trading, and nothing before it can substitute.

**The next gap, already visible:** the corrected sandbox ranks `absolute_momentum` third, while real
SPY ranks it first. The regime generator has no long-horizon trend structure for a 252-day momentum
rule to work with, so it cannot represent whatever real markets are offering there. That is the next
thing the simulator is missing, and it is worth finding out whether the real result is a trend
premium or thirty-three years of luck — the noise floor for that window is 0.29, and
`absolute_momentum` scores 0.798 against it.

---

## 2026-09-04 — Phase 3: the drawdown breaker is net harmful against gap risk

**Decision:** The circuit breaker stays, and its failure mode is documented rather than tuned away.
Evidence in `reports/2026-09-04-adversarial.txt`.

**What was measured:** every crash scenario run twice — once with the breaker, once without — so its
value is a number rather than a reassurance.

    35% fall over 60 days     +6.1% drawdown saved, costs +0.72%/yr
    35% fall over 20 days     +7.5% saved
    50% fall over 20 days    +12.5% saved
    35% fall over  1 day      −1.5% saved, costs −1.09%/yr
    50% fall over  1 day      −0.9% saved, costs −1.56%/yr

**Why the one-day rows matter:** a drawdown breaker reacts to losses already realised. When the whole
fall lands overnight it cannot act until the damage is done, and what it then does is sell at the
bottom and sit in cash through the rebound — converting an unavoidable loss into a permanent one,
and paying more than a percent a year for it.

This is gap risk: 1987, a currency peg breaking, an overnight halt. **No parameter choice removes
it**, because the mechanism is trailing by construction. The honest response is position sizing that
survives a gap, not a better breaker, and pretending otherwise would be the exact failure the
project exists to avoid — a control that is believed in rather than measured.

Pinned in `tests/test_adversarial.py` so it cannot be rediscovered expensively.

---

## 2026-09-04 — Every multi-asset result so far understates risk by about 17 points of drawdown

**Decision:** A caveat on published numbers, not a bug to fix. `CorrelationBreakdownGenerator` now
exists so future multi-asset work can be measured properly.

**Why:** Every generator in this project until now uses a single fixed correlation matrix, which
quietly promises that assets will keep behaving differently during a crash. They do not — the
nastiest thing real markets do is push correlations toward one exactly when diversification is
needed.

The same equally weighted four-asset portfolio, same weights, same assets:

    fixed correlation 0.2         vol  8.9%, max drawdown −15.4%
    correlation breaks to 0.95    vol 15.4%, max drawdown −32.8%

Seventeen points of drawdown that a fixed-correlation backtest would never show, and a portfolio
volatility nearly twice what was promised. Every multi-asset number in `reports/` was measured on
the first kind of market and carries this understatement.

The real-data results do not, since real prices contain whatever correlation structure they contain.
But the rung-1 multi-asset conclusions do, and they should be re-measured against this generator
before any of them is acted on.

---

## 2026-09-04 — Every piece of machinery added to this project has made real results worse

**Decision:** Recorded as the session's most uncomfortable finding, because it is the one most likely
to be rationalised away later.

Ranked by Sharpe on 32.9 years of real SPY, with each strategy's own bootstrap noise floor:

    absolute_momentum          +0.798  (floor +0.344)   two parameters
    volatility_target          +0.731  (floor +0.384)   one parameter
    buy_and_hold               +0.654  (floor +0.414)   no parameters
    regime_volatility_target   +0.583  (floor +0.372)   an HMM
    regime_aware               +0.504  (floor +0.387)   an HMM
    regime_gate                +0.453  (floor +0.392)   an HMM
    ai_walkforward_classifier  +0.143  (floor +0.039)   LightGBM, 16 features
    short_momentum             +0.008  (floor +0.214)   below its own floor
    ai_walkforward             +0.055  (floor +0.079)   below its own floor

The ordering is almost perfectly inverse to complexity. The multi-asset ladder says the same thing
independently: rotating into cash (Sharpe 0.761) beats rotating into treasuries (0.724), which beats
adding momentum-based asset selection (0.394). And `regime_volatility_target` loses to plain
`volatility_target` by 0.15, meaning a fitted two-state HMM forecasts volatility *worse* than a
21-day rolling standard deviation.

This is not an argument that models never work. It is what the control arm was put there to detect,
and it detected it every time it was asked. `plan.md` §4 says an AI that matches a two-parameter
rule "has learned nothing and is strictly worse, because it has far more ways to fail silently" —
none of these matched, they lost.

**What follows:** the two candidates worth carrying to paper trading are `absolute_momentum` and
`volatility_target`. Both are simple enough to verify by hand. Neither beat buy-and-hold *and* the
noise floor by a margin that survives the caveats — 32.9 years of SPY is one path, and it is the
path everyone already knows went up.

---

## 2026-09-04 — The demeaned noise floor is too generous for judging a timing strategy

**Decision:** Any question of the form "did the timing add value?" is judged against a bootstrap that
**preserves the market's drift** and destroys only the ordering. The demeaned floor stays for the
Null Test, where it remains correct. Evidence in `reports/2026-09-04-is-it-timing.txt`.

**Why:** Every null market in this project is demeaned, for a good reason — with drift left in, any
strategy holding the asset makes money from exposure rather than skill, and the Null Test would
measure the wrong thing.

But applied to a real-data result it produces a bar that means less than it appears to. Buy and hold
scored Sharpe 0.654 on real SPY against a demeaned floor of 0.414, and so "beat its noise floor"
while doing nothing whatsoever except being invested in a market that went up. If buy-and-hold
clears a bar, clearing it is not evidence of skill.

Resampling with the drift left in fixes it. The ordering is still destroyed, so there is nothing to
time — but the market still rises, so a strategy is still paid for exposure. A timing rule's score on
that market is exactly what its average exposure earns, and nothing more is available. The gap
between real and shuffled is then the part attributable to using the order of returns, which is the
only thing the shuffle removed and the only thing timing could be.

**What it did to the results**, 300 shuffled markets over the same 32.9 years:

    strategy              real   shuffled   timing edge   percentile
    buy_and_hold        +0.637     +0.636        +0.001         52%   <- calibration check
    absolute_momentum   +0.769     +0.526        +0.243         87%
    volatility_target   +0.742     +0.599        +0.143         79%
    regime_aware        +0.453     +0.607        -0.154         18%
    short_momentum      +0.040     +0.299        -0.259         10%

The calibration check passes exactly: buy-and-hold has no timing ability, so the shuffle takes
nothing from it, and it lands at the 52nd percentile.

**The conclusion, stated plainly: nothing in this project reaches significance.** `absolute_momentum`
is the best candidate at p = 0.13 — suggestive, and short of the conventional bar. Against the
demeaned floor it had looked decisive (0.798 against 0.344); the entire difference was credit for
being invested in a rising market, which is available for free by holding the index.

And two strategies did *worse* than shuffling their own market. `regime_aware`'s apparent
floor-clearing was pure exposure; its timing actively cost 0.154 of Sharpe.

**What this does not reduce:** the drawdown result. `absolute_momentum`'s worst drawdown was −27.8%
against buy-and-hold's −51.0%. Roughly halving the worst loss follows from always stepping aside
after a sustained decline, which is a mechanical property and not a claim about predicting anything.
It holds whether or not the timing edge is real, and for a project whose stated goal is capital
preservation rather than return maximisation, it may be the more relevant number.

**What would settle it:** not more backtesting on SPY — the shuffle has extracted what this path can
say. Either more independent paths (other indices, other countries, pre-1993 data), or paper trading
forward, where the answer is not yet known to anybody.

---

## 2026-09-04 — Replication across 391 market-years: the return edge does not survive, the drawdown does

**Decision:** The project's answer, on the evidence it now has, is that **trend following is a
risk-reduction tool and not a return-generation tool.** Any further work should be framed that way.
Evidence in `reports/2026-09-04-more-paths.txt`.

**Why:** `is_it_timing.py` reached p = 0.13 on SPY and said the next step had to be more independent
paths. Eight national equity indices, 391 market-years, each judged against 200 reshuffles of its own
returns with drift preserved:

    absolute_momentum, timing edge and percentile
      US S&P 500 (1927)         +0.172   94%
      Japan Nikkei 225 (1965)   +0.223   96%
      Canada TSX (1979)         -0.075   32%
      UK FTSE 100 (1984)        +0.040   60%
      Hong Kong Hang Seng       +0.054   64%
      Germany DAX (1987)        +0.036   59%
      France CAC 40 (1990)      +0.027   56%
      Australia ASX 200 (1992)  -0.129   22%
      positive in 6 of 8       sign test p = 0.145

The calibration check passes: `buy_and_hold`'s median percentile is 46%, near the 50% it must be,
so the shuffle is doing what it claims in every market.

**The return edge does not replicate to significance.** Six of eight is p = 0.145, and the median
edge is +0.038 Sharpe — economically negligible even if real. The two strong results are the US and
Japan, which are the two longest series and the two containing the largest sustained crashes; that
pattern is itself a clue about what the rule is doing.

**The drawdown result replicates in 8 of 8 markets**, by a median of 17.6 points, sign test
p = 0.0039. It is the only result in this project that reaches significance.

**Why the two differ, which is the whole point:** the drawdown benefit does not require the rule to
predict anything. Stepping aside after a sustained decline mechanically truncates a long fall,
whether or not there is any forecastable structure. It is a property of the rule's *shape*, and that
is exactly why it survives when the return edge does not.

It is not free: the same rule costs a median of 1.96% a year across these markets and sits in cash
through part of every recovery. Whether roughly halving the worst drawdown is worth roughly two
points of annual return is a question about the person holding it, not about the data — and stating
it that way is the honest form of the question `plan.md` set out to answer.

**Caveats that stay attached to this result.** The eight markets are not independent — 2008 happened
everywhere — so the sign test is the number to trust and Fisher's p (0.203) is optimistic. They are
also the markets that *survived*: every exchange here stayed open with continuous records, while
Russia 1917 and China 1949 are missing, and those are precisely the cases where buy-and-hold was
catastrophic and trend following would have looked heroic. That bias runs against trend following
here, which is worth knowing and is not a licence to add anything back.

Most are price-return indices, so absolute Sharpes are understated by the missing dividends. The
real-versus-shuffled comparison is unaffected, since both arms share the same drift.

---

## 2026-09-04 — Status: no strategy in this project is ready for real money

**Decision:** Recorded plainly so that it cannot be softened by later summarising.

`plan.md` §2 lists five success criteria. Against 33 years of SPY, the best candidate
(`absolute_momentum`) meets three of them: CAGR 9.84% against a 6% target, Sharpe 0.798 against 0.7,
and turnover of 1 round trip a year against a limit of 50. It fails max drawdown, at −27.8% against
a −15% target. On the multi-asset universe, `rotate_SPY_to_cash` meets the drawdown target at −12.9%
but does not beat buy-and-hold risk-adjusted.

More importantly, the criterion `plan.md` added — "**and above the null floor**" — is the one that
decides, and against the correct floor no strategy clears it. The demeaned floors every earlier
report quoted were too generous, because they credited a strategy for being invested in a market
that went up.

So the position is: nothing here is ready to fund, and the plan anticipated this and permits it.
What the project *has* produced is a measured, replicated, significant finding about risk reduction,
and a set of instruments — the Null Test, the noise floor with its window attached, the truncation
detector, the drift-preserving shuffle, the oracle ladder — that are reusable and that caught every
false positive this project generated. That is a better outcome than a strategy that looked good
because nobody had built the instruments to check it.

---

## 2026-09-04 — PRE-REGISTRATION: what will be tried next, and what is predicted

**Decision:** Predictions written down *before* the experiments are run, so they can be checked
rather than rationalised. `plan.md` §10 does this for the project as a whole; this does it for one
round of work, because the temptation this time is specific and named.

**The trap being avoided.** The obvious next move is to tune `absolute_momentum` until its timing
p-value crosses 0.05. There is now a precise number to tune against (p = 0.145), which makes that
easier to do accidentally than usual. `HANDOFF.md` already lists it as the one dishonest
continuation. Nothing below adjusts a parameter of an existing strategy against real-data results.

**The principle guiding what *is* tried.** Return forecasting failed, and the Recovery Test said why:
real return signals are three times weaker than this method can detect. But the project's own
generators contain a second, separate kind of predictability that has never been pursued —
`HestonGenerator` produces **forecastable risk with unforecastable direction**, and `GroundTruth`
has carried `has_predictable_volatility` as a distinct field since day one precisely because
conflating the two is how people overstate what a model has found.

Volatility is worth attacking for a reason that is about measurement rather than about markets:

> A Sharpe ratio estimated over 33 years has a standard error of about 0.18. A volatility forecast
> can be scored on all 8,000 daily observations with a proper scoring rule. **The second question
> the data can actually answer; the first it cannot.** Every negative result in this project so far
> came from asking a question with too little power to answer it.

### What will be tried

1. **Better volatility forecasters.** EWMA (RiskMetrics), HAR-RV (Corsi), GARCH(1,1), against the
   21-day rolling standard deviation currently used, scored by QLIKE and MSE against realised
   volatility — not by Sharpe.
2. **A multi-horizon trend ensemble.** Averaging signals over 1/3/6/12-month lookbacks. This is a
   *robustness* change, not an optimisation: it reduces sensitivity to a lookback nobody can
   justify choosing, and it should be judged on dispersion across markets rather than on mean return.
3. **Trend and volatility targeting combined**, since they answer orthogonal questions — whether to
   be exposed, and how much. Tested on the corrected sandbox first; real data is confirmation only,
   and gets one look.

### What is predicted

- **HAR-RV will beat the 21-day rolling standard deviation on QLIKE**, and by a margin that is
  statistically clear, because the sample is thousands of observations rather than one Sharpe.
- **EWMA will beat rolling standard deviation modestly**, and will land close to HAR.
- **The improvement will not translate into much Sharpe.** Volatility targeting's benefit comes from
  the broad level of risk, not from fine accuracy, so a materially better forecast should be worth
  well under 0.1 of Sharpe. If it appears to be worth much more, suspect a bug before celebrating.
- **The trend ensemble will not improve mean performance** and will reduce the spread across the
  eight markets. If mean performance improves noticeably, that is a warning sign that a horizon was
  chosen with hindsight rather than a discovery.
- **The combination will improve drawdown more than return**, consistent with everything measured so
  far.

Each of these is falsifiable and cheap. Getting one wrong is more informative than getting all five
right, and they are recorded here so that cannot be quietly forgotten.

---

## 2026-09-04 — RESULTS of the pre-registered round: 5 of 5 predictions held

**Decision:** Two changes adopted, one rejected, all from measurement. Evidence in
`reports/2026-09-04-volatility-forecasting.txt` and `reports/2026-09-04-more-paths.txt`.

| # | Prediction | Outcome |
|---|---|---|
| 1 | HAR-RV beats the 21-day rolling window on QLIKE, clearly | **Held.** DM t = −3.7 on SPY |
| 2 | EWMA beats rolling modestly, lands close to HAR | **Held.** −8.2440 vs −8.2483 vs −8.1899 |
| 3 | A better forecast is worth well under 0.1 Sharpe | **Held.** Best was +0.012 |
| 4 | The trend ensemble won't improve the mean, will cut the spread | **Held.** +0.365→+0.392, spread 0.112→0.105 |
| 5 | The combination improves drawdown more than return | **Held.** −0.003 Sharpe, +36.9 points of drawdown |

Five of five is a weaker outcome than it sounds. The pre-registration said so in advance: *"Getting
one wrong is more informative than getting all five right."* Predictions that all hold mostly confirm
the reasoning was already sound; they teach less than a surprise would have.

**The surprise was not predicted.** GARCH(1,1) forecasts volatility *significantly better* than the
incumbent — Diebold-Mariano p below 0.05 in 8 of 8 markets — and produces a **worse strategy**, by
0.032 of Sharpe. A sharper forecast tracks volatility more tightly, which means trading more (3.1
round trips a year against 1.7), and the extra accuracy does not pay for the extra cost.

That is worth stating plainly, because it is the cleanest example this project has produced of the
distinction it keeps having to make: **a better model is not automatically better money.** No
prediction covered it, and it would have been easy to report "GARCH beats the incumbent at p = 1e-11"
and stop there — which would have been true and misleading.

### Adopted

**`VolatilityTarget` now defaults to EWMA (lambda = 0.94) rather than a 21-day rolling window.**
Better on every axis measured: forecast accuracy (significant, 8 of 8 markets), turnover (1.7 against
2.2), drawdown (−45.8% against −49.2%), and Sharpe (+0.012). The decay parameter is the published
RiskMetrics value from 1994, not fitted here, which is what keeps it out of sample with respect to
this data.

**`EnsembleMomentum` and `TrendScaledVolatility` join the strategy set.** The ensemble removes a
parameter rather than tuning one — a 252-day lookback was inherited, not measured, and averaging over
21/63/126/252 days makes the result insensitive to a choice nobody could justify. The composite
multiplies the two, which is the simplest composition available and has no free parameter of its own.

    across 8 markets       mean Sharpe    spread    mean maxDD
    buy_and_hold                +0.415     0.064        −66.0%
    absolute_momentum           +0.365     0.112        −44.5%
    ensemble_momentum           +0.392     0.105        −39.2%
    volatility_target           +0.456     0.108        −45.8%
    trend_scaled_volatility     +0.411     0.124        −29.1%

**`trend_scaled_volatility` is the best result the project has**: roughly the same risk-adjusted
return as holding the market for **less than half the worst loss**, shallower in 8 of 8 markets by a
median of 37 points (sign test p = 0.0039).

Its return edge is still not significant — 5 of 8 markets positive, p = 0.363 — and that is the
consistent finding. The drawdown benefit does not need the timing edge to be real: stepping aside
after a sustained decline mechanically truncates a long fall, and sizing down when volatility rises
mechanically shrinks the position going into one. Neither requires anticipating anything, which is
exactly why they replicate when the return edges do not.

It is not free. It holds less than the market most of the time and will trail badly through a long
calm bull run.

### Rejected

**GARCH and HAR as the sizing forecast.** Both forecast better and trade worse. They stay in the
codebase as measured comparisons, which is what the volatility experiment exists to provide.

**Any tuning of `absolute_momentum`.** `HANDOFF.md` named this as the one dishonest continuation and
nothing in this round touched a parameter of an existing strategy against real-data results. The
ensemble *removes* a parameter; the composite adds none; EWMA's decay is a published constant.

---

## 2026-09-04 — Two measurement bugs in the volatility work, both invisible

Logged because both produced plausible wrong numbers rather than a crash, and both were in the
*measuring* apparatus rather than in a model.

**HAR was biased by a constant factor, twice, in opposite directions.** The model predicts
`log|return|`; exponentiating gives the geometric mean, which for a right-skewed quantity sits below
the arithmetic mean that is wanted — Jensen's inequality. Uncorrected it under-forecast by 9.7%. The
textbook fix, smearing by `exp(residual variance / 2)`, assumes the residuals are normal in logs;
`log|z|` is sharply left-skewed with a variance near 1.23, so smearing overshot by 12%. The correct
constant is exact and specific: `-E[log|Z|] = (gamma + log 2)/2 = 0.63518` for a standard normal,
with an empirical calibration on the training window on top of it for fat tails.

Nothing about any version looked wrong. The forecasts tracked volatility perfectly well and only
their *level* was off, so QLIKE charged the mistake to the model as poor forecasting rather than to
the arithmetic — and it would have been recorded as "HAR does not work here".

**Models were being ranked on different samples.** A rolling window needs 21 days of warmup and a
walk-forward regression needs two years, so scoring each over its own full range compares them on
different periods. On SPY that difference was 1993–95, which contains a bear market, and it *reversed
the ranking* between the per-model table and the pairwise tests. `score_all` now intersects the valid
ranges. A ranking is only a ranking if every model faced the same days.

**And one bad check of my own.** The first version of the sandbox validation compared point-estimate
orderings with no error bars, and reported a disagreement between the exact and proxy rankings that
was pure noise — three models tied to four decimal places will "rank" in arbitrary order. It now
compares pairwise with a paired t-test and flags only *reversals*, where one measure says A is
significantly better and the other says B is. One measure finding a difference the other cannot is a
difference in power, not a contradiction.

---

## 2026-09-04 — A silent multi-asset bug: volatility sizing was single-asset all along

**Decision:** `VolatilityTarget` now forecasts and sizes every asset separately, and
`TrendScaledVolatility` checks that its two components produced matching columns instead of relying
on array shapes lining up.

**What was wrong.** `VolatilityTarget.compute_weights` returned a single column — always
`tickers[0]`. On a six-asset universe anything multiplying it against a multi-asset weight matrix got
**numpy broadcasting rather than an error**, so a portfolio of equities, small caps, international,
two treasury funds and gold was sized entirely by the S&P's volatility. Gold does not have equity's
volatility; sizing it as though it did is not a conservative approximation, it is a different
strategy.

Nothing failed. No shape mismatch was raised, the weights summed to something sensible, no leverage
appeared, and the reported Sharpe and drawdown were entirely plausible. It was caught only because a
CAGR of 3.91% alongside a −8.6% drawdown looked oddly extreme and was worth a second look — not by
any test.

**And a second bug underneath it.** Fixing the first exposed the fact that both components are
complete allocators, and both divide their conviction across the assets available. Multiplying their
raw weights divides twice, and that version returned 0.43% a year against buy-and-hold's 8.09% — low
enough to notice, but not obviously a bug rather than a very conservative strategy. The composition
now multiplies the components' *aggregate* exposures and takes the split across assets from the
elementwise product. For a single asset it is exactly `trend x volatility` and nothing changes.

**What was affected.** Only multi-asset results, and only in the rung-2 universe section. Every
single-asset number — all of `more_paths`, the eight-country replication, the whole SPY table, and
the Null Test — was computed on one asset at a time and is unchanged. Verified by re-running all
three: the null gate still passes 39 of 39 cells, and the replication drawdown result is identical
at 8 of 8, p = 0.0039.

**Corrected multi-asset result**, 2004-2025, six assets:

    strategy                      Sharpe    floor     CAGR    maxDD
    buy_and_hold                  +0.805   +0.469   +7.56%   -24.5%
    trend_scaled_volatility       +0.843   +0.426   +3.89%    -8.5%

The composite now has the highest Sharpe in the table and clears its floor, with a max drawdown a
third of buy-and-hold's — while giving up half the return. Same shape as everywhere else in this
project: the risk result is strong and the return result is not.

**The lesson worth keeping.** Both of these were in the *composition* of correct components rather
than in any component itself, and neither could have been caught by the causality detector, the Null
Test, or any existing check — all of them ran happily on the broken version and reported passes.
Silent broadcasting between array shapes is a whole class of bug this project had no defence against
until now. `tests/test_composite.py::TestMultiAssetSizing` is that defence.

---

## 2026-09-04 — The control nobody runs: is timing better than simply holding less?

**Decision:** Recorded because it is the first question a sceptic should ask and it had never been
asked here.

A strategy that holds less of a risky asset will naturally show lower return and lower drawdown. So
before crediting the *timing* for anything, it has to beat a constant allocation matched to the same
risk. On 33 years of SPY:

    strategy                      CAGR      vol   maxDD
    buy_and_hold                +9.30%    15.4%  -51.0%
    always 57% stocks           +5.95%     9.9%  -32.3%
    trend_scaled_volatility     +6.48%     9.0%  -18.0%

It wins on all three: more return, less volatility, and a drawdown 14 points shallower than the
constant allocation at matched risk. So the timing is doing something that holding less does not,
which is worth knowing — the alternative explanation was that this whole project had built an
elaborate way to own fewer shares.

That is one path, and the timing edge across eight countries remains p = 0.145. The drawdown gap is
the part that replicates.

---

## 2026-09-04 — Why this is not a free lunch, stated so it is not mistaken for one

**Decision:** The strategy is a **trade**, not an edge, and every summary should say so.

Measured on SPY: it trailed buy-and-hold **continuously for 16.7 years**, from April 2009 to the
present, falling from 1.17x the index to 0.42x. Through the 2009-2021 bull market it returned 7.6% a
year against the index's 16.0%.

That is the honest answer to "if this works, why doesn't everyone do it". Trend following and
volatility targeting are neither obscure nor secret — they are standard institutional practice and
appear throughout the academic literature. Nothing here was invented; it was rebuilt and measured.
What stops people using them is that almost nobody holds a strategy through sixteen consecutive years
of watching a friend who did nothing get richer.

And structurally there is nothing to arbitrage away. This is not an inefficiency that would close if
more people knew about it. It is an exchange of return for smaller drawdowns, available to anyone,
which most people decline because they want the bigger number. Any future summary that presents it as
a way to make more money is wrong on the project's own evidence.
