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
