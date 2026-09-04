# market-sentinel — Project Plan

An AI trading system developed inside a **synthetic market laboratory**, where the ground truth is
known, before it is ever pointed at a real market.

Status: **planning only.** No code written. No money involved.

---

## 0. The core idea

Most retail quant projects fail the same way. Someone trains a model on real price history, gets a
beautiful backtest, trades it, and loses money. They never find out why, because they only ever had
**one copy of history** and no way to tell the difference between a model that learned something
real and a model that memorized noise. Both look identical on the only data that exists.

This project fixes that by **building the markets first.**

If you generate a market yourself, you know exactly what is in it. You know whether it contains an
exploitable pattern, what kind, and how strong. That turns the central unanswerable question —
*"did my AI actually learn anything?"* — into a question with a checkable answer.

Two experiments fall out of this immediately, and they are the backbone of the entire project:

### The Null Test — can it correctly find nothing?

Generate a market that is pure randomness. No pattern, no signal, nothing to discover — by
construction, mathematically guaranteed. Then run the complete AI pipeline on it.

**The AI must fail.** It must report roughly zero profit.

If it reports profit, you have not discovered a strategy. You have discovered a **bug in your own
code** — almost always the model accidentally seeing future data. This test catches, automatically
and immediately, the single most expensive class of error in the field. And it catches it on day
one, on fake money, instead of eighteen months later on real money.

Run it a thousand times and you get something even more valuable: a **noise floor.** You learn what
returns your method produces on markets containing *nothing*. If random markets routinely hand your
method a Sharpe ratio of 0.8 by pure chance, then a Sharpe of 0.6 on real data is not evidence of
skill — it is below the noise floor of your own instrument. Almost no retail project ever computes
this number, and it is the difference between a result and a coincidence.

### The Recovery Test — how weak a signal can it find?

Now generate a market with a pattern deliberately planted in it, at a strength you control. Turn
that strength down gradually and find the point where the AI stops finding it.

That number is your AI's **sensitivity**, and it is brutally informative. Real market signals are
extremely weak. If your model needs a strong signal to detect anything, and real markets only offer
faint ones, then your model cannot work on real markets — and you know this **before** risking
anything, from a measurement rather than a guess.

Together these two tests turn the AI from something you hope works into an instrument with known
error bars. That is the whole project.

---

## 1. The Reality Ladder

Four rungs. Each adds exactly one element of reality. You do not climb until the current rung is
solid, and the same code runs at every level.

| Rung | Market | Money | What it proves |
|---|---|---|---|
| **1. Sandbox** | Synthetic | Fake | The AI works when a correct answer exists — and correctly finds nothing when it does not |
| **2. Backtest** | Real history | Fake | It survives real market messiness: fat tails, gaps, costs, crashes |
| **3. Paper** | Real, live | Fake | The plumbing works in real time — data feeds, orders, timing |
| **4. Live** | Real, live | Real | It works when it actually matters |

The value of this structure is that **each rung isolates one kind of failure.** A break at rung 2
that was fine at rung 1 means real markets contain something your simulator did not. A break at
rung 3 that was fine at rung 2 is an engineering bug, not a strategy problem. Without this
separation, a failure at the end is unattributable and you are just guessing.

Most of the calendar time in this project is spent on rung 1, which costs nothing and can be run a
million times overnight.

---

## 2. Goal

Success, measured out-of-sample:

| Metric | Target | Why |
|---|---|---|
| CAGR | > 6% | Must beat cash (~4%) enough to justify the risk and effort |
| Max drawdown | < 15% | The peak-to-trough loss you must be able to sit through |
| Sharpe ratio | > 0.7, **and above the null floor** | Return per unit of risk — the second clause is the one that matters |
| Trades per year | < 50 | Low turnover: low cost, low tax, less room to overfit |
| Beats buy-and-hold SPY risk-adjusted | yes | If not, buy the index fund and stop |

**The kill switch:** if the AI cannot clear the noise floor established by the Null Test, the
project has produced a real and valuable finding — *this approach does not work* — and the correct
action is to stop and buy an index fund. The plan is designed to be able to reach that conclusion.
A plan that cannot fail is not a plan.

---

## 3. The synthetic market generator

The centerpiece of the codebase, and the first thing built. Each generator produces price series
with **known, controllable statistical properties.**

| Model | What it produces | What it tests |
|---|---|---|
| **Geometric Brownian motion** | Pure random walk. No signal, guaranteed | The Null Test. The most important generator |
| **AR(1) momentum** | Returns mildly predict future returns, strength tunable | The Recovery Test. Can the AI find a planted trend? |
| **Ornstein-Uhlenbeck** | Mean reversion — prices pull back to a level | The opposite signal. Does the AI adapt, or only ever see trends? |
| **Markov regime-switching** | Flips between calm-bull and volatile-bear states at known times | The big one. Regime detection is the AI's actual job |
| **Merton jump-diffusion** | Sudden crashes out of nowhere | Risk controls. Do circuit breakers fire correctly? |
| **Heston stochastic volatility** | Volatility clusters — calm periods and stormy ones | Realism. Real markets do this; random walks do not |
| **Block bootstrap of real returns** | Reshuffled chunks of actual history | The bridge to rung 2. Real fat tails, destroyed ordering |

Multi-asset from the start, since the strategy allocates *across* assets — which means simulating a
correlation structure, including the nastiest real-world behavior there is: **correlations spiking
toward 1 during crashes.** Diversification tends to evaporate exactly when it is needed, and a
simulator that misses this will make any strategy look far safer than it is.

The simulator also models the boring things that quietly destroy returns: bid-ask spread, slippage,
and the fact that you trade at tomorrow's open, not at the close price that triggered your decision.

**Deliberately not using AI to generate the markets** (GANs, diffusion models), despite it being
possible and fashionable. A learned generator produces data whose true properties are *unknown* —
which destroys the entire reason for having a simulator. Parametric models are the point precisely
because their ground truth is exact. Revisit only as a late stretch goal.

---

## 4. The AI

Present from day one, because the sandbox makes it safe to be.

### Architecture

Three components, built in order:

**1. Regime classifier.** Given recent market behavior, output a probability distribution over
market states: calm-trending, volatile, crashing, mean-reverting. This is the core, and it is a far
more tractable question than "what is the price tomorrow." The sandbox is ideal for it because
regime-switching generators come with the true regime labels attached — so you can measure
classification accuracy directly, which is impossible on real data where nobody knows the true
label.

*Model: gradient-boosted trees (LightGBM) as the workhorse — strong on tabular data, fast, hard to
catastrophically overfit, and it reports feature importance so you can see what it is using.
Cross-checked against a Hidden Markov Model, which is the classical interpretable approach.*

**2. Position sizer.** Convert regime probabilities plus uncertainty into target portfolio weights.
Deliberately conservative: high uncertainty produces small positions. **A model that says "I don't
know" and sizes down is worth more than one that is confidently wrong**, and the sandbox lets us
verify it actually behaves that way by feeding it markets it has never seen.

**3. Anomaly detector.** Flag when current conditions do not resemble anything in training. Its
output is not a trade — it is a *reduce risk* signal. This is the system's defense against the
thing that has no defense: a genuinely unprecedented market.

### Rules on the AI

- **Confidence must be calibrated, not just accurate.** When it says 70%, it must be right about
  70% of the time. Verified on synthetic data where truth is known. An overconfident model is worse
  than no model, because you will size positions off its certainty.
- **A dumb rules baseline runs alongside every single experiment**, permanently — not as a warm-up
  phase, but as a control arm. Every result is reported as "AI vs. simple rule." If a two-parameter
  moving-average rule matches the neural network, the network has learned nothing and is added
  fragility for free. This is how the earlier plan's caution survives in a form that lets the AI
  start immediately.
- **Interpretability is a requirement.** Every trade must come with a reason you can read. You
  cannot maintain, debug, or hold your nerve during a drawdown on a system you cannot interrogate.
- **No deep learning until gradient boosting is exhausted.** Not conservatism — on this data volume,
  trees genuinely outperform neural nets, and they train in seconds instead of hours. Sequence
  models are a Phase 5 experiment, run against the same tests.

### Reinforcement learning

Deferred, deliberately, and worth explaining since it is the obvious thing to reach for. RL needs
enormous amounts of data and is notoriously unstable — but the simulator *can* generate unlimited
data, which is exactly the condition RL needs. So it becomes genuinely viable here, later, as a
Phase 5 experiment once the supervised pipeline provides a baseline to beat. The danger is that an
RL agent will learn to exploit the *simulator's* quirks rather than market structure, which is why
it comes after the simulator has been validated against reality at rung 2.

---

## 5. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    MARKET SOURCES                        │
│                                                          │
│   ┌────────────────┐   ┌──────────┐   ┌──────────────┐   │
│   │   SYNTHETIC    │   │   REAL   │   │  REAL, LIVE  │   │
│   │   GENERATOR    │   │ HISTORY  │   │    FEED      │   │
│   │ (known truth)  │   │ (yahoo)  │   │  (alpaca)    │   │
│   └───────┬────────┘   └────┬─────┘   └──────┬───────┘   │
└───────────┼─────────────────┼────────────────┼───────────┘
            └─────────────────┼────────────────┘
                              │  ← identical interface: all
                              │    three look the same to
                   ┌──────────▼──────────┐   everything below
                   │    FEATURE LAYER    │
                   │ returns, vol, trend │
                   └──────────┬──────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                                     │
  ┌────────▼─────────┐                 ┌─────────▼────────┐
  │    AI ENGINE     │                 │  RULES BASELINE  │
  │ regime · sizing  │   ← compared →  │  (control arm,   │
  │    · anomaly     │      always     │   always on)     │
  └────────┬─────────┘                 └─────────┬────────┘
           └──────────────────┬──────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │     RISK LAYER      │  veto power.
                   │ caps · breakers ·   │  can shrink or
                   │   sanity checks     │  reject, never
                   └──────────┬──────────┘  enlarge
                              │
                   ┌──────────▼──────────┐
                   │  EXECUTION / SIM    │
                   │ costs · slippage ·  │
                   │   realistic fills   │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  EVALUATION HARNESS │
                   │ null test · recovery│
                   │  test · tearsheets  │
                   └─────────────────────┘
```

**The critical design constraint:** synthetic, historical, and live markets present the **same
interface**. The AI cannot tell which one it is trading. This is what makes the Reality Ladder
meaningful — climbing a rung changes the data source and nothing else, so any change in behavior is
attributable to reality itself rather than to a code path that differs between test and production.
Most retail systems have subtly different backtest and live code, and that gap is where undetected
bugs live.

**The risk layer holds veto power** over both the AI and the baseline. It can shrink or reject a
position, never enlarge one. A bug in the model therefore cannot produce an oversized trade.

---

## 6. Repository layout

```
market-sentinel/
├── plan.md · README.md · DECISIONS.md
├── sentinel/
│   ├── sandbox/          ← THE SIMULATOR. built first
│   │   ├── generators/     gbm, ar1, ou, regime, jump, heston
│   │   ├── correlations/   multi-asset structure, crash coupling
│   │   └── microstructure/ spread, slippage, fill timing
│   ├── data/             ← real market adapters (same interface)
│   ├── features/         ← returns, volatility, trend, regime inputs
│   ├── ai/
│   │   ├── regime/         classifier + calibration
│   │   ├── sizing/         probabilities → weights
│   │   └── anomaly/        out-of-distribution detection
│   ├── baseline/         ← the dumb rules control arm
│   ├── risk/             ← caps, circuit breakers, sanity checks
│   ├── engine/           ← unified backtest/live loop
│   ├── execution/        ← broker adapters (paper first)
│   └── journal/          ← decision logging, reporting
├── experiments/
│   ├── null_test.py      ← THE MOST IMPORTANT FILE IN THE REPO
│   ├── recovery_test.py  ← sensitivity curve
│   └── adversarial.py    ← crash and regime-shift stress tests
├── tests/
└── reports/              ← dated tearsheets, committed to git
```

`experiments/null_test.py` runs in CI on every commit. If the AI ever starts making money on pure
randomness, the build fails. This is a permanent, automated guard against the field's most
expensive bug, and having it run continuously is worth more than any amount of care.

---

## 7. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.12** | Every relevant library assumes it; already installed |
| Numerics | numpy + pandas | The simulator is mostly numpy and will be fast |
| ML | **LightGBM**, scikit-learn | Best-in-class on tabular data; seconds to train |
| Regime cross-check | `hmmlearn` | Classical, interpretable comparison |
| Storage | **parquet** | No database to run |
| Real data | yfinance, FRED | Free, sufficient for daily bars |
| Broker | **Alpaca** | Free paper trading with a real API |
| Testing | pytest + CI | The null test must run automatically |
| Reporting | static HTML | Generated per run, no server |

Not using: deep learning frameworks (nothing needs them yet), a database (overkill), cloud compute
(the simulator runs fine locally), or paid data (until free is proven to be the bottleneck).

**Cost through rung 3: $0.**

---

## 8. Risk controls

Enforced in the risk layer, not overridable by any model:

1. **No leverage, margin, options, shorting, or crypto.** Permanent policy. These cause most
   catastrophic retail losses, and excluding them structurally beats relying on discipline.
2. **Broad ETFs only.** A single company can gap down 40% overnight on news no model could
   anticipate. An index cannot.
3. **Drawdown circuit breaker.** Down 12% from the high-water mark → full cash, halt automation,
   require human review.
4. **Cash is always a valid position.** The system's main defense is the ability to be entirely out
   of the market.
5. **Uncertainty shrinks positions.** Low model confidence directly reduces exposure.
6. **Staleness and sanity guards.** Stale data or an order more than 3x expected size is blocked
   automatically, not flagged.
7. **Anomaly detector reduces risk, never increases it.** Asymmetric by design.

---

## 9. Roadmap

Each phase has an exit criterion. Enthusiasm does not get to outrun evidence.

### Phase 0 — Build the sandbox *(~2 weeks)*
Generators, correlation structure, cost model, unified market interface.

*Exit: generated markets pass statistical checks — a GBM series is verifiably indistinguishable from
a random walk, a regime-switching series verifiably switches at the times you specified.*

### Phase 1 — The AI, and the Null Test *(~3 weeks)*
Build the regime classifier, sizer, and anomaly detector. Then immediately try to make money on
pure noise.

*Exit: **the AI fails on random markets.** Near-zero return across 1,000 null markets, with a
measured noise floor. A pass here is the AI proving it can find nothing when there is nothing.*

### Phase 2 — The Recovery Test *(~2 weeks)*
Plant signals of known strength; sweep strength down; find the detection limit. Same for regime
switches: how sharp must a regime change be before the classifier catches it, and how fast?

*Exit: a sensitivity curve, plus a written judgment on whether real markets plausibly contain
signals above that threshold. **This phase is allowed to kill the project** — and if it does, it
will have done so cheaply, on fake money, with a clear reason.*

### Phase 3 — Adversarial markets *(~2 weeks)*
Crashes, correlation breakdowns, regime shifts the model never trained on, and markets deliberately
built to break it. Verify circuit breakers fire and the anomaly detector actually notices.

*Exit: survives every stress scenario with drawdown inside limits. No silent failures.*

### Phase 4 — Real history *(~3 weeks)*
First contact with reality. Walk-forward: train on 1990-2005, test untouched on 2006-2024. Compare
against the null floor and the rules baseline.

*Exit: an honest out-of-sample tearsheet, plus a diagnosis of every place reality diverged from the
simulator — each divergence is a lesson about what the sandbox is missing.*

### Phase 5 — Paper trading *(6 months, calendar time, no shortcut)*
Live data, fake money. Every decision logged *before* the outcome is known. Monthly written review
comparing live behavior to expectations.

*Exit: 6 months, no execution bugs, results inside the predicted range. Divergence means a bug —
find it.*

### Phase 6 — Real money
An amount you would be genuinely fine losing entirely. Human approves every trade for the first
year. Scale only on evidence.

**Realistic timeline: 8-10 months.** But unlike the previous draft, the first six weeks now produce
real, publishable findings about what the AI can and cannot do — rather than six weeks of
scaffolding before you learn anything.

---

## 10. Things that will go wrong (predicted in advance)

Written down now so they can be checked rather than rationalized later.

1. **The AI will make money on the Null Test the first time you run it.** It will be a lookahead
   bug. It is always a lookahead bug. Budget several days for this, and treat finding it as the
   project's first real success.
2. **The sensitivity threshold will be discouraging.** Expect to find the AI needs a stronger signal
   than real markets plausibly contain. That is the most likely single outcome of Phase 2, and it is
   *information*, obtained for free.
3. **Real markets will break something the simulator never did.** Guaranteed. That gap is the most
   interesting thing the project will produce.
4. **The model will be overconfident** before calibration. Raw classifier probabilities are almost
   never honest.
5. **There will be a stretch where it badly underperforms just holding SPY** — likely 1-2 years.
   Normal for defensive systems in bull markets, and exactly when people abandon working systems.
6. **The urge to add features after a bad month will be intense.** That urge is how strategies die.
   Changes require a written rationale in `DECISIONS.md` and out-of-sample evidence.

---

## 11. Practical notes

- Personal tool. Not financial advice. Managing other people's money requires registration — do not.
- **Brokerage accounts require 18+**; under that, a custodial account with a parent or guardian.
  Affects Phase 6 only — every rung below it is unaffected.
- **Taxes** are a real drag: profits are taxable, and holdings sold inside a year are taxed higher.
  Low turnover helps. The backtest must model this.
- Keep records from day one.

---

## 12. Open questions

1. **Advisory or autonomous?** Recommend-and-you-click, or place its own trades? *(Recommendation:
   advisory well into Phase 6 — safer, simpler, and it keeps you learning what each trade means.)*
2. **Capital and timing** — how much real money, and when? Affects broker choice and tax modeling.
3. **Account type** — taxable or retirement? Changes the tax math substantially.
4. **Emphasis** — is this primarily about learning markets and programming, or about producing a
   working system fastest? Both are valid and they lead to different amounts of build-it-yourself.
5. **Universe** — which ETFs, and do we include a defensive sleeve (gold, TIPS) from the start?

---

## 13. Next step

Phase 0: the geometric Brownian motion generator, and a statistical test proving its output is a
genuine random walk. Roughly fifty lines of code, and everything else in the project is built on
trusting it.

*Last updated: 2026-09-03*
