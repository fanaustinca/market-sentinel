# market-sentinel — Project Plan

A defensive, rules-first market system. Built to **lose slowly and predictably** before it is
ever asked to win.

Status: **planning only.** No code written. No money involved. Nothing is decided yet.

---

## 0. The honest starting point

Read this section twice. Everything else in the plan follows from it.

**There is no such thing as an AI that does not lose money in markets.** If a system can make
money it can lose money — those are the same machine running in two directions. Anyone who tells
you otherwise is selling something.

What we *can* build is a system where:

- losses are **small** (bounded by design, not by hope),
- losses are **expected** (you saw the same size loss 40 times in the backtest before it happened
  with real money),
- and the system **gets out of the way** during the market conditions that cause big losses.

If the requirement is *literally* "never lose a dollar," the answer is not this repo. The answer
is a high-yield savings account or short-term Treasury bills: roughly 4% a year, government
insured, no skill required, no code required. That is a genuinely good option and it is the
benchmark this project has to beat to justify its own existence.

So the bar this project must clear is not "make money." It is:

> **Beat a savings account, after taxes and fees, over a full market cycle, without a drawdown
> that would make you quit.**

That is a much harder bar than it sounds, and most professionals do not clear it.

### The second honest thing

The hard part of this project is **not** predicting the market. The hard part is **not fooling
yourself**. It is trivially easy to produce a backtest showing 40% annual returns. It is trivially
easy because the tools let you cheat by accident — using data you would not have had at the time,
testing 500 ideas and keeping the one that worked, forgetting that trades cost money.

Therefore: **most of the engineering effort in this plan goes into validation infrastructure, not
strategy.** Roughly 70/30. That ratio is the single most important design decision in the document.

---

## 1. Goal, stated in numbers

The system is a success if, measured over a multi-year out-of-sample period:

| Metric | Target | Why |
|---|---|---|
| CAGR | > 6% | Must beat cash (~4%) by enough to be worth the effort and risk |
| Max drawdown | < 15% | The peak-to-trough loss you must be able to sit through |
| Sharpe ratio | > 0.7 | Return per unit of volatility; SPY buy-and-hold is ~0.5 long-run |
| Time in market | < 100% | Must be willing to hold cash; that is where the safety comes from |
| Trades per year | < 30 | Low turnover = low costs, low taxes, low chance of overfitting |
| Beats SPY buy-and-hold on **risk-adjusted** return | yes | If it does not, just buy SPY and stop |

**The kill switch:** if after Phase 3 the system cannot beat "buy SPY and never look at it" on a
risk-adjusted basis, the correct outcome is to **shut the project down and buy the index fund.**
The plan must be able to conclude that. A plan that cannot fail is not a plan.

---

## 2. Guiding principles

1. **Simple beats clever.** Every parameter you add is a new way to fool yourself. A 2-parameter
   strategy that works is worth more than a 50-parameter one that backtests better.
2. **ETFs, not individual stocks.** A single company can go to zero overnight on news no model
   saw coming. A broad index cannot. Individual stock picking is deferred indefinitely.
3. **Cash is a position.** The main source of safety in this design is the ability to be 100% out
   of the market. Most retail losses come from being fully invested during a crash.
4. **Paper trade for a minimum of 6 months before a single real dollar.** Non-negotiable.
5. **Out-of-sample or it did not happen.** Any result measured on data the strategy was tuned on
   is marketing, not evidence.
6. **Every number gets a benchmark next to it.** "12% return" is meaningless. "12% vs SPY's 14%"
   is information.
7. **Log everything, immutably.** Every decision the system makes gets written down *before* the
   outcome is known. This is the only defense against rewriting history in your own memory.
8. **The system proposes, the human disposes** — at least until Phase 5.

---

## 3. What "AI" actually means here

You asked for a stock market AI, so let me be specific about where machine learning helps and
where it is actively dangerous.

**Where ML is dangerous:** predicting tomorrow's price. Financial data has a terrible
signal-to-noise ratio and non-stationary behavior (the rules change over time). Neural nets given
price data will find patterns in what is essentially noise, and will do so with tremendous
confidence. This is the #1 way retail quant projects lose money.

**Where ML genuinely helps:**

- **Regime classification** — is the market currently calm, volatile, trending, or breaking down?
  This is a much easier question than "what happens tomorrow" and it drives the risk-on/risk-off
  decision that provides most of the safety.
- **Reading text at scale** — earnings call transcripts, filings, news. An LLM is legitimately good
  at "summarize the risk factors in this 10-K" or "did guidance improve or worsen versus last
  quarter." This is language work, which LLMs are good at, not prediction, which they are not.
- **Anomaly detection** — flagging that today does not look like the data the strategy was
  validated on, which is a signal to reduce risk rather than to trade.

**The sequencing rule: no ML until a dumb rules-based baseline works.** The baseline is not a
warm-up exercise; it is the control group. If a neural net cannot beat a 2-line moving-average
rule out-of-sample, the neural net has learned nothing and is added risk for free. Phases 1-3 have
zero ML in them, deliberately.

---

## 4. The strategy family (Phase 1 baseline)

Starting point is **dual momentum applied to a small ETF universe**, a well-documented approach
whose main appeal is that it is simple enough to fully understand and has few parameters to
overfit.

The mechanic, in plain English, evaluated once a month:

1. Look at a handful of broad ETFs: US stocks (`SPY`), international stocks (`VEU`), bonds (`AGG`),
   and cash/T-bills (`BIL`).
2. Compare each one's return over the last ~12 months.
3. **Absolute momentum:** if US stocks have underperformed T-bills over that window, the market
   regime is bad — hold bonds or cash instead of stocks.
4. **Relative momentum:** if the regime is fine, hold whichever of US/international did better.
5. Re-check next month. Otherwise do nothing.

Why this specific family for a first build:

- **It has an explicit "get out" rule**, which is what caps drawdown. In 2008 and 2022 this class
  of rule moved to bonds/cash rather than riding the decline down.
- **Two parameters** (lookback window, rebalance frequency). Very little room to overfit.
- **~2-6 trades a year.** Costs and taxes stay negligible, and it does not require you to watch
  screens.
- **You can verify every decision by hand in a spreadsheet.** Critical when you are learning: you
  should never run a system you cannot check manually.

Its published long-run numbers are attractive, but **published backtests by the authors of a
strategy are not evidence.** Phase 2 exists specifically to reproduce or refute them with our own
data and our own cost assumptions. Fully expect the honest numbers to be worse than the brochure.

Candidate variants to test *after* the baseline is validated, not before: volatility targeting
(size positions by recent volatility), a trend filter (only hold what is above its 200-day
average), and a defensive sleeve (gold/TIPS as an alternative safe asset).

---

## 5. Architecture

```
                  ┌─────────────────┐
                  │   DATA LAYER    │  yfinance / Alpaca / FRED
                  │  fetch + cache  │  → parquet on local disk
                  └────────┬────────┘
                           │  point-in-time, survivorship-safe
                  ┌────────▼────────┐
                  │  FEATURE LAYER  │  returns, momentum, volatility,
                  │                 │  drawdown, regime flags
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │ STRATEGY LAYER  │  pure function:
                  │                 │  (features, date) → target weights
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │   RISK LAYER    │  position caps, drawdown circuit
                  │   (veto power)  │  breaker, cash floor, sanity checks
                  └────────┬────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
 ┌────────▼────────┐              ┌─────────▼────────┐
 │   BACKTESTER    │              │ EXECUTION LAYER  │
 │  historical     │              │  paper → live    │
 │  walk-forward   │              │  (Alpaca API)    │
 └────────┬────────┘              └─────────┬────────┘
          │                                 │
          └────────────────┬────────────────┘
                           │
                  ┌────────▼────────┐
                  │  JOURNAL / LOG  │  every decision, before outcome
                  │   + dashboard   │  known. Append-only.
                  └─────────────────┘
```

**The critical design constraint:** the strategy layer is a **pure function** — same inputs always
produce the same outputs, no hidden state, no network calls. This is what makes it possible to run
the *exact same code* in backtest and in live trading. Most retail systems have subtly different
backtest and live code paths, and that gap is where undetected bugs live.

**The risk layer has veto power over the strategy layer.** The strategy proposes target weights;
risk can shrink or reject them but never increase them. Separating these means a bug in strategy
logic cannot produce an oversized position.

---

## 6. Repository layout

```
market-sentinel/
├── plan.md                  ← this document
├── README.md                ← what it is, honest disclaimer, how to run
├── DECISIONS.md             ← running log of design choices + why (dated)
├── data/
│   ├── raw/                 ← cached downloads, never edited (gitignored)
│   └── curated/             ← cleaned parquet, point-in-time correct
├── sentinel/
│   ├── data/                ← fetchers, cache, corporate-action handling
│   ├── features/            ← momentum, volatility, regime indicators
│   ├── strategies/          ← one file per strategy, all pure functions
│   ├── risk/                ← position sizing, circuit breakers, limits
│   ├── backtest/            ← engine, cost model, walk-forward harness
│   ├── execution/           ← broker adapters (paper first)
│   └── journal/             ← decision logging, reporting
├── notebooks/               ← exploration, clearly marked non-production
├── tests/
│   ├── test_no_lookahead.py ← the most important test file in the repo
│   └── ...
├── reports/                 ← generated tearsheets, dated, committed
└── config/
    └── strategy.yaml        ← all parameters live here, never hardcoded
```

`tests/test_no_lookahead.py` deserves the emphasis. It automatically feeds the strategy truncated
history and asserts that decisions made on day N never change when day N+1 data arrives. Lookahead
bias is the most common and most expensive bug in this domain, and it is silent — it just makes
your backtest look great.

---

## 7. Tech stack

| Layer | Choice | Reasoning |
|---|---|---|
| Language | **Python 3.12** | Every finance library assumes it; you already have it installed |
| Data wrangling | pandas + numpy | The default; enormous amount of help available online |
| Storage | **parquet files** | No database to run. Fast, compressed, versionable |
| Market data | **yfinance** (free) → Alpaca (free) | Start free. Yahoo is fine for daily ETF bars |
| Macro data | **FRED** (free API key) | Interest rates, unemployment — for regime work |
| Backtest engine | **write our own, ~300 lines** | Off-the-shelf engines hide their assumptions; for a small monthly-rebalance universe, a custom loop is simple and you will understand every line. Cross-check against `vectorbt` |
| Broker | **Alpaca** | Free paper trading with a real API, commission-free live, good docs |
| Dashboard | Static HTML report | Generated after each run. No server to babysit |
| Scheduling | cron on this machine | Monthly rebalance. Does not need cloud infra |
| Testing | pytest | Non-negotiable given the correctness stakes |

Deliberately **not** using: TensorFlow/PyTorch (nothing in Phases 1-3 needs them), a database
(overkill), a cloud VM (monthly rebalancing does not need uptime), or a paid data vendor (until
free data is proven to be the bottleneck).

---

## 8. Risk controls (the part that actually protects you)

These are enforced in code, in the risk layer, and are not overridable by any strategy.

1. **Max position size:** no single ticker above 50% of the portfolio (Phase 1 holds one at a time,
   so this is a backstop against a sizing bug).
2. **Drawdown circuit breaker:** if the portfolio falls more than 12% from its high-water mark, go
   fully to cash and **halt automated trading pending human review.** This is the hard floor.
3. **Cash floor:** always keep a defined minimum in cash equivalents.
4. **Staleness guard:** if market data is older than N hours or a price moves more than X% versus
   the prior close, refuse to trade and alert. Bad ticks have caused real disasters.
5. **Order sanity check:** every order is validated against expected notional before submission. An
   order more than 3x the expected size is blocked automatically.
6. **No leverage. No margin. No options. No shorting. No crypto.** Every one of these can lose more
   than you put in, or is a fee sink. Cash accounts only.
7. **Position limit on new capital:** never add money to the system faster than the paper-trading
   record justifies.

Number 6 is the single largest determinant of whether a beginner loses catastrophically. Most
retail blowups are leverage or options, not bad stock picks.

---

## 9. Phased roadmap

Each phase has an **exit criterion**. You do not start the next phase until the current one is met.
The criteria exist to stop enthusiasm from outrunning evidence.

### Phase 0 — Foundations *(~1 week)*
Repo scaffolding, data fetching, caching, parquet storage. Handle splits and dividends correctly
(total-return prices, not raw prices — otherwise every backtest is wrong by several percent a year).
Plot SPY's history and eyeball it against a known chart.

*Exit: you can reproduce SPY's 2008 drawdown and 2020 crash to within a fraction of a percent.*

### Phase 1 — Baseline strategy *(~1 week)*
Implement dual momentum as a pure function. Config-driven. Unit tested. **No backtest yet** —
build the logic and prove it is correct on hand-checkable examples first.

*Exit: you can compute the strategy's decision for a given month by hand in a spreadsheet and it
matches the code exactly.*

### Phase 2 — Backtester and the truth *(~2-3 weeks — the real work)*
The custom engine, with an explicit cost model (commission, spread, slippage) and correct
point-in-time data handling. The anti-lookahead test suite. Then walk-forward validation: tune on
1990-2005, test untouched on 2006-2024.

*Exit: an out-of-sample tearsheet with CAGR, max drawdown, Sharpe, and a benchmark comparison —
plus a written honest assessment of whether it beats buy-and-hold. **This phase is allowed to kill
the project.***

### Phase 3 — Paper trading *(6 months minimum, calendar time, no shortcut)*
Alpaca paper account. Cron job. Every month the system logs its intended decision *before* acting,
then acts. Monthly written review comparing live paper results to backtest expectations.

*Exit: 6 months of paper results with no execution bugs, and live behavior tracking the backtest's
expected range. Any material divergence means a bug — go find it.*

### Phase 4 — ML augmentation *(optional, only if Phases 1-3 succeeded)*
Add a regime classifier and/or LLM-based text analysis of holdings. Must be A/B tested against the
Phase 1 baseline out-of-sample. **If it does not clearly beat the dumb baseline, it is deleted.**

*Exit: statistically meaningful improvement on out-of-sample data, or an honest decision not to
ship it.*

### Phase 5 — Real money *(only after all of the above)*
Start with an amount you would be genuinely fine losing entirely. Increase only after a year of
live results matching paper. Human approves every trade for the first year.

*Exit: none. This phase never ends, and it is always under review.*

**Realistic timeline: 8-10 months before real money is involved.** If that sounds slow, that
slowness *is* the product. The alternative timeline is "3 weeks to a live system and 4 months to
finding out it was broken the whole time."

---

## 10. How we know it is working

Every backtest and every monthly live review produces the same standard report:

- Equity curve vs SPY and vs cash on the same axes
- CAGR, volatility, Sharpe, Sortino, max drawdown, longest drawdown *duration*
- Worst 10 months, and what the system was holding during each
- Trade log with costs
- **Drawdown duration** gets special emphasis: it is not the depth of a loss that makes people
  quit, it is being underwater for 18 months while their friends make money in the index.

And one qualitative question answered in writing each month: *"Did the system do anything I did not
expect? If so, why?"* Unexplained behavior is treated as a bug until proven otherwise.

---

## 11. Costs

| Item | Cost |
|---|---|
| Market data (yfinance, FRED) | $0 |
| Alpaca paper + live trading | $0 |
| Compute (this machine) | $0 |
| LLM API for Phase 4 text analysis | ~$5-20/mo, only if Phase 4 happens |
| **Total through Phase 3** | **$0** |

The entire project through paper trading costs nothing but time. There is no reason to spend money
on data or infrastructure until free versions are demonstrably the limiting factor.

---

## 12. Things that will go wrong (predicted in advance)

Writing these down now so they can be checked against later, rather than rationalized away:

1. **The backtest will look too good the first time.** It will be a lookahead bug. It is always a
   lookahead bug.
2. **Live results will be worse than backtest.** Always. Costs, slippage, and the fact that the
   backtest period is over-fit even when you are careful.
3. **There will be a stretch where the strategy underperforms SPY badly** — probably 1-2 years.
   This is normal and expected for defensive strategies during bull markets, and it is precisely
   when people abandon a working system.
4. **The urge to add parameters after a bad month will be strong.** That urge is how strategies get
   destroyed. Changes require a written rationale in `DECISIONS.md` *and* out-of-sample evidence.
5. **Free data will have errors** — bad ticks, wrong split adjustments. Hence the staleness guard.

---

## 13. Practical and legal notes

- This is a personal tool, not financial advice, and not a product for anyone else. Managing other
  people's money requires registration; do not do it.
- **Brokerage accounts require you to be 18+.** If that is not the case yet, the path is a
  custodial account opened with a parent or guardian — worth sorting out early since it affects
  which broker to use. Phases 0-3 are entirely unaffected either way; paper trading has no such
  requirement.
- **Taxes:** selling at a profit in a taxable account creates a tax bill, and holdings sold within
  a year are taxed at a higher rate. This is a real drag on returns and the backtest must model it.
  Low turnover helps a great deal here.
- Keep records from day one. Future-you will need them.

---

## 14. Open questions to decide together

1. **Advisory or autonomous?** Should the system place trades itself, or email you a recommendation
   each month that you execute manually? *(Recommendation: advisory through Phase 4. It is safer,
   simpler, and at 2-6 trades a year the manual work is trivial.)*
2. **Capital size and timeline** — what amount, and when? This affects broker choice and whether
   fixed costs matter.
3. **Account type** — taxable, or a retirement account (which changes the tax math significantly)?
4. **Learning goal weighting** — is this primarily about building a good system, or primarily about
   learning markets and programming? Both are valid; they lead to different amounts of "write it
   yourself" versus "use the library."
5. **Universe** — stick to 4 broad ETFs, or include a defensive sleeve (gold, TIPS) from the start?

---

## 15. Immediate next steps

Nothing until the questions in §14 are settled. When they are, the first commit is Phase 0: data
fetching and the SPY reproduction check.

*Last updated: 2026-09-03*
