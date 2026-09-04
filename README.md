# market-sentinel

A defensive, rules-first market system — designed to lose slowly and predictably before it is ever
asked to win.

**Status:** planning. No code yet. See [plan.md](plan.md).

## What this is

A personal research project to build and rigorously validate a low-turnover, risk-managed
allocation strategy across broad ETFs. The design priority is capital preservation and bounded
drawdown, not maximum return.

Most of the engineering effort goes into *validation* — backtesting correctly, detecting lookahead
bias, and comparing honestly against a buy-and-hold benchmark — rather than into strategy
invention.

## What this is not

- **Not financial advice.** Personal tool only.
- **Not a guarantee against losses.** No such system exists. Any system that can gain can lose.
- **Not a get-rich system.** The benchmark it must beat is a savings account, and it may well fail
  to. The plan explicitly allows for the conclusion "just buy an index fund."

## Roadmap

| Phase | What | Gate |
|---|---|---|
| 0 | Data layer | Reproduce SPY history accurately |
| 1 | Baseline strategy | Hand-verifiable against a spreadsheet |
| 2 | Backtester + validation | Honest out-of-sample tearsheet |
| 3 | Paper trading | 6 months, no bugs, matches backtest |
| 4 | ML augmentation (optional) | Must beat the dumb baseline |
| 5 | Real money | Only after all of the above |

No real capital before Phase 5. Realistic timeline: 8-10 months.

## License

MIT — see [LICENSE](LICENSE).
