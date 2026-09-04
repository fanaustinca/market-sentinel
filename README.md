# market-sentinel

An AI trading system developed inside a **synthetic market laboratory** — where the ground truth is
known — before it is ever pointed at a real market.

**Status:** Phase 0 — the null-market generator is built and validated. See [plan.md](plan.md).

## The idea

You cannot tell whether a model trained on real market history learned something or memorized
noise, because there is only one copy of history and no answer key.

So this project builds the markets first. If you generate a market yourself, you know exactly what
is in it — whether it holds a pattern, what kind, and how strong. Two experiments follow:

**The Null Test.** Generate a market of pure randomness, with nothing to find. Run the AI on it.
*The AI must fail.* If it reports profit on noise, that is not a strategy — it is a bug, caught on
day one on fake money instead of eighteen months later on real money. Run it a thousand times and
you get a **noise floor**: what your method scores on markets containing nothing. Any real result
below that line is a coincidence, not a discovery.

**The Recovery Test.** Plant a pattern of known strength, then weaken it until the AI loses the
scent. That threshold is the AI's sensitivity. Real market signals are very weak — so if the
threshold sits above what real markets plausibly offer, you know the approach cannot work, from a
measurement rather than a hunch.

## The Reality Ladder

Each rung adds exactly one element of reality, so any failure is attributable.

| Rung | Market | Money | Proves |
|---|---|---|---|
| 1. Sandbox | Synthetic | Fake | The AI finds real signals — and correctly finds nothing in noise |
| 2. Backtest | Real history | Fake | It survives real messiness: fat tails, gaps, costs, crashes |
| 3. Paper | Real, live | Fake | The plumbing works in real time |
| 4. Live | Real, live | Real | It works when it matters |

The AI cannot tell which rung it is on — all three market sources share one interface.

## What this is not

- **Not financial advice.** Personal tool.
- **Not a guarantee against losses.** No such system exists; anything that can gain can lose.
- **Not a get-rich system.** It must beat a savings account, and it may fail to. The plan explicitly
  allows the conclusion "this doesn't work — buy an index fund."

Cost through paper trading: **$0**. Realistic timeline to real money: 8-10 months.

## License

MIT — see [LICENSE](LICENSE).

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

pytest                                  # 28 tests proving the null market is empty
python experiments/validate_sandbox.py  # the validation report, with commentary
```

## Progress

- [x] **Phase 0** — market interface, GBM null generator, random-walk test battery
- [ ] Phase 0 — remaining generators: AR(1), Ornstein-Uhlenbeck, regime-switching, jump, Heston
- [ ] Phase 1 — the AI, and the Null Test
- [ ] Phase 2 — the Recovery Test
