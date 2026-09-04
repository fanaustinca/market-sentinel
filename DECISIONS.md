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
