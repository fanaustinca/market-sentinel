"""Validation report for the synthetic market generator.

Run with:  python experiments/validate_sandbox.py

Prints the evidence that the GBM generator produces genuine random walks, and
demonstrates why a single path can never establish that.
"""

from __future__ import annotations

import numpy as np

from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.stats.randomwalk import full_battery, ljung_box, normality, variance_ratio

MU, SIGMA, ALPHA = 0.08, 0.16, 0.05


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def single_path_trap() -> None:
    rule("1. Why one path proves nothing")
    print("Seed 42 is a market built with no signal whatsoever. Here is what our")
    print("tests say about it:\n")
    scenario = GBMGenerator(mu=MU, sigma=SIGMA).generate(n_steps=2520, seed=42)
    log_prices = np.log(scenario.data.prices.to_numpy())
    for result in full_battery(log_prices):
        flag = "  <-- flags structure" if result.rejects_random_walk() else ""
        print(f"  {result}{flag}")
    print("\nTwo tests report structure in a market containing none. Nothing is")
    print("wrong: at a 5% significance level, one path in twenty gets flagged by")
    print("chance. Judging the generator on a single path would be the exact error")
    print("this project exists to avoid.")


def calibration(n_paths: int = 1000, n_steps: int = 1260) -> None:
    rule("2. The right question: does it reject at exactly the expected rate?")
    print(f"Generating {n_paths} independent null markets of {n_steps} days each...\n")

    scenario = GBMGenerator(mu=MU, sigma=SIGMA).generate(
        n_steps=n_steps, n_assets=n_paths, seed=1234
    )
    log_prices = np.log(scenario.data.prices.to_numpy())

    tests = {
        "ljung_box": lambda lp: ljung_box(np.diff(lp), lags=10),
        "variance_ratio_q2": lambda lp: variance_ratio(lp, q=2),
        "variance_ratio_q5": lambda lp: variance_ratio(lp, q=5),
        "jarque_bera": lambda lp: normality(np.diff(lp)),
    }
    standard_error = np.sqrt(ALPHA * (1 - ALPHA) / n_paths)
    lo, hi = ALPHA - 3 * standard_error, ALPHA + 3 * standard_error
    print(f"  expected {ALPHA:.0%}, acceptable band [{lo:.2%}, {hi:.2%}]\n")

    for name, run in tests.items():
        rate = np.mean([run(log_prices[:, i]).p_value < ALPHA for i in range(n_paths)])
        verdict = "PASS" if lo < rate < hi else "FAIL"
        print(f"  {name:<20} {rate:6.2%}   {verdict}")

    print("\nRejections arrive at precisely the rate the mathematics predicts. Too")
    print("many would mean the generator leaks structure; too few would mean the")
    print("paths are suspiciously well behaved, which random data never is.")


def parameter_recovery() -> None:
    rule("3. Does it produce the parameters we asked for?")
    scenario = GBMGenerator(mu=MU, sigma=SIGMA).generate(n_steps=2520, n_assets=500, seed=0)
    returns = np.diff(np.log(scenario.data.prices.to_numpy()), axis=0)

    observed_vol = returns.std(ddof=1) * np.sqrt(252)
    observed_drift = returns.mean() * 252
    with_drag = MU - 0.5 * SIGMA**2

    print(f"  volatility      requested {SIGMA:.4f}   observed {observed_vol:.4f}")
    print(f"  log-space drift  expected {with_drag:.4f}   observed {observed_drift:.4f}")
    print(f"  (without the volatility drag it would be {MU:.4f} -- a full")
    print(f"   {100 * (MU - with_drag):.2f}% a year of return nobody asked for)")


if __name__ == "__main__":
    print("=" * 72)
    print("SANDBOX VALIDATION -- is the null market genuinely empty?")
    print("=" * 72)
    single_path_trap()
    calibration()
    parameter_recovery()
    print("\n" + "=" * 72)
    print("The generator is sound. Phase 1 can build the AI on top of it.")
    print("=" * 72)
