"""Regime detection: which state is the market in, and how sure are we?

The plan's centrepiece, and the component the sandbox exists to make measurable.
"What is the price tomorrow" is close to unanswerable; "is this a calm market or
a stressed one" is a far more tractable question, and it is the one that decides
position size.

It can only be scored properly here. On real data nobody knows the true regime,
so a classifier can be argued about but not graded. `RegimeSwitchingGenerator`
ships the true state for every day, which turns the argument into a measurement --
including the measurement that matters most and is never reported elsewhere:
**detection lag**, the number of days after a regime actually changes before the
model notices. A classifier that is 95% accurate but three weeks late is useless
for trading, and accuracy alone hides that completely.
"""

from sentinel.ai.regime.classifier import WalkForwardRegimeClassifier
from sentinel.ai.regime.hmm import GaussianHMM2State, HMMParameters

__all__ = ["GaussianHMM2State", "HMMParameters", "WalkForwardRegimeClassifier"]
