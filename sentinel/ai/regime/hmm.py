"""A two-state Gaussian hidden Markov model, fitted with Baum-Welch.

The market is modelled as switching between a calm state and a stressed state,
each with its own mean and volatility, with the state itself unobserved. This is
the classical approach to regime detection and it is here rather than imported
for the same reason the random-walk tests are hand-written: it is the instrument
the project's central claim rests on, it is sixty lines, and the cost of
understanding it fully is far below the cost of trusting it blindly.

Filtered, not smoothed -- the distinction that decides whether this is usable
--------------------------------------------------------------------------
Every HMM library's headline method (`predict`, Viterbi decoding, the posteriors
from forward-backward) is **smoothed**: the estimate of the state on day t uses
the whole series, including everything after t. That is the right answer for
describing history and completely wrong for trading, because on day t the future
has not happened.

Smoothing makes regime detection look far easier than it is. A crash is obvious
once you can see the recovery. `filter()` here runs the forward recursion only,
so the estimate for day t uses observations up to and including t and nothing
after. `smooth()` exists as well, clearly named, and is used only to measure how
much is lost by being honest -- which is the interesting number.

Reaching for a library's default method here would have introduced lookahead that
looks like nothing, produces a plausible-seeming state sequence, and inflates
every downstream result. `tests/test_no_lookahead.py` covers the strategy built
on this, and would catch it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Variance is floored here so a state that collects almost no probability mass
#: cannot collapse to a spike of infinite likelihood on a single observation --
#: the standard degenerate solution of EM on Gaussian mixtures.
MIN_VARIANCE = 1e-12


@dataclass
class HMMParameters:
    """What the model learned. Two states, ordered calm first.

    Attributes:
        initial: state distribution before any observation.
        transition: `transition[i, j]` is P(state j tomorrow | state i today).
        means: per-state mean of one period's log return.
        variances: per-state variance of one period's log return.
    """

    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    @property
    def volatilities(self) -> np.ndarray:
        return np.sqrt(self.variances)

    @property
    def expected_durations(self) -> np.ndarray:
        """Average consecutive periods spent in each state before switching."""
        stay = np.diag(self.transition)
        return np.where(stay < 1.0, 1.0 / np.maximum(1.0 - stay, 1e-12), np.inf)


def _emission_likelihood(observations: np.ndarray, params: HMMParameters) -> np.ndarray:
    """P(observation | state) for every observation and state."""
    variance = np.maximum(params.variances, MIN_VARIANCE)
    deviation = observations[:, None] - params.means[None, :]
    return np.exp(-0.5 * deviation**2 / variance) / np.sqrt(2.0 * np.pi * variance)


def _forward(likelihood: np.ndarray, params: HMMParameters) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward recursion.

    Returns:
        `alpha[t]` = P(state at t | observations up to and including t), and the
        scaling factors, whose logs sum to the log-likelihood of the sequence.
    """
    n = len(likelihood)
    alpha = np.empty((n, 2))
    scale = np.empty(n)

    weighted = params.initial * likelihood[0]
    scale[0] = max(weighted.sum(), 1e-300)
    alpha[0] = weighted / scale[0]

    for t in range(1, n):
        weighted = (alpha[t - 1] @ params.transition) * likelihood[t]
        scale[t] = max(weighted.sum(), 1e-300)
        alpha[t] = weighted / scale[t]

    return alpha, scale


def _backward(likelihood: np.ndarray, params: HMMParameters, scale: np.ndarray) -> np.ndarray:
    """Scaled backward recursion, sharing the forward pass's scaling factors."""
    n = len(likelihood)
    beta = np.empty((n, 2))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = params.transition @ (likelihood[t + 1] * beta[t + 1]) / scale[t + 1]
    return beta


def _order_calm_first(params: HMMParameters) -> HMMParameters:
    """Put the lower-variance state first, so state 0 always means "calm".

    EM has no notion of which state is which -- the two are interchangeable and
    the fit lands on whichever ordering the initialisation happened to favour.
    Without this, half of all fits would come out with the labels swapped and
    every accuracy score would be a coin flip on top of the real answer.

    Volatility is the right thing to sort on rather than mean return: the
    difference in volatility between a calm and a stressed market is large and
    reliably estimated, while the difference in mean is small and drowned in
    noise over any realistic sample.
    """
    if params.variances[0] <= params.variances[1]:
        return params
    order = [1, 0]
    return HMMParameters(
        initial=params.initial[order],
        transition=params.transition[np.ix_(order, order)],
        means=params.means[order],
        variances=params.variances[order],
    )


class GaussianHMM2State:
    """Two-state Gaussian HMM: calm and stressed, fitted by Baum-Welch.

    Args:
        n_iterations: EM steps. Convergence on this problem is fast; the default
            is generous and `tolerance` usually stops it earlier.
        tolerance: stop when the log-likelihood improves by less than this.
        seed: only affects initialisation, which is deliberately deterministic
            given a seed. A model that silently redraws its own starting point
            produces different regimes on different runs of identical data.
    """

    def __init__(self, n_iterations: int = 100, tolerance: float = 1e-6, seed: int = 0) -> None:
        self.n_iterations = int(n_iterations)
        self.tolerance = float(tolerance)
        self.seed = int(seed)
        self.params: HMMParameters | None = None
        self.log_likelihood: float = -np.inf
        self.n_iterations_run: int = 0

    # -- fitting -------------------------------------------------------------

    def _initialise(self, observations: np.ndarray) -> HMMParameters:
        """Start from a volatility split rather than at random.

        Observations are divided by whether their magnitude is above or below the
        median, which is a crude but reliable first guess at calm versus
        stressed. EM on this problem has local optima, and starting somewhere
        sensible matters more than starting somewhere random: a random start
        lands on the degenerate one-state solution often enough to matter.
        """
        magnitude = np.abs(observations - np.median(observations))
        split = np.median(magnitude)
        calm = observations[magnitude <= split]
        stressed = observations[magnitude > split]
        if len(calm) < 2 or len(stressed) < 2:
            calm = stressed = observations

        return HMMParameters(
            initial=np.array([0.5, 0.5]),
            transition=np.array([[0.95, 0.05], [0.10, 0.90]]),
            means=np.array([calm.mean(), stressed.mean()]),
            variances=np.array(
                [max(calm.var(), MIN_VARIANCE), max(stressed.var() * 1.5, MIN_VARIANCE)]
            ),
        )

    def fit(self, observations: np.ndarray) -> GaussianHMM2State:
        """Estimate the parameters from a sequence of log returns."""
        observations = np.asarray(observations, dtype=float).ravel()
        if len(observations) < 20:
            raise ValueError("need at least 20 observations to fit two states")
        if not np.isfinite(observations).all():
            raise ValueError("observations contain non-finite values")

        params = self._initialise(observations)
        previous = -np.inf

        for iteration in range(self.n_iterations):
            likelihood = _emission_likelihood(observations, params)
            alpha, scale = _forward(likelihood, params)
            beta = _backward(likelihood, params, scale)

            log_likelihood = float(np.log(scale).sum())

            # gamma[t, i] = P(state i at t | all observations) -- smoothed, which
            # is correct here: fitting is allowed to use the whole training
            # sample. Only *inference at decision time* must be causal.
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            # xi[t, i, j] = P(state i at t, state j at t+1 | all observations)
            xi = (
                alpha[:-1, :, None]
                * params.transition[None, :, :]
                * (likelihood[1:] * beta[1:])[:, None, :]
                / np.maximum(scale[1:, None, None], 1e-300)
            )

            transition = xi.sum(axis=0)
            transition /= np.maximum(transition.sum(axis=1, keepdims=True), 1e-300)

            weight = np.maximum(gamma.sum(axis=0), 1e-300)
            means = (gamma * observations[:, None]).sum(axis=0) / weight
            variances = (gamma * (observations[:, None] - means) ** 2).sum(axis=0) / weight

            params = HMMParameters(
                initial=gamma[0] / gamma[0].sum(),
                transition=transition,
                means=means,
                variances=np.maximum(variances, MIN_VARIANCE),
            )

            self.n_iterations_run = iteration + 1
            if log_likelihood - previous < self.tolerance:
                previous = log_likelihood
                break
            previous = log_likelihood

        self.params = _order_calm_first(params)
        self.log_likelihood = previous
        return self

    # -- inference -----------------------------------------------------------

    def filter(self, observations: np.ndarray) -> np.ndarray:
        """P(state at t | observations up to and including t). **Causal.**

        This is the only inference method a strategy may use. Row t depends on
        rows 0..t of the input and nothing later, so appending observations never
        changes an earlier row -- the property `check_causality` tests for.
        """
        observations = np.asarray(observations, dtype=float).ravel()
        alpha, _ = _forward(_emission_likelihood(observations, self._fitted()), self._fitted())
        return alpha

    def predict_next(self, observations: np.ndarray) -> np.ndarray:
        """P(state at t+1 | observations up to t). One step ahead, still causal.

        This is the quantity a position sizer actually wants. Knowing which state
        yesterday was in is history; what matters is the state that will govern
        the return being decided on now, which is the filtered estimate pushed
        one step through the transition matrix.
        """
        return self.filter(observations) @ self._fitted().transition

    def smooth(self, observations: np.ndarray) -> np.ndarray:
        """P(state at t | ALL observations). **Not causal -- never trade on this.**

        Provided to measure the gap between what is knowable in hindsight and
        what is knowable at the time. That gap is the honest difficulty of regime
        detection, and quoting a smoothed accuracy as if it were achievable live
        is one of the most common ways regime models are oversold.
        """
        params = self._fitted()
        likelihood = _emission_likelihood(np.asarray(observations, dtype=float).ravel(), params)
        alpha, scale = _forward(likelihood, params)
        beta = _backward(likelihood, params, scale)
        gamma = alpha * beta
        return gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    def _fitted(self) -> HMMParameters:
        if self.params is None:
            raise RuntimeError("fit() must be called before inference")
        return self.params
