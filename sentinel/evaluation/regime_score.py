"""Scoring a regime classifier against the true state.

Only possible inside the sandbox, which is the whole reason the sandbox exists.
On real data nobody knows what regime the market was in, so a regime model can be
argued about but never graded. Here the generator wrote the answer down.

Four things are measured, and the project cares about them in roughly reverse
order of how often they get reported:

**Accuracy** is the number everyone quotes and the least useful one here. A
market that is calm 75% of the time hands 75% accuracy to a model that simply
never predicts stress. It is reported alongside *balanced* accuracy for that
reason, and neither should be read alone.

**Calibration** decides whether the probability can be used for sizing. A model
that says 70% must be right about 70% of the time, or a position sized off its
confidence is sized off a number that means nothing. Raw classifier probabilities
are almost never honest, which `plan.md` section 10 predicted in advance.

**Detection lag** is the one that actually matters for trading and is almost
never published. How many days after the market genuinely changes state does the
model notice? A classifier that is 95% accurate but three weeks late is useless:
it is right about the past and silent about the present, and its accuracy score
looks superb throughout. Accuracy hides lag completely, because lagged errors are
a small fraction of days.

**False alarms** are lag's opposite failure and must be counted with it. A model
tuned to notice stress instantly will also announce it constantly, and one number
without the other can be gamed in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RegimeScore:
    """How well a classifier tracked the true state."""

    n_scored: int
    accuracy: float
    balanced_accuracy: float
    auc: float
    brier: float
    calibration_error: float
    stressed_share: float
    detection_lags: np.ndarray
    missed_switches: int
    n_switches: int
    false_alarm_rate: float
    reliability: list = field(default_factory=list)

    @property
    def median_lag(self) -> float:
        """Days from a true regime change to the model noticing it."""
        return float(np.median(self.detection_lags)) if len(self.detection_lags) else float("nan")

    @property
    def p90_lag(self) -> float:
        """The slow tail -- the case that hurts, since it is the crash it missed."""
        return float(np.percentile(self.detection_lags, 90)) if len(self.detection_lags) else float("nan")

    @property
    def detected_fraction(self) -> float:
        return 1.0 - self.missed_switches / self.n_switches if self.n_switches else float("nan")

    def report(self) -> str:
        return (
            f"  scored on       {self.n_scored} days, {self.stressed_share:.1%} truly stressed\n"
            f"  accuracy        {self.accuracy:.1%}   balanced {self.balanced_accuracy:.1%}"
            f"   AUC {self.auc:.3f}\n"
            f"  calibration     Brier {self.brier:.4f}, mean |gap| {self.calibration_error:.3f}\n"
            f"  detection lag   median {self.median_lag:.0f}d, p90 {self.p90_lag:.0f}d "
            f"({self.detected_fraction:.0%} of {self.n_switches} switches caught)\n"
            f"  false alarms    {self.false_alarm_rate:.1%} of calm days called stressed"
        )


def _auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC, ties averaged. Equivalent to the Mann-Whitney statistic."""
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(len(probabilities), dtype=float)
    ranks[order] = np.arange(1, len(probabilities) + 1)

    # Average ranks within ties, or a model that outputs one constant value would
    # score 1.0 or 0.0 depending only on sort order.
    unique, inverse, counts = np.unique(probabilities, return_inverse=True, return_counts=True)
    summed = np.zeros(len(unique))
    np.add.at(summed, inverse, ranks)
    ranks = (summed / counts)[inverse]

    n_pos, n_neg = positive.sum(), negative.sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _reliability(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Predicted probability against observed frequency, bin by bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (probabilities >= low) & (probabilities < high if high < 1.0 else probabilities <= 1.0)
        if inside.sum() == 0:
            continue
        rows.append(
            {
                "bin": (round(float(low), 2), round(float(high), 2)),
                "n": int(inside.sum()),
                "predicted": float(probabilities[inside].mean()),
                "observed": float(labels[inside].mean()),
            }
        )
    return rows


def _detection_lags(
    predicted: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Days from each true regime change until the model agrees.

    A switch counts as missed if the model never agrees before the market
    switches back -- which is the honest accounting. A stress regime the model
    only recognises after it has ended was not detected late; it was not detected.
    """
    switches = np.flatnonzero(np.diff(truth) != 0) + 1
    if len(switches) == 0:
        return np.array([]), 0, 0

    boundaries = np.append(switches, len(truth))
    lags, missed = [], 0
    for start, end in zip(switches, boundaries[1:]):
        target = truth[start]
        agreed = np.flatnonzero(predicted[start:end] == target)
        if len(agreed):
            lags.append(int(agreed[0]))
        else:
            missed += 1
    return np.array(lags), missed, len(switches)


def score_regimes(
    stressed_probability: np.ndarray,
    true_regimes: np.ndarray,
    threshold: float = 0.5,
) -> RegimeScore:
    """Grade a classifier's output against the generator's answer key.

    Args:
        stressed_probability: P(stressed), one per day, aligned so that entry `i`
            is the model's belief about the state governing `true_regimes[i]`.
            `NaN` entries are dropped as warmup, not counted as wrong -- a model
            that has not seen enough data yet is silent, and scoring silence as
            error would penalise having been honest about it.
        true_regimes: 0 for calm, 1 for stressed.
    """
    probabilities = np.asarray(stressed_probability, dtype=float).ravel()
    truth = np.asarray(true_regimes, dtype=int).ravel()

    if len(probabilities) != len(truth):
        raise ValueError(
            f"got {len(probabilities)} probabilities for {len(truth)} labels; "
            "an alignment error here would silently score the model against the "
            "wrong day, which is the one mistake this function must not make"
        )

    scored = np.isfinite(probabilities)
    probabilities, truth = probabilities[scored], truth[scored]
    if len(truth) == 0:
        raise ValueError("no finite probabilities to score")

    predicted = (probabilities > threshold).astype(int)
    correct = predicted == truth

    calm_mask, stressed_mask = truth == 0, truth == 1
    recall_calm = float(correct[calm_mask].mean()) if calm_mask.any() else float("nan")
    recall_stressed = float(correct[stressed_mask].mean()) if stressed_mask.any() else float("nan")

    lags, missed, n_switches = _detection_lags(predicted, truth)
    reliability = _reliability(probabilities, truth)
    calibration_error = (
        float(np.average([abs(r["predicted"] - r["observed"]) for r in reliability],
                         weights=[r["n"] for r in reliability]))
        if reliability
        else float("nan")
    )

    return RegimeScore(
        n_scored=int(len(truth)),
        accuracy=float(correct.mean()),
        balanced_accuracy=float(np.nanmean([recall_calm, recall_stressed])),
        auc=_auc(probabilities, truth),
        brier=float(np.mean((probabilities - truth) ** 2)),
        calibration_error=calibration_error,
        stressed_share=float(truth.mean()),
        detection_lags=lags,
        missed_switches=missed,
        n_switches=n_switches,
        false_alarm_rate=float(predicted[calm_mask].mean()) if calm_mask.any() else float("nan"),
        reliability=reliability,
    )
