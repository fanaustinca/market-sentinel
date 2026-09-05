"""FinBERT over earnings press releases, and what it is being asked to beat.

FinBERT is a BERT model fine-tuned on financial phrasing, so it knows that
"headwinds" is bad news and "record quarter" is good, which general sentiment
models routinely get wrong. It is the right tool for the job. The question this
module exists to answer is whether the job is worth doing, and the oracle
experiment already set a hard ceiling on that: a model with *perfect* knowledge
of which way each earnings reaction went earns Sharpe +0.25 once the release is
aligned to a day it could actually be traded. FinBERT does not have perfect
knowledge, so whatever it achieves must be a fraction of +0.25.

That framing is the point. Without it, a small positive number from a language
model reads like a discovery. With it, the same number reads as a fraction of a
ceiling that was already too low to trade, which is what it is.

How the text is scored
----------------------
A press release is longer than BERT's 512-token window, so it is split into
sentences and scored in chunks, then combined. Two aggregates come out and they
are not interchangeable:

`tone` is the mean of (positive - negative) across chunks. It is a *direction*
estimate and is graded against the oracle ceiling.

`intensity` is the mean of (positive + negative) -- how strongly the document
commits either way, regardless of which way. It is a *magnitude* estimate, and
magnitude is the thing this project keeps finding to be predictable when
direction is not. A release full of confident claims in either direction plausibly
moves a stock more than one full of hedged neutralities.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "ProsusAI/finbert"

#: FinBERT's window is 512 tokens; chunks are built well under it so that
#: sentence boundaries never force a mid-word truncation.
MAX_TOKENS = 320

#: Sentences shorter than this are headers, tickers, and table fragments. They
#: score as confident neutrals and dilute every average they enter.
MIN_SENTENCE_CHARS = 40

#: Filing boilerplate that precedes the actual release text.
_BOILERPLATE = re.compile(
    r"^\s*(EX-\d+(\.\d+)?\s*\d*\s*\S+\.(htm|html|txt)\s*)?(EX-\d+(\.\d+)?)?\s*"
    r"(- PRESS RELEASE)?\s*(Document)?\s*(Exhibit\s+\d+(\.\d+)?)?\s*",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def clean(text: str) -> str:
    """Strip the EDGAR exhibit header so scoring starts at the actual headline."""
    return _BOILERPLATE.sub("", text or "", count=1).strip()


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(clean(text)) if len(s) >= MIN_SENTENCE_CHARS]


class FinBERTScorer:
    """Batch sentiment scoring, cached to disk because the model is the slow part.

    The cache key is a hash of the text itself rather than the filing it came
    from, so re-running after changing the extraction logic correctly recomputes
    instead of silently serving scores for text that is no longer being used.
    """

    def __init__(self, cache_path: Path | None = None, device: str | None = None,
                 batch_size: int = 64) -> None:
        self.cache_path = cache_path
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._tokenizer = None
        self._cache: dict[str, tuple[float, float, float]] = {}
        if cache_path is not None and cache_path.exists():
            prior = pd.read_parquet(cache_path)
            self._cache = {
                row.key: (row.positive, row.negative, row.neutral)
                for row in prior.itertuples()
            }

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self._model = model.to(self._device).eval()

        # Label order is read from the config rather than assumed. ProsusAI's
        # checkpoint is positive/negative/neutral, which is *not* the order most
        # examples online assume, and getting it wrong silently inverts the sign
        # of every score while leaving the magnitudes looking entirely plausible.
        labels = {v.lower(): k for k, v in model.config.id2label.items()}
        self._index = (labels["positive"], labels["negative"], labels["neutral"])

    def score_texts(self, texts: list[str]) -> np.ndarray:
        """Probabilities for each string, shaped (n, 3) as positive/negative/neutral."""
        import torch

        keys = [hashlib.sha256(t.encode()).hexdigest()[:24] for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]

        if missing:
            self._load()
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start : start + self.batch_size]
                batch = self._tokenizer(
                    [t for _, t in chunk], return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_TOKENS,
                ).to(self._device)
                with torch.no_grad():
                    probs = torch.softmax(self._model(**batch).logits, dim=-1).cpu().numpy()
                for (key, _), row in zip(chunk, probs):
                    self._cache[key] = tuple(float(row[i]) for i in self._index)
            self.save()

        return np.array([self._cache[k] for k in keys], dtype=float)

    def score_document(self, text: str) -> dict[str, float]:
        """Aggregate a whole press release into tone, intensity, and coverage.

        Returns NaN rather than 0.0 for an unscoreable document. Zero would mean
        "read it, felt neutral", which is a real and different finding from "there
        was nothing to read", and averaging the two together quietly fills gaps
        with fake neutrality.
        """
        parts = sentences(text)
        if not parts:
            return {"tone": np.nan, "intensity": np.nan, "n_sentences": 0}
        probs = self.score_texts(parts)
        positive, negative = probs[:, 0], probs[:, 1]
        return {
            "tone": float(np.mean(positive - negative)),
            "intensity": float(np.mean(positive + negative)),
            "n_sentences": len(parts),
        }

    def save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"key": k, "positive": p, "negative": n, "neutral": u}
             for k, (p, n, u) in self._cache.items()]
        ).to_parquet(self.cache_path)
