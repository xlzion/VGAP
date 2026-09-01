"""Detector interface.

A detector maps a Record (possibly compressed) to a risk score in [0, 1].
`fit` is optional: trainable detectors use it for Protocol II (fit on
compressed data). LLM-judge and rule-based detectors are training-free, so
their Protocol I and II results coincide (the orchestrator notes this).
"""
from __future__ import annotations

from typing import List, Sequence

from ..schema import Record


class Detector:
    name = "base"
    trainable = False

    def __init__(self, backend=None, cfg: dict | None = None):
        self.backend = backend
        self.cfg = cfg or {}

    def fit(self, recs: Sequence[Record]) -> "Detector":
        return self  # training-free by default

    def score(self, rec: Record) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def score_batch(self, recs: Sequence[Record]) -> List[float]:
        return [self.score(r) for r in recs]
