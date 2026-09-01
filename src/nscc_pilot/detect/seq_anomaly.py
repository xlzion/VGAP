"""Lightweight sequence / trajectory anomaly detector (detector class #2).

Trainable. Combines cheap trajectory features (turn counts, tool-call density,
URL count, imperative-verb density, role-transition stats) with a hashed
bag-of-words over the flattened context, and fits a logistic model. Being
trainable is what makes Protocol I vs II meaningful for this detector:
  * Protocol I  -> fit on RAW contexts, scored on compressed  (representation mismatch)
  * Protocol II -> fit on COMPRESSED contexts at the same ratio (info loss only)
"""
from __future__ import annotations

import re
from typing import List, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from .base import Detector
from ..schema import Record, ROLE_TOOL, ROLE_USER

_URL = re.compile(r"https?://|www\.", re.I)
_IMPERATIVE = re.compile(
    r"\b(ignore|disable|export|send|delete|escalate|override|forward|leak|"
    r"exfiltrate|bypass|disregard|reveal|transfer)\b", re.I)


def _traj_features(rec: Record) -> List[float]:
    turns = rec.turns
    n = max(1, len(turns))
    text = rec.flat_text()
    n_tool = sum(1 for t in turns if t.role == ROLE_TOOL)
    n_user = sum(1 for t in turns if t.role == ROLE_USER)
    n_url = len(_URL.findall(text))
    n_imp = len(_IMPERATIVE.findall(text))
    lens = [len(t.content.split()) for t in turns]
    # role-transition churn
    trans = sum(1 for a, b in zip(turns, turns[1:]) if a.role != b.role)
    return [
        float(len(turns)),
        float(rec.n_tokens()),
        float(np.mean(lens)) if lens else 0.0,
        float(np.max(lens)) if lens else 0.0,
        n_tool / n,
        n_user / n,
        float(n_url),
        float(n_imp),
        n_imp / n,
        trans / n,
        float(rec.n_sessions),
    ]


class SeqAnomalyDetector(Detector):
    name = "seq_anomaly"
    trainable = True

    def __init__(self, backend=None, cfg: dict | None = None):
        super().__init__(backend, cfg)
        self.vec = HashingVectorizer(n_features=2 ** 16, alternate_sign=False,
                                     ngram_range=(1, 2), norm="l2")
        self.clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        self._fitted = False

    def _matrix(self, recs: Sequence[Record]):
        X_text = self.vec.transform([r.flat_text() for r in recs])
        X_num = sparse.csr_matrix(np.array([_traj_features(r) for r in recs], dtype=float))
        return sparse.hstack([X_text, X_num]).tocsr()

    def fit(self, recs: Sequence[Record]) -> "SeqAnomalyDetector":
        X = self._matrix(recs)
        y = np.array([r.label for r in recs], dtype=int)
        if len(set(y.tolist())) < 2:
            # degenerate split; leave unfitted -> uniform scores
            self._fitted = False
            return self
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def score_batch(self, recs: Sequence[Record]) -> List[float]:
        if not self._fitted:
            return [0.5] * len(recs)
        X = self._matrix(recs)
        return self.clf.predict_proba(X)[:, 1].tolist()

    def score(self, rec: Record) -> float:
        return self.score_batch([rec])[0]
