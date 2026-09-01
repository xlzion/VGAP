"""Detection metrics for the pilot.

Primary metric: Recall @ FPR=0.05 (threshold set on the benign score
distribution, recall measured on attacks). This is base-rate invariant, which
is why it is the headline number.

Secondary: PR-AUC (average precision) computed under a *realistic* base rate
via negative reweighting -- never 50/50 (see plan section 1). And
time-to-detect for the streaming/prefix path.

All point estimates come with a stratified bootstrap CI.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def recall_at_fpr(scores: Sequence[float], labels: Sequence[int],
                  fpr: float = 0.05) -> float:
    """Recall at a fixed false-positive rate.

    Threshold is the (1-fpr) quantile of the *negative* scores; recall is the
    fraction of positives at or above it. Base-rate invariant.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    # threshold s.t. at most `fpr` of negatives exceed it.
    thr = np.quantile(neg, 1.0 - fpr, method="higher")
    # ">=" with a tie-break: count positives strictly needed to hold FPR budget.
    return float(np.mean(pos >= thr))


def average_precision(scores, labels, base_rate: Optional[float] = None) -> float:
    """Average precision (PR-AUC).

    If base_rate is given, negatives are reweighted so that
    P(positive) == base_rate, making precision reflect a realistic prior
    instead of the (usually balanced) evaluation set.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    if base_rate is None:
        w_neg = 1.0
    else:
        # solve n_pos / (n_pos + w_neg * n_neg) = base_rate
        base_rate = min(max(base_rate, 1e-6), 1 - 1e-6)
        w_neg = (n_pos * (1 - base_rate)) / (base_rate * n_neg)

    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    w = np.where(y == 1, 1.0, w_neg)

    tp = np.cumsum(y * w)
    fp = np.cumsum((1 - y) * w)
    precision = tp / np.maximum(tp + fp, 1e-12)
    total_pos = tp[-1]
    # recall increments only at positive ranks
    rec = tp / max(total_pos, 1e-12)
    d_rec = np.diff(np.concatenate([[0.0], rec]))
    return float(np.sum(precision * d_rec))


def roc_auc(scores, labels) -> float:
    """Threshold-free separability (AUROC). Robust to a tiny negative pool,
    unlike Recall@FPR=0.05 -- this is the fair pillar metric when n_neg is small."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except Exception:
        # Mann-Whitney fallback with tie handling
        from scipy.stats import rankdata
        r = rankdata(scores)
        n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
        return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def time_to_detect(prefix_scores: Sequence[float], threshold: float,
                   attack_turn_idx: Optional[int]) -> Optional[int]:
    """Turns between attack onset and first threshold crossing.

    prefix_scores[i] = detector score using turns[:i+1]. Returns None if never
    detected or if the record has no attack.
    """
    if attack_turn_idx is None:
        return None
    ps = np.asarray(prefix_scores, dtype=float)
    fired = np.where(ps >= threshold)[0]
    fired = fired[fired >= attack_turn_idx]
    if len(fired) == 0:
        return None
    return int(fired[0] - attack_turn_idx)


def threshold_at_fpr(neg_scores: Sequence[float], fpr: float = 0.05) -> float:
    neg = np.asarray(neg_scores, dtype=float)
    return float(np.quantile(neg, 1.0 - fpr, method="higher"))


def bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 scores: Sequence[float], labels: Sequence[int],
                 n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Stratified bootstrap CI. Returns (point, lo, hi)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    point = metric_fn(scores, labels)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return point, float("nan"), float("nan")
    vals = []
    for _ in range(n_boot):
        bi = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        vals.append(metric_fn(scores[bi], labels[bi]))
    vals = np.asarray(vals, dtype=float)
    lo = float(np.nanquantile(vals, alpha / 2))
    hi = float(np.nanquantile(vals, 1 - alpha / 2))
    return float(point), lo, hi


def evaluate(scores: Sequence[float], labels: Sequence[int],
             fpr: float = 0.05, base_rate: float = 0.02,
             n_boot: int = 500, seed: int = 0) -> Dict[str, float]:
    """Full metric bundle for one (span, compressor, ratio, detector, protocol)."""
    r_pt, r_lo, r_hi = bootstrap_ci(
        lambda s, y: recall_at_fpr(s, y, fpr), scores, labels, n_boot, seed=seed)
    ap_pt, ap_lo, ap_hi = bootstrap_ci(
        lambda s, y: average_precision(s, y, base_rate), scores, labels, n_boot, seed=seed)
    return {
        "recall_at_fpr": r_pt, "recall_lo": r_lo, "recall_hi": r_hi,
        "pr_auc": ap_pt, "pr_auc_lo": ap_lo, "pr_auc_hi": ap_hi,
        "n_pos": int(np.sum(np.asarray(labels) == 1)),
        "n_neg": int(np.sum(np.asarray(labels) == 0)),
        "fpr": fpr, "base_rate": base_rate,
    }
