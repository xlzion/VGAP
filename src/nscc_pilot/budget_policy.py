"""Budget-constrained allocation primitives for adaptive monitoring.

The allocator consumes only cheap per-fragment sketches and structural metadata.
It never receives labels or expensive activation features at allocation time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Allocation:
    selected: tuple[int, ...]
    used_cost: int
    budget: int

    @property
    def within_budget(self) -> bool:
        return self.used_cost <= self.budget


class BudgetLedger:
    """Enforces a hard integer cost cap and records every allocation."""

    def __init__(self, budget: int) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.budget = int(budget)
        self.used_cost = 0
        self.selected: list[int] = []

    def try_add(self, index: int, cost: int) -> bool:
        if cost < 0:
            raise ValueError("cost must be non-negative")
        if self.used_cost + cost > self.budget:
            return False
        self.selected.append(int(index))
        self.used_cost += int(cost)
        return True

    def freeze(self) -> Allocation:
        return Allocation(tuple(self.selected), self.used_cost, self.budget)


def contextual_features(
    sketch_scores: Sequence[float],
    scenario_ids: Sequence[int],
    positions: Sequence[float],
    session_positions: Sequence[float],
) -> np.ndarray:
    """Create label-free local and within-scenario sketch features.

    Scores are normalized per scenario to prevent a policy from exploiting a
    dataset-specific score scale. Neighbour summaries make the selector depend
    on coverage context rather than only reproducing raw-risk top-K.
    """
    scores = np.asarray(sketch_scores, dtype=float)
    groups = np.asarray(scenario_ids)
    pos = np.asarray(positions, dtype=float)
    sess = np.asarray(session_positions, dtype=float)
    if not (len(scores) == len(groups) == len(pos) == len(sess)):
        raise ValueError("all feature inputs must have the same length")

    result = np.zeros((len(scores), 6), dtype=float)
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        local = scores[indices]
        mean = float(local.mean())
        std = float(local.std())
        normalized = (local - mean) / max(std, 1e-6)
        order = np.argsort(pos[indices], kind="stable")
        ordered = local[order]
        left = np.concatenate(([ordered[0]], ordered[:-1]))
        right = np.concatenate((ordered[1:], [ordered[-1]]))
        neighbour = (left + right) / 2.0
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        result[indices, 0] = local
        result[indices, 1] = normalized
        result[indices, 2] = neighbour[inverse]
        result[indices, 3] = pos[indices]
        result[indices, 4] = sess[indices]
        result[indices, 5] = len(indices)
    return result


class ContextualBudgetPolicy:
    """A small learned scorer followed by a hard cost-constrained allocator."""

    def __init__(self, c: float = 0.2) -> None:
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=c, max_iter=3000, class_weight="balanced", random_state=0
        )
        self._fitted = False

    def fit(self, features: np.ndarray, labels: Sequence[int]) -> "ContextualBudgetPolicy":
        labels = np.asarray(labels, dtype=int)
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")
        if len(np.unique(labels)) != 2:
            raise ValueError("policy fitting requires both classes")
        self.model.fit(self.scaler.fit_transform(features), labels)
        self._fitted = True
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit the policy before scoring")
        return self.model.predict_proba(self.scaler.transform(features))[:, 1]

    def allocate(
        self,
        scores: Sequence[float],
        costs: Sequence[int],
        budget: int,
    ) -> Allocation:
        scores = np.asarray(scores, dtype=float)
        costs = np.asarray(costs, dtype=int)
        if len(scores) != len(costs):
            raise ValueError("scores and costs must have the same length")
        ledger = BudgetLedger(budget)
        # Score per cost makes variable-size fragments compete fairly.
        density = scores / np.maximum(costs, 1)
        for index in np.argsort(-density, kind="stable"):
            ledger.try_add(int(index), int(costs[index]))
        return ledger.freeze()


def grouped_policy_scores(
    features: np.ndarray,
    labels: Sequence[int],
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Return OOF policy scores; every test row is scored by an unseen policy."""
    labels = np.asarray(labels, dtype=int)
    scores = np.zeros(len(labels), dtype=float)
    for train_indices, test_indices in folds:
        policy = ContextualBudgetPolicy().fit(features[train_indices], labels[train_indices])
        scores[test_indices] = policy.score(features[test_indices])
    return scores
