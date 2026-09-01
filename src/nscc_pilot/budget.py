"""Shared strict token-budget and dataset-routing helpers.

Both `run_stage1b_tokenbudget.py` and `run_contextual_tokenbudget.py` used to
inline (a) an ad-hoc dataset switch that only understood ``cstm`` and a raw
JSONL path, and (b) ``math.floor(budget * total)`` with no ``max(1, ...)`` floor.
The latter silently gives a *zero* token budget to short records (e.g. a 2-turn
InjecAgent case at 1%), which does not satisfy the hard problem's "retain 1-10%
of the context" requirement -- a scenario must keep at least one token when the
budget is positive. Centralising both here keeps the strict accounting identical
across every acceptance experiment.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .adapters import load_dataset
from .schema import Record

# Named datasets resolve to their adapter + default config. Anything else is
# treated as a path to a pre-exported JSONL file.
_DATASET_CONFIGS: Dict[str, Tuple[str, dict]] = {
    "cstm": ("cstm", {"cstm_dir": "../datasets/cstm-bench",
                      "cstm_splits": ["cross_session"]}),
    "a3s": ("a3s", {"a3s_dir": "../datasets/Agent3Sigma-Stage"}),
    "agentdojo": ("agentdojo", {"agentdojo_dir": "../datasets/agentdojo"}),
    "injecagent": ("injecagent", {"injecagent_dir": "../datasets/InjecAgent"}),
    "agentlab": ("agentlab", {}),
    "fixture": ("fixture", {}),
}


def resolve_dataset(records_arg: str, overrides: dict | None = None) -> List[Record]:
    """Load records from a named dataset adapter or a JSONL path.

    ``records_arg`` may be one of the named datasets (``cstm``, ``a3s``,
    ``agentdojo``, ``injecagent``, ``agentlab``, ``fixture``) or a path to a
    JSONL file exported by ``scripts/export_balanced_jsonl.py``.
    """
    key = records_arg.strip()
    if key in _DATASET_CONFIGS:
        name, cfg = _DATASET_CONFIGS[key]
        cfg = dict(cfg)
        if overrides:
            cfg.update(overrides)
        return load_dataset(name, cfg)
    return load_dataset("jsonl", {"jsonl_path": records_arg})


def strict_token_cap(budget: float, total_tokens: int) -> int:
    """Strict per-scenario token cap.

    Returns ``floor(budget * total)`` but never drops a positive-budget scenario
    to zero tokens, and never exceeds the scenario's own length.
    """
    if total_tokens <= 0 or budget <= 0.0:
        return 0
    cap = math.floor(budget * total_tokens)
    cap = max(1, cap)
    return min(cap, total_tokens)


def global_ratio(retained_tokens: List[int], total_tokens: List[int]) -> float:
    """Dataset-wide retained/original ratio (the global budget accounting)."""
    denom = sum(total_tokens)
    if denom <= 0:
        return float("nan")
    return sum(retained_tokens) / denom
