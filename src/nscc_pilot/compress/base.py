"""Compressor interface.

A compressor maps a Record's context to a *compressed* Record at a target
`ratio` = fraction of the original whitespace tokens to keep (0.30 / 0.10 /
0.05 in the pilot). Compressors must preserve the schema (roles, ts,
session_id, is_attack ground truth on surviving turns) so detectors and
time-to-detect still work on the output.
"""
from __future__ import annotations

import copy
from typing import List

from ..schema import Record, Turn


class Compressor:
    name = "base"

    def __init__(self, ratio: float, backend=None, cfg: dict | None = None):
        self.ratio = float(ratio)
        self.backend = backend
        self.cfg = cfg or {}

    def compress(self, rec: Record) -> Record:  # pragma: no cover - interface
        raise NotImplementedError

    def compress_batch(self, recs: List[Record]) -> List[Record]:
        return [self.compress(r) for r in recs]

    # helper for subclasses
    def _clone(self, rec: Record, turns: List[Turn]) -> Record:
        out = copy.copy(rec)
        out.turns = turns
        out.meta = dict(rec.meta)
        out.meta["compressor"] = self.name
        out.meta["ratio"] = self.ratio
        # keep attack_turn_idx pointing at the first surviving attack turn
        atk_positions = [i for i, t in enumerate(turns) if t.is_attack]
        out.attack_turn_idx = atk_positions[0] if atk_positions else None
        # renumber ts to the compressed order (preserve for time-to-detect)
        for i, t in enumerate(turns):
            t.ts = i
        return out


def keep_budget_tokens(rec: Record) -> int:
    """Token budget for this record at the compressor's ratio is applied by the
    caller; here we expose the original size."""
    return rec.n_tokens()
