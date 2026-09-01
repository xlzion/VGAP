"""Compressors for E2 (compression-as-attack-surface).

Same token budget, three deletion policies:
  * OracleAdversarialCompressor -- drops the ATTACK turns FIRST (oracle upper
    bound of an attacker who controls / games the memory compressor to hide the
    chain: chain-splitting + evidence-flooding taken to their limit).
  * RandomCompressor -- drops turns at random (an attack-agnostic "benign"
    compressor at the same budget: the control).
  * BenignFirstCompressor -- drops benign filler first, KEEPS attack turns (the
    opposite extreme; a sanity ceiling -- detection should stay high).

E2 signal:  Delta = Recall(random) - Recall(adversarial)  at the same ratio.
Delta > 0 and growing as the budget tightens => the collapse is attacker-
controllable => "attack surface" is earned (not just "compression sensitivity").
"""
from __future__ import annotations

import copy
import random
from typing import List

from .base import Compressor
from ..schema import Record, ROLE_SYSTEM


def _keep_by_order(rec: Record, ordered_idx: List[int], budget: int):
    """Keep turns in the given priority order until the token budget is spent;
    always keep any system turns. Return kept turns in original temporal order."""
    sys_idx = [i for i, t in enumerate(rec.turns) if t.role == ROLE_SYSTEM]
    keep = set(sys_idx)
    used = sum(len(rec.turns[i].content.split()) for i in sys_idx)
    for i in ordered_idx:
        if i in keep:
            continue
        c = len(rec.turns[i].content.split())
        if used + c > budget and len(keep) > len(sys_idx):
            continue
        keep.add(i)
        used += c
    return [copy.copy(rec.turns[i]) for i in sorted(keep)]


class OracleAdversarialCompressor(Compressor):
    name = "adv_oracle"

    def compress(self, rec: Record) -> Record:
        budget = max(1, int(round(rec.n_tokens() * self.ratio)))
        # priority: keep benign turns first -> attack turns are the first dropped
        order = sorted(range(len(rec.turns)),
                       key=lambda i: (rec.turns[i].is_attack,))  # benign(False) first
        return self._clone(rec, _keep_by_order(rec, order, budget))


class RandomCompressor(Compressor):
    name = "random"

    def compress(self, rec: Record) -> Record:
        budget = max(1, int(round(rec.n_tokens() * self.ratio)))
        rng = random.Random(hash(rec.uid) & 0xFFFFFFFF)
        order = list(range(len(rec.turns)))
        rng.shuffle(order)
        return self._clone(rec, _keep_by_order(rec, order, budget))


class BenignFirstCompressor(Compressor):
    name = "keep_attack"

    def compress(self, rec: Record) -> Record:
        budget = max(1, int(round(rec.n_tokens() * self.ratio)))
        # priority: keep attack turns first (drop benign filler) -> detection ceiling
        order = sorted(range(len(rec.turns)),
                       key=lambda i: (not rec.turns[i].is_attack,))
        return self._clone(rec, _keep_by_order(rec, order, budget))
