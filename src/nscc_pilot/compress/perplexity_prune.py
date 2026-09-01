"""LLMLingua-style perplexity pruning (self-contained fallback).

The real `llmlingua` package cannot be pip-installed on the air-gapped cluster,
so this implements its core mechanism directly: score each context unit by the
information it carries (per-token perplexity under a small local LM) and drop
the *lowest-information* (most predictable, hence most redundant) units until
the token budget is met.

Granularity here is turn-level (cheap, runnable today). To upgrade to the
official token-level LLMLingua later, keep this class name/interface and swap
the body -- nothing downstream changes. See README "Compression method #3".
"""
from __future__ import annotations

import copy
from typing import List

import numpy as np

from .base import Compressor
from ..schema import Record, ROLE_SYSTEM


class PerplexityPruneCompressor(Compressor):
    name = "ppl_prune"

    def compress(self, rec: Record) -> Record:
        return self.compress_batch([rec])[0]

    def compress_batch(self, recs: List[Record]) -> List[Record]:
        # batch every non-system turn's text through the LM once. Each turn is
        # capped to a short SAMPLE (default 120 words) -- the salience score is a
        # mean per-token perplexity, and a leading sample estimates it fine while
        # keeping the full-vocab log_softmax bounded (CSTM turns can embed
        # thousands of tokens of history, which otherwise OOMs the logprob pass).
        sample_words = (self.cfg or {}).get("ppl_sample_words", 120)
        unit_texts, index = [], []
        for ri, r in enumerate(recs):
            for ti, t in enumerate(r.turns):
                if t.role == ROLE_SYSTEM:
                    continue
                unit_texts.append(" ".join(t.content.split()[:sample_words]))
                index.append((ri, ti))
        logprobs = self.backend.prompt_logprobs(unit_texts) if unit_texts else []

        # info score = mean negative logprob (higher = more surprising = keep)
        scores = {}
        for (ri, ti), lps in zip(index, logprobs):
            scores[(ri, ti)] = float(-np.mean(lps)) if lps else 0.0

        out = []
        for ri, r in enumerate(recs):
            budget = max(1, int(round(r.n_tokens() * self.ratio)))
            sys_turns = [(i, t) for i, t in enumerate(r.turns) if t.role == ROLE_SYSTEM]
            cand = [(i, t) for i, t in enumerate(r.turns) if t.role != ROLE_SYSTEM]
            # rank candidates by info score, keep greedily under budget
            cand_sorted = sorted(cand, key=lambda it: scores.get((ri, it[0]), 0.0),
                                 reverse=True)
            used = sum(len(t.content.split()) for _, t in sys_turns)
            keep_idx = {i for i, _ in sys_turns}
            for i, t in cand_sorted:
                c = len(t.content.split())
                if used + c > budget and len(keep_idx) > len(sys_turns):
                    continue
                keep_idx.add(i)
                used += c
            kept = [copy.copy(t) for i, t in enumerate(r.turns) if i in keep_idx]
            out.append(self._clone(r, kept))
        return out
