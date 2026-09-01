"""Paired Gate-A contexts with identical content at three time spans."""
from __future__ import annotations

import copy
import random
from typing import List

from ..schema import (
    ROLE_SYSTEM,
    ROLE_USER,
    SPAN_CROSS,
    SPAN_SESSION,
    SPAN_SINGLE,
    Record,
    Turn,
    read_jsonl,
)


def _chunks(record: Record, count: int) -> List[tuple[str, bool]]:
    units = []
    for turn in record.turns:
        if turn.role == ROLE_SYSTEM:
            continue
        words = turn.content.split()
        units.extend((word, bool(turn.is_attack)) for word in words)
    if not units:
        units = [("empty", bool(record.label))]
    count = max(1, min(count, len(units)))
    base, extra = divmod(len(units), count)
    chunks = []
    start = 0
    for index in range(count):
        width = base + int(index < extra)
        part = units[start : start + width]
        start += width
        chunks.append((" ".join(word for word, _ in part), any(flag for _, flag in part)))
    return chunks


def _variant(record: Record, span: str, chunk_count: int) -> Record:
    chunks = _chunks(record, chunk_count)
    if span == SPAN_SINGLE:
        chunks = [(" ".join(text for text, _ in chunks), any(flag for _, flag in chunks))]
    turns = []
    for index, (text, is_attack) in enumerate(chunks):
        session_id = index if span == SPAN_CROSS else 0
        turns.append(Turn(
            role=ROLE_USER,
            content=text,
            ts=index,
            session_id=session_id,
            is_attack=is_attack,
        ))
    attack_positions = [i for i, turn in enumerate(turns) if turn.is_attack]
    meta = dict(record.meta)
    meta.update({
        "pair_id": record.uid,
        "paired_source_uid": record.uid,
        "paired_span": span,
        "paired_content_words": sum(len(text.split()) for text, _ in chunks),
    })
    return Record(
        uid=f"{record.uid}::{span}",
        dataset="paired_gate_a",
        span=span,
        turns=turns,
        label=int(record.label),
        attack_turn_idx=attack_positions[0] if attack_positions else None,
        n_sessions=len(turns) if span == SPAN_CROSS else 1,
        meta=meta,
    )


def load_paired_gate_a(cfg: dict) -> List[Record]:
    source = read_jsonl(cfg.get("paired_records", "data/a3s_balanced.jsonl"))
    seed = int(cfg.get("seed", 0))
    limit = int(cfg.get("paired_max_records", 0))
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for record in source:
        by_label[int(record.label)].append(record)
    if limit:
        per_label = max(1, limit // 2)
        selected = []
        for label in (0, 1):
            pool = list(by_label[label])
            rng.shuffle(pool)
            selected.extend(pool[:per_label])
    else:
        selected = source
    spans = (SPAN_SINGLE, SPAN_SESSION, SPAN_CROSS)
    chunk_count = int(cfg.get("paired_chunks", 3))
    return [
        _variant(copy.deepcopy(record), span, chunk_count)
        for record in selected
        for span in spans
    ]
