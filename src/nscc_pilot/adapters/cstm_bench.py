"""CSTM-Bench adapter (wired to the real released schema).

Source: HF dataset `intrinsec-ai/cstm-bench`, two 54-scenario splits
(`dilution`, `cross_session`). Each ROW is a *scenario*; we map one scenario ->
one normalized Record.

Real schema (confirmed against the parquet):
  row: scenario_id, taxonomy, attack_name, scenario_class{attack,
       benign_pristine, benign_hard}, modality, num_sessions,
       num_total_messages, num_attack_messages, snr,
       sessions_json (str), ground_truth_json (str)
  sessions_json -> [ {session_id, cert, identity_anchor (the agent policy),
       messages: [ {message_id, text, image_b64, is_attack_fragment,
                    attack_fragment_index, ground_truth_labels,
                    timestamp_relative_ms} ]} ]
  ground_truth_json -> {attack_type, total_fragments,
       detection_should_trigger_by_fragment, session_narrative_roles[],
       arcs:[{arc_id, session_indices[]}], ...}

Mapping decisions:
  * label = 1 iff scenario_class == 'attack', else 0 (both benign classes -> 0).
  * span  = 'cross_session' (every scenario spans >=14 sessions -- CSTM is the
            cross-session battlefield). Single/session tiers come from AgentLAB
            / InjecAgent.
  * is_attack (per turn) = message['is_attack_fragment'].
  * attack_turn_idx = first attack fragment (for time-to-detect); the paper's
    `detection_should_trigger_by_fragment` is carried in meta as the target.
  * each session's identity_anchor is emitted as a leading system turn so the
    policy the attack violates is visible to the detectors.

Default split is `cross_session` (the on-thesis, adversarially-rewritten shard).
Set cfg['cstm_splits'] = ['dilution','cross_session'] to also load the dilution
shard -- but note the two share the same scenario skeleton, so the recall DIFF
between them isolates the rewrite effect rather than an independent sample.
"""
from __future__ import annotations

import glob
import json
import os
from typing import List

import pandas as pd

from ..schema import (Record, Turn, SPAN_SESSION, SPAN_CROSS,
                      ROLE_USER, ROLE_SYSTEM)


def load_cstm(cfg: dict) -> List[Record]:
    local_dir = cfg.get("cstm_dir", "datasets/cstm-bench")
    splits = cfg.get("cstm_splits", ["cross_session"])
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"CSTM-Bench not found at {local_dir}. Download it and set "
            f"cfg['cstm_dir'], or use dataset='fixture'.")
    records: List[Record] = []
    for split in splits:
        files = sorted(glob.glob(os.path.join(local_dir, "data", f"{split}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet for split {split!r} under {local_dir}/data")
        for path in files:
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                records.append(_row_to_record(row, split))
    if cfg.get("cstm_facet") == "dilution":
        _apply_dilution_facet(records)
    return records


def _apply_dilution_facet(records: List[Record]) -> None:
    """Re-label `span` by DILUTION level so Gate A's span facet measures a
    within-CSTM axis instead of cross_session-vs-single (which would be
    cross-dataset). Least-diluted attacks -> 'single', most-diluted ->
    'cross_session'. This removes the dataset confound: same scenarios, same
    detectors, only dilution varies.

    Dilution metric = benign filler per attack (higher snr and lower attack
    density = more diluted). Benign scenarios carry no dilution level, so they
    are spread evenly across the three bins to give each bin FPR negatives.
    """
    from ..schema import SPAN_SINGLE, SPAN_SESSION, SPAN_CROSS

    def dilution(r: Record) -> float:
        snr = r.meta.get("snr") or 0
        n_atk = max(1, r.meta.get("num_attack_messages", 1))
        density = n_atk / max(1, len(r.turns))   # low density = more diluted
        return snr - 5.0 * density                # higher = more diluted

    attacks = sorted([r for r in records if r.label == 1], key=dilution)
    benign = [r for r in records if r.label == 0]
    n = len(attacks)
    t1, t2 = n // 3, 2 * n // 3
    bins = ([SPAN_SINGLE] * t1 + [SPAN_SESSION] * (t2 - t1) + [SPAN_CROSS] * (n - t2))
    for r, s in zip(attacks, bins):
        r.span = s
        r.meta["dilution"] = round(dilution(r), 3)
    for i, r in enumerate(benign):
        r.span = (SPAN_SINGLE, SPAN_SESSION, SPAN_CROSS)[i % 3]


def _row_to_record(row, split: str) -> Record:
    sessions = json.loads(row["sessions_json"])
    gt = json.loads(row["ground_truth_json"])
    label = 1 if row["scenario_class"] == "attack" else 0

    turns: List[Turn] = []
    attack_turn_idx = None
    ts = 0
    for si, sess in enumerate(sessions):
        anchor = sess.get("identity_anchor")
        if anchor:
            turns.append(Turn(role=ROLE_SYSTEM, content=anchor, ts=ts,
                              session_id=si, is_attack=False,
                              meta={"session_id": sess.get("session_id")}))
            ts += 1
        for m in sess.get("messages", []):
            is_atk = bool(m.get("is_attack_fragment", False))
            text = m.get("text") or ""
            turns.append(Turn(role=ROLE_USER, content=str(text), ts=ts,
                              session_id=si, is_attack=is_atk,
                              meta={"message_id": m.get("message_id"),
                                    "fragment_index": m.get("attack_fragment_index")}))
            if is_atk and attack_turn_idx is None:
                attack_turn_idx = len(turns) - 1
            ts += 1

    n_sessions = int(row["num_sessions"])
    span = SPAN_CROSS if n_sessions > 1 else SPAN_SESSION
    arc_sessions = sorted({idx for arc in gt.get("arcs", [])
                           for idx in arc.get("session_indices", [])})
    return Record(
        uid=str(row["scenario_id"]), dataset="cstm", span=span, turns=turns,
        label=label, attack_turn_idx=attack_turn_idx, n_sessions=n_sessions,
        meta={
            "split": split,
            "taxonomy": row["taxonomy"],
            "scenario_class": row["scenario_class"],
            "attack_name": row["attack_name"],
            "modality": row["modality"],
            "num_attack_messages": int(row["num_attack_messages"]),
            "snr": int(row["snr"]),
            "arc_session_indices": arc_sessions,
            "detection_should_trigger_by_fragment":
                gt.get("detection_should_trigger_by_fragment"),
        })


if __name__ == "__main__":  # sanity dump
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "datasets/cstm-bench"
    recs = load_cstm({"cstm_dir": d, "cstm_splits": ["cross_session"]})
    from collections import Counter
    print(f"loaded {len(recs)} records")
    print("labels:", Counter(r.label for r in recs))
    print("spans:", Counter(r.span for r in recs))
    r = next(x for x in recs if x.label == 1)
    print(f"\nexample attack {r.uid}: {len(r.turns)} turns, {r.n_sessions} sessions, "
          f"attack_turn_idx={r.attack_turn_idx}, n_tokens={r.n_tokens()}")
    print("meta:", {k: r.meta[k] for k in ("taxonomy", "num_attack_messages", "snr")})
