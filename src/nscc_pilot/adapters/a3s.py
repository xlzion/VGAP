"""A3S-Bench (Agent3Sigma-Stage) adapter -- the pure-mosaic testbed.

Source: github antgroup/Agent3Sigma-Stage. 424 benign seeds + 726 adversarial
injections (each injected conv built from a benign seed; early turns are the
benign seed verbatim, later turns carry the payload). Per-turn attack labels are
BUILT IN: turn['injection_in_user'] / turn['injection_in_tool'].

Attack techniques (meta['technique']) span the mosaic spectrum:
  Fragmented payload assembly / Context-rich request embedding / Gradual
  extraction / Staged multi-step / Multi-turn progressive escalation  -> MOSAIC
  Direct request                                                       -> non-mosaic control
This lets us CERTIFY (per-turn probe AUROC ~0.5 on mosaic techniques) and test
whether session-level aggregation (L2) beats per-turn max-pool where it should.
"""
from __future__ import annotations

import json
import os
from typing import List

from ..schema import Record, Turn, SPAN_SESSION, ROLE_USER, ROLE_TOOL


def _turn_text(t: dict) -> str:
    parts = []
    if t.get("user"):
        parts.append("[user] " + str(t["user"]))
    if t.get("tool_response"):
        parts.append(f"[tool:{t.get('tool_name')}] " + str(t["tool_response"]))
    return "\n".join(parts)


def _conv_to_record(c: dict, label: int, uid: str) -> Record:
    turns, attack_idx = [], None
    for j, t in enumerate(c.get("turns", [])):
        is_atk = bool(t.get("injection_in_user") or t.get("injection_in_tool"))
        turns.append(Turn(role=ROLE_USER, content=_turn_text(t), ts=j, session_id=0,
                          is_attack=is_atk, tool_name=t.get("tool_name")))
        if is_atk and attack_idx is None:
            attack_idx = j
    return Record(uid=uid, dataset="a3s", span=SPAN_SESSION, turns=turns, label=label,
                  attack_turn_idx=attack_idx, n_sessions=1,
                  meta={"technique": c.get("technique", "benign"),
                        "risk_category": c.get("risk_category"),
                        "scenario": c.get("scenario"), "seed_id": c.get("seed_id")})


def load_a3s(cfg: dict) -> List[Record]:
    root = cfg.get("a3s_dir", "../datasets/Agent3Sigma-Stage")
    d = os.path.join(root, "data", "advance")
    seeds = json.load(open(os.path.join(d, "seeds.json")))
    inj = json.load(open(os.path.join(d, "injected.json")))
    recs: List[Record] = []
    for s in seeds:
        recs.append(_conv_to_record(s, 0, f"a3s-seed-{s.get('id')}"))
    for a in inj:
        recs.append(_conv_to_record(a, 1, f"a3s-inj-{a.get('id')}"))
    return recs


if __name__ == "__main__":
    import sys
    from collections import Counter
    recs = load_a3s({"a3s_dir": sys.argv[1] if len(sys.argv) > 1 else "../datasets/Agent3Sigma-Stage"})
    print(f"loaded {len(recs)} conversations | label:", Counter(r.label for r in recs))
    tt = sum(len([t for t in r.turns if t.is_attack]) for r in recs)
    tot = sum(len(r.turns) for r in recs)
    print(f"turns: {tot} total, {tt} attack turns ({100*tt/tot:.1f}%)")
    print("techniques:", Counter(r.meta["technique"] for r in recs if r.label == 1).most_common())
