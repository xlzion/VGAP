"""InjecAgent adapter -- third threat model (single-step tool-injection),
perfectly PAIRED and BALANCED.

Source: github uiuc-kang-lab/InjecAgent, data/test_cases_{dh,ds}_{base,enhanced}.json.
Each case ships BOTH:
  * 'Tool Response Template' = the clean tool output (no injection)  -> benign
  * 'Tool Response'          = the same output with the attacker instruction
                              injected                              -> attack
So for each case we emit two records that differ ONLY by the injection -- a
controlled benign/attack pair. This is the cleanest external-validity set for
the activation probe / attack-direction transfer.
"""
from __future__ import annotations

import json
import os
from typing import List

from ..schema import Record, Turn, SPAN_SINGLE, ROLE_USER, ROLE_TOOL


def _mk(uid, user_instr, tool_resp, is_attack, meta):
    turns = [
        Turn(role=ROLE_USER, content=str(user_instr), ts=0, session_id=0, is_attack=False),
        Turn(role=ROLE_TOOL, content=str(tool_resp), ts=1, session_id=0,
             is_attack=is_attack, tool_name=meta.get("User Tool")),
    ]
    return Record(uid=uid, dataset="injecagent", span=SPAN_SINGLE, turns=turns,
                  label=1 if is_attack else 0,
                  attack_turn_idx=1 if is_attack else None, n_sessions=1,
                  meta={"attack_type": meta.get("Attack Type"),
                        "split": meta.get("_split"),
                        "case_id": meta.get("_case_id")})


def load_injecagent(cfg: dict) -> List[Record]:
    root = cfg.get("injecagent_dir", "../datasets/InjecAgent")
    splits = cfg.get("injecagent_splits", ["dh_base", "ds_base"])
    recs: List[Record] = []
    for split in splits:
        path = os.path.join(root, "data", f"test_cases_{split}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"InjecAgent split not found: {path}")
        cases = json.load(open(path))
        for i, c in enumerate(cases):
            c = dict(c); c["_split"] = split; c["_case_id"] = f"{split}-{i}"
            ui = c.get("User Instruction", "")
            clean = c.get("Tool Response Template", "")
            inj = c.get("Tool Response", "")
            if inj:
                recs.append(_mk(f"inj-{split}-{i}-atk", ui, inj, True, c))
            if clean:
                recs.append(_mk(f"inj-{split}-{i}-ben", ui, clean, False, c))
    return recs


if __name__ == "__main__":
    import sys
    from collections import Counter
    cfg = {"injecagent_dir": sys.argv[1] if len(sys.argv) > 1 else "../datasets/InjecAgent"}
    recs = load_injecagent(cfg)
    print(f"loaded {len(recs)} records | label:", Counter(r.label for r in recs))
    tt = sum(len([t for t in r.turns if t.is_attack]) for r in recs)
    tot = sum(len(r.turns) for r in recs)
    print(f"turns: {tot} total, {tt} attack turns")
