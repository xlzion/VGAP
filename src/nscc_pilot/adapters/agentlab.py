"""AgentLAB adapter (wired to the released static data).

Source: GitHub `TanqiuJiang/AgentLAB`. AgentLAB is primarily an *attack-
generation framework* (planner/attacker/victim/judge, needs API keys + a served
attacker model). Its `data/` ships two static pools we can use without running
rollouts:

  data/filtered_top_200_attacks.json  -- 200 top attack cases, each with a full
      `interaction_history` (list of {role, content}) and a `malicious_score`
      (6-10). SESSION-tier, multi-turn agent trajectories.
  data/all_refused_combined_200.json  -- 200 single-instruction harmful requests
      with an environment spec and a baseline `evaluation` (REFUSED). SINGLE-tier.

>>> IMPORTANT CAVEAT (read before using for recall@FPR) <<<
Both pools are attack-heavy: `filtered_top_200` is ALL malicious, and
`all_refused` are all should-refuse harmful requests. AgentLAB's static release
does NOT ship matched benign negatives. Recall@FPR needs a benign score
distribution, so for a *balanced* session/single-tier metric you must supply
negatives -- e.g. run the benign task variants through the same envs, or borrow
CSTM's benign scenarios. Until then, treat AgentLAB here as an attack-only
stress set (positives), and run the balanced Gate-A metric on CSTM.
The adapter therefore exposes a `agentlab_benign_score_thresh` weak-label knob
(default None = every case is label 1); set it to split by malicious_score only
if you explicitly want a weak-label gradient and document it.
"""
from __future__ import annotations

import json
import os
from typing import List

from ..schema import (Record, Turn, SPAN_SINGLE, SPAN_SESSION,
                      ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL, ROLE_SYSTEM)

_ROLE_MAP = {"user": ROLE_USER, "assistant": ROLE_ASSISTANT, "agent": ROLE_ASSISTANT,
             "system": ROLE_SYSTEM, "tool": ROLE_TOOL, "function": ROLE_TOOL,
             "observation": ROLE_TOOL}


def load_agentlab(cfg: dict) -> List[Record]:
    root = cfg.get("agentlab_dir", "datasets/AgentLAB")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"AgentLAB not found at {root}. Clone it and set cfg['agentlab_dir'], "
            f"or use dataset='fixture'.")
    which = cfg.get("agentlab_pools", ["session", "single"])
    thresh = cfg.get("agentlab_benign_score_thresh")  # None -> all label 1
    records: List[Record] = []
    if "session" in which:
        p = os.path.join(root, "data", "filtered_top_200_attacks.json")
        if os.path.exists(p):
            records.extend(_load_session_attacks(p, thresh))
    if "single" in which:
        p = os.path.join(root, "data", "all_refused_combined_200.json")
        if os.path.exists(p):
            records.extend(_load_single_refused(p))
    return records


def _load_session_attacks(path: str, thresh) -> List[Record]:
    data = json.load(open(path))
    out = []
    for i, ex in enumerate(data):
        ih = ex.get("interaction_history") or []
        turns, attack_idx = [], None
        for j, m in enumerate(ih):
            role = _ROLE_MAP.get(str(m.get("role", "user")).lower(), ROLE_USER)
            turns.append(Turn(role=role, content=str(m.get("content", "")),
                              ts=j, session_id=0, is_attack=False))
        # weak turn-level attack mark: last user/assistant turn carries the objective
        score = ex.get("malicious_score", 10)
        label = 1 if (thresh is None or score >= thresh) else 0
        if label == 1 and turns:
            # mark the final turns as the action-on-objective region
            attack_idx = max(0, len(turns) - 2)
            for t in turns[attack_idx:]:
                t.is_attack = True
        out.append(Record(
            uid=f"agentlab-sess-{ex.get('original_id', i)}", dataset="agentlab",
            span=SPAN_SESSION, turns=turns, label=label,
            attack_turn_idx=attack_idx, n_sessions=1,
            meta={"malicious_score": score,
                  "completion_score": ex.get("completion_score"),
                  "generation_config": ex.get("generation_config", {}),
                  "task_description": ex.get("task_description")}))
    return out


def _load_single_refused(path: str) -> List[Record]:
    data = json.load(open(path))
    out = []
    for i, ex in enumerate(data):
        instr = ex.get("instruction", "")
        turns = [Turn(role=ROLE_USER, content=str(instr), ts=0, session_id=0,
                      is_attack=True)]
        # append the environment payload (e.g. suspicious email) as tool context
        for env in ex.get("environments", []):
            params = env.get("parameters", {})
            turns.append(Turn(role=ROLE_TOOL, content=json.dumps(params, ensure_ascii=False)[:2000],
                              ts=len(turns), session_id=0, is_attack=True,
                              tool_name=env.get("name")))
        meta = ex.get("_baseline_metadata", {})
        # 'fulfillable'==0 means the harmful goal is present -> attack (label 1)
        label = 1 if ex.get("fulfillable", 0) == 0 else 0
        out.append(Record(
            uid=f"agentlab-single-{ex.get('id', i)}", dataset="agentlab",
            span=SPAN_SINGLE, turns=turns, label=label, attack_turn_idx=0,
            n_sessions=1,
            meta={"risks": ex.get("risks"), "evaluation": meta.get("evaluation"),
                  "failure_modes": ex.get("failure_modes")}))
    return out


if __name__ == "__main__":
    import sys
    from collections import Counter
    root = sys.argv[1] if len(sys.argv) > 1 else "datasets/AgentLAB"
    recs = load_agentlab({"agentlab_dir": root})
    print(f"loaded {len(recs)} records")
    print("spans:", Counter(r.span for r in recs))
    print("labels:", Counter(r.label for r in recs))
    r = next(x for x in recs if x.span == SPAN_SESSION)
    print(f"\nexample session {r.uid}: {len(r.turns)} turns, label={r.label}, "
          f"attack_turn_idx={r.attack_turn_idx}, n_tokens={r.n_tokens()}")
