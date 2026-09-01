"""AgentDojo adapter -- second dataset (single-session tool-injection) for the
activation-probe external-validity test.

Source: github ethz-spylab/agentdojo, `runs/<model>/<suite>/<user_task>/
<injection_task or none>/<attack_type>.json`. Each transcript:
  injection_task_id (null=benign), injections{name: text}, attack_type, security
  (did the injection succeed), messages[{role, content:[{type,content}]}].

Mapping:
  * label = 1 if the transcript has injections, else 0 (benign).
  * span  = 'single' (single-session tool-injection threat model).
  * per-turn is_attack = a message whose (whitespace-normalized) text contains a
    distinctive slice of any injection payload -> the tool turn carrying the
    injection. Robust to escaping/formatting (exact match fails).
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import List

from ..schema import (Record, Turn, SPAN_SINGLE,
                      ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL, ROLE_SYSTEM)

_ROLE = {"system": ROLE_SYSTEM, "user": ROLE_USER, "assistant": ROLE_ASSISTANT,
         "tool": ROLE_TOOL, "function": ROLE_TOOL}
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def _msg_text(content) -> str:
    if isinstance(content, list):
        return " ".join(_norm(x.get("content", "")) for x in content
                        if isinstance(x, dict) and x.get("content"))
    return _norm(str(content or ""))


def _inj_cores(injections: dict) -> List[str]:
    """Distinctive normalized slices of each injection payload for matching."""
    cores = []
    for v in (injections or {}).values():
        n = _norm(v)
        if len(n) >= 30:
            cores.append(n[:60])       # leading distinctive chunk
    return cores


def load_agentdojo(cfg: dict) -> List[Record]:
    root = cfg.get("agentdojo_dir", "../datasets/agentdojo")
    model = cfg.get("agentdojo_model", "gpt-4o-2024-05-13")
    suites = cfg.get("agentdojo_suites", ["workspace", "banking", "slack", "travel"])
    base = os.path.join(root, "runs", model)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"AgentDojo runs not found at {base}")

    files = []
    for s in suites:
        files += glob.glob(os.path.join(base, s, "**", "*.json"), recursive=True)
    files = sorted(files)
    recs: List[Record] = []
    for path in files:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        r = _to_record(d, os.path.relpath(path, base))
        if r is not None:
            recs.append(r)

    # optional balancing/subsample: keep all benign, subsample attack transcripts
    # (preferring those with a matched injected turn) for a balanced probe set.
    max_atk = cfg.get("agentdojo_max_attack")
    if max_atk is not None:
        import random
        rng = random.Random(cfg.get("seed", 0))
        benign = [r for r in recs if r.label == 0]
        atk = [r for r in recs if r.label == 1]
        atk.sort(key=lambda r: r.attack_turn_idx is None)  # matched-turn first
        matched = [r for r in atk if r.attack_turn_idx is not None]
        rng.shuffle(matched)
        recs = benign + matched[:max_atk]
        rng.shuffle(recs)
    return recs


def _to_record(d: dict, uid: str) -> Record:
    injections = d.get("injections") or {}
    label = 1 if injections else 0
    cores = _inj_cores(injections)
    turns, attack_idx = [], None
    for j, m in enumerate(d.get("messages", [])):
        role = _ROLE.get(str(m.get("role", "user")).lower(), ROLE_USER)
        txt = _msg_text(m.get("content"))
        is_atk = bool(cores) and any(c in txt for c in cores)
        if is_atk and attack_idx is None:
            attack_idx = len(turns)
        turns.append(Turn(role=role, content=txt, ts=j, session_id=0, is_attack=is_atk))
    if not turns:
        return None
    return Record(uid=f"adojo-{uid}", dataset="agentdojo", span=SPAN_SINGLE,
                  turns=turns, label=label, attack_turn_idx=attack_idx, n_sessions=1,
                  meta={"suite": d.get("suite_name"), "attack_type": d.get("attack_type"),
                        "security": d.get("security"), "utility": d.get("utility"),
                        "injection_task_id": d.get("injection_task_id")})


if __name__ == "__main__":
    import sys
    from collections import Counter
    cfg = {"agentdojo_dir": sys.argv[1] if len(sys.argv) > 1 else "../datasets/agentdojo"}
    recs = load_agentdojo(cfg)
    print(f"loaded {len(recs)} transcripts")
    print("label:", Counter(r.label for r in recs))
    atk = [r for r in recs if r.label == 1]
    n_with_atkturn = sum(1 for r in atk if r.attack_turn_idx is not None)
    print(f"attack transcripts: {len(atk)}, of which {n_with_atkturn} have a matched attack turn "
          f"({100*n_with_atkturn/max(1,len(atk)):.0f}%)")
    tt = sum(len([t for t in r.turns if t.is_attack]) for r in recs)
    tot = sum(len(r.turns) for r in recs)
    print(f"attack turns: {tt} / {tot} total turns")
