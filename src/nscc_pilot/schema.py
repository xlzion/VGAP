"""Normalized record schema for the NSCC pilot.

Every dataset (CSTM-Bench, AgentLAB, or the synthetic fixtures) is mapped by an
adapter into a list of `Record`s that share this schema, so the compression,
detection, and metric code is dataset-agnostic.

The single most important field is `span` -- the *attack time-span* facet that
Gate A slices every metric by (single / session / cross_session).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# Attack time-span layers (the Gate A facet).
SPAN_SINGLE = "single"          # payload lands within one step / one message
SPAN_SESSION = "session"        # payload spread across turns of one session
SPAN_CROSS = "cross_session"    # payload spread across multiple sessions
SPANS = (SPAN_SINGLE, SPAN_SESSION, SPAN_CROSS)

# Roles in a normalized trajectory.
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


@dataclass
class Turn:
    role: str
    content: str
    ts: int                       # logical time / turn index within the record
    session_id: int = 0           # which session this turn belongs to
    is_attack: bool = False       # ground truth: this turn carries attack payload
    tool_name: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Record:
    uid: str
    dataset: str                  # 'cstm' | 'agentlab' | 'fixture'
    span: str                     # one of SPANS
    turns: List[Turn]
    label: int                    # 1 = attack present, 0 = benign
    attack_turn_idx: Optional[int] = None   # first attack turn (for time-to-detect)
    n_sessions: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience ----
    def flat_text(self, sep: str = "\n") -> str:
        """Full context flattened to a single string (role-tagged)."""
        return sep.join(f"[{t.role}] {t.content}" for t in self.turns)

    def n_tokens(self) -> int:
        """Cheap whitespace token count of the full context."""
        return sum(len(t.content.split()) for t in self.turns)


# --------------------------------------------------------------------------- #
# IO helpers -- records are persisted as JSONL so adapters can be run offline
# once and reused across every Gate A/B experiment.
# --------------------------------------------------------------------------- #
def record_to_dict(r: Record) -> Dict[str, Any]:
    d = asdict(r)
    return d


def record_from_dict(d: Dict[str, Any]) -> Record:
    turns = [Turn(**t) for t in d.pop("turns")]
    return Record(turns=turns, **d)


def write_jsonl(records: List[Record], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(record_to_dict(r), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Record]:
    out: List[Record] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(record_from_dict(json.loads(line)))
    return out
