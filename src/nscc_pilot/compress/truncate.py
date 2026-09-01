"""Fixed-window truncation: keep the most recent turns until the token budget
is spent. The classic, cheapest context-management baseline."""
from __future__ import annotations

from .base import Compressor
from ..schema import Record


class TruncateCompressor(Compressor):
    name = "truncate"

    def compress(self, rec: Record) -> Record:
        budget = max(1, int(round(rec.n_tokens() * self.ratio)))
        kept_rev = []
        used = 0
        # always keep a system turn if present (turn 0)
        sys_turns = [t for t in rec.turns[:1] if t.role == "system"]
        body = rec.turns[len(sys_turns):]
        for t in reversed(body):
            c = len(t.content.split())
            if used + c > budget and kept_rev:
                break
            kept_rev.append(t)
            used += c
        kept = sys_turns + list(reversed(kept_rev))
        # deep-copy turns so ts renumbering doesn't mutate the original
        import copy
        kept = [copy.copy(t) for t in kept]
        return self._clone(rec, kept)
