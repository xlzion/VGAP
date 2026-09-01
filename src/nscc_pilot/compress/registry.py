from __future__ import annotations

from .truncate import TruncateCompressor
from .llm_summary import LLMSummaryCompressor
from .perplexity_prune import PerplexityPruneCompressor

COMPRESSORS = {
    TruncateCompressor.name: TruncateCompressor,       # 'truncate'
    LLMSummaryCompressor.name: LLMSummaryCompressor,    # 'llm_summary'
    PerplexityPruneCompressor.name: PerplexityPruneCompressor,  # 'ppl_prune'
}

# which compressors need an LLM backend (so the orchestrator loads one lazily)
NEEDS_BACKEND = {"llm_summary": "small", "ppl_prune": "small"}


def build_compressor(name: str, ratio: float, backend=None, cfg: dict | None = None):
    if name not in COMPRESSORS:
        raise KeyError(f"unknown compressor {name!r}; have {list(COMPRESSORS)}")
    return COMPRESSORS[name](ratio=ratio, backend=backend, cfg=cfg)
