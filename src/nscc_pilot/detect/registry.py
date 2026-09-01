from __future__ import annotations

from .llm_judge import LLMJudgeDetector
from .seq_anomaly import SeqAnomalyDetector
from .rule_graph import RuleGraphDetector

DETECTORS = {
    LLMJudgeDetector.name: LLMJudgeDetector,     # 'llm_judge'
    SeqAnomalyDetector.name: SeqAnomalyDetector,  # 'seq_anomaly'
    RuleGraphDetector.name: RuleGraphDetector,    # 'rule_graph'
}

# detectors that require an LLM backend (role) so the orchestrator loads one
NEEDS_BACKEND = {"llm_judge": "judge"}


def build_detector(name: str, backend=None, cfg: dict | None = None):
    if name not in DETECTORS:
        raise KeyError(f"unknown detector {name!r}; have {list(DETECTORS)}")
    return DETECTORS[name](backend=backend, cfg=cfg)
