"""Dataset adapters: map each raw dataset into List[Record].

Until CSTM-Bench / AgentLAB finish downloading, `load_dataset('fixture')`
returns synthetic records so the pipeline runs end-to-end. Once the data is in
place, fill the TWO TODO blocks in cstm_bench.py / agentlab.py (field names
only) and switch the config `dataset:` key. Nothing else changes.
"""
from __future__ import annotations

from typing import List

from ..schema import Record
from ..fixtures import make_fixture


def load_dataset(name: str, cfg: dict | None = None) -> List[Record]:
    cfg = cfg or {}
    if name == "fixture":
        return make_fixture(n_per_cell=cfg.get("fixture_n", 60),
                            seed=cfg.get("seed", 0))
    if name == "cstm":
        from .cstm_bench import load_cstm
        return load_cstm(cfg)
    if name == "agentlab":
        from .agentlab import load_agentlab
        return load_agentlab(cfg)
    if name == "agentdojo":
        from .agentdojo import load_agentdojo
        return load_agentdojo(cfg)
    if name == "injecagent":
        from .injecagent import load_injecagent
        return load_injecagent(cfg)
    if name == "a3s":
        from .a3s import load_a3s
        return load_a3s(cfg)
    if name == "jsonl":
        from ..schema import read_jsonl
        return read_jsonl(cfg["jsonl_path"])
    if name == "combined":
        # merge all tiers so the Gate-A span facet (single/session/cross_session)
        # is populated: CSTM -> cross_session, AgentLAB -> session + single.
        # NOTE: the session tier from AgentLAB's static release is attack-heavy;
        # spans without benign negatives are dropped from recall@FPR (see the
        # AgentLAB adapter caveat). Curate benign session trajectories to close it.
        from .cstm_bench import load_cstm
        from .agentlab import load_agentlab
        return load_cstm(cfg) + load_agentlab(cfg)
    if name == "paired_gate_a":
        from .paired_gate_a import load_paired_gate_a
        return load_paired_gate_a(cfg)
    raise KeyError(f"unknown dataset {name!r}")
