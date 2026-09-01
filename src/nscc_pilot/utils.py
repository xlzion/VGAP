"""Config loading, stratified splitting, small IO helpers."""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from .schema import Record


def load_config(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
            return yaml.safe_load(text)
        except Exception:
            pass
    return json.loads(text)


def stratified_split(records: List[Record], test_frac: float = 0.5,
                     seed: int = 0) -> Tuple[List[Record], List[Record]]:
    """Split stratified by (span, label) so every cell is represented in both
    train and test."""
    rng = random.Random(seed)
    buckets: Dict[tuple, List[Record]] = defaultdict(list)
    for r in records:
        buckets[(r.span, r.label)].append(r)
    train, test = [], []
    for _, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_frac)))
        test.extend(items[:n_test])
        train.extend(items[n_test:] or items[:1])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def grouped_stratified_split(records: List[Record], test_frac: float = 0.5,
                             seed: int = 0, group_key: str = "pair_id"):
    """Keep paired span variants together while stratifying by group label."""
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record.meta.get(group_key) or record.uid)].append(record)
    buckets = defaultdict(list)
    for key, members in grouped.items():
        labels = tuple(sorted({int(record.label) for record in members}))
        buckets[labels].append(key)
    rng = random.Random(seed)
    test_groups = set()
    for keys in buckets.values():
        keys = list(keys)
        rng.shuffle(keys)
        n_test = max(1, int(round(len(keys) * test_frac)))
        if len(keys) > 1:
            n_test = min(len(keys) - 1, n_test)
        test_groups.update(keys[:n_test])
    train = [r for r in records if str(r.meta.get(group_key) or r.uid) not in test_groups]
    test = [r for r in records if str(r.meta.get(group_key) or r.uid) in test_groups]
    return train, test


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows: List[dict], path: str) -> None:
    import csv
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or ".")
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
