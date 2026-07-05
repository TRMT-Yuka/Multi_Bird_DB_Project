from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class QidSplit:
    """QID-level split assignment used before multimodal pattern expansion."""

    train_qids: list[str]
    validation_qids: list[str]
    test_qids: list[str]


def split_qids(
    qids: list[str],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> QidSplit:
    """Split QIDs deterministically before expanding intra-QID modality combinations."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be < 1.0")

    unique_qids = sorted({str(qid).strip() for qid in qids if str(qid).strip()})
    rng = np.random.default_rng(seed)
    if unique_qids:
        order = rng.permutation(len(unique_qids)).tolist()
        shuffled = [unique_qids[index] for index in order]
    else:
        shuffled = []

    qid_count = len(shuffled)
    test_count = int(round(qid_count * test_fraction))
    validation_count = int(round(qid_count * validation_fraction))
    if test_count + validation_count > qid_count:
        overflow = test_count + validation_count - qid_count
        validation_count = max(0, validation_count - overflow)

    test_qids = sorted(shuffled[:test_count])
    validation_qids = sorted(shuffled[test_count : test_count + validation_count])
    train_qids = sorted(shuffled[test_count + validation_count :])
    return QidSplit(train_qids=train_qids, validation_qids=validation_qids, test_qids=test_qids)


def qid_split_lookup(split: QidSplit) -> dict[str, str]:
    """Return a QID -> split-name lookup map."""

    lookup: dict[str, str] = {}
    for qid in split.train_qids:
        lookup[qid] = 'train'
    for qid in split.validation_qids:
        lookup[qid] = 'validation'
    for qid in split.test_qids:
        lookup[qid] = 'test'
    return lookup
