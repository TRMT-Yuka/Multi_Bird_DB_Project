from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass(frozen=True, slots=True)
class TaxonLabelAssignment:
    """Resolved upper-taxon label for one QID and one target rank."""

    qid: str
    target_rank: str
    label_qid: str
    label_name: str
    distance_to_label: int


def _normalize_rank_name(rank_name: str) -> str:
    return str(rank_name or "").strip().lower().replace("_", " ")


def iter_ancestor_chain(graph: nx.DiGraph, qid: str) -> list[str]:
    """Return ancestors by repeatedly following parent_taxon pointers."""

    if qid not in graph:
        raise KeyError(f"QID not found in taxonomy graph: {qid}")
    chain: list[str] = []
    seen = {qid}
    current = qid
    while True:
        parent_qid = str(graph.nodes[current].get("parent_taxon") or "").strip()
        if not parent_qid:
            break
        if parent_qid not in graph or parent_qid in seen:
            break
        chain.append(parent_qid)
        seen.add(parent_qid)
        current = parent_qid
    return chain


def assign_upper_taxon_label(graph: nx.DiGraph, qid: str, target_rank: str) -> TaxonLabelAssignment | None:
    """Resolve the nearest ancestor whose taxon_rank_name matches target_rank."""

    normalized_target = _normalize_rank_name(target_rank)
    for distance, ancestor_qid in enumerate(iter_ancestor_chain(graph, qid), start=1):
        node = graph.nodes[ancestor_qid]
        rank_name = _normalize_rank_name(str(node.get("taxon_rank_name") or ""))
        if rank_name != normalized_target:
            continue
        label_name = str(node.get("label_en") or node.get("en_name") or ancestor_qid).strip() or ancestor_qid
        return TaxonLabelAssignment(
            qid=qid,
            target_rank=normalized_target,
            label_qid=ancestor_qid,
            label_name=label_name,
            distance_to_label=distance,
        )
    return None


def assign_labels_for_qids(graph: nx.DiGraph, qids: list[str], target_rank: str) -> list[TaxonLabelAssignment]:
    """Resolve one upper-taxon label per QID when possible."""

    assignments: list[TaxonLabelAssignment] = []
    for qid in qids:
        assignment = assign_upper_taxon_label(graph, qid, target_rank)
        if assignment is not None:
            assignments.append(assignment)
    return assignments
