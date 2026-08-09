from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx


TAXON_RANK_QIDS = {
    "class": "Q37517",
    "subclass": "Q5867051",
    "infraclass": "Q2007442",
    "superorder": "Q5868144",
    "order": "Q36602",
    "suborder": "Q5867959",
    "infraorder": "Q2889003",
    "parvorder": "Q6311258",
    "superfamily": "Q2136103",
    "family": "Q35409",
    "subfamily": "Q164280",
    "tribe": "Q227936",
    "subtribe": "Q3965313",
    "genus": "Q34740",
    "subgenus": "Q3238261",
    "species": "Q7432",
    "subspecies": "Q68947",
    "form": "Q279749",
    "ichnogenus": "Q112082101",
}

TAXON_RANK_NAMES = {qid: name for name, qid in TAXON_RANK_QIDS.items()}


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


def _resolve_target_rank(target_rank: str) -> tuple[str | None, str | None]:
    normalized = _normalize_rank_name(target_rank)
    if normalized in TAXON_RANK_QIDS:
        return normalized, TAXON_RANK_QIDS[normalized]
    if normalized in TAXON_RANK_NAMES:
        return TAXON_RANK_NAMES[normalized], normalized
    return normalized or None, None


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

    normalized_target_name, normalized_target_qid = _resolve_target_rank(target_rank)
    for distance, ancestor_qid in enumerate(iter_ancestor_chain(graph, qid), start=1):
        node = graph.nodes[ancestor_qid]
        rank_name = _normalize_rank_name(str(node.get("taxon_rank_name") or ""))
        rank_qid = str(node.get("taxon_rank") or "").strip()
        if normalized_target_qid is not None:
            if rank_qid != normalized_target_qid and rank_name != normalized_target_name:
                continue
        elif rank_name != normalized_target_name:
            continue
        label_name = str(node.get("label_en") or node.get("en_name") or ancestor_qid).strip() or ancestor_qid
        return TaxonLabelAssignment(
            qid=qid,
            target_rank=normalized_target_name or _normalize_rank_name(target_rank),
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
