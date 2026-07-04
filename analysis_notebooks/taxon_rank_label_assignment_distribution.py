from __future__ import annotations

import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "processed" / "graph" / "bird_taxonomy_graph.pkl"
DETAIL_CSV_PATH = ROOT / "analysis_notebooks" / "taxon_rank_label_assignment_detail.csv"
SUMMARY_CSV_PATH = ROOT / "analysis_notebooks" / "taxon_rank_label_assignment_summary.csv"

FALLBACK_TAXON_RANK_NAMES: dict[str, str] = {
    "Q68947": "subspecies",
    "Q7432": "species",
    "Q34740": "genus",
    "Q35409": "family",
    "Q164280": "subfamily",
    "Q227936": "tribe",
    "Q2136103": "superfamily",
    "Q36602": "order",
    "Q37517": "class",
    "Q5867959": "suborder",
    "Q2889003": "infraorder",
    "Q5867051": "subclass",
    "Q6311258": "parvorder",
    "Q3238261": "subgenus",
    "Q5868144": "superorder",
    "Q279749": "form",
    "Q2007442": "infraclass",
    "Q112082101": "ichnogenus",
}

RANK_ORDER = [
    "class",
    "subclass",
    "infraclass",
    "superorder",
    "order",
    "suborder",
    "infraorder",
    "parvorder",
    "superfamily",
    "family",
    "subfamily",
    "tribe",
    "genus",
    "subgenus",
    "species",
    "subspecies",
    "form",
    "ichnogenus",
    "unknown",
]
RANK_ORDER_INDEX = {name: index for index, name in enumerate(RANK_ORDER)}

def load_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def resolve_rank_name(rank_qid: str, graph_rank_name: str) -> str:
    cleaned = str(graph_rank_name or "").strip()
    if cleaned:
        return cleaned
    return FALLBACK_TAXON_RANK_NAMES.get(rank_qid, "unknown")


def discover_leaf_nodes(graph: nx.DiGraph) -> set[str]:
    return {str(node) for node in graph.nodes if graph.out_degree(node) == 0}


def build_detail_rows(graph: nx.DiGraph) -> list[dict[str, str | int]]:
    leaf_nodes = discover_leaf_nodes(graph)
    rows: list[dict[str, str | int]] = []
    for node, data in graph.nodes(data=True):
        rank_qid = str(data.get("taxon_rank") or "").strip() or "unknown"
        rank_name = resolve_rank_name(rank_qid, str(data.get("taxon_rank_name") or ""))
        descendants = nx.descendants(graph, node)
        strict_descendant_count = len(descendants)
        leaf_descendant_count = sum(1 for descendant in descendants if str(descendant) in leaf_nodes)
        is_leaf_label = 1 if str(node) in leaf_nodes else 0

        rows.append(
            {
                "label_rank_qid": rank_qid,
                "label_rank_name": rank_name,
                "label_qid": str(node),
                "label_name": str(data.get("label_en") or data.get("taxon_name") or node),
                "assigned_node_count": strict_descendant_count,
                "assigned_leaf_node_count": leaf_descendant_count,
                "strict_descendant_count": strict_descendant_count,
                "leaf_descendant_count": leaf_descendant_count,
                "is_leaf_label": is_leaf_label,
            }
        )

    rows.sort(
        key=lambda row: (
            RANK_ORDER_INDEX.get(str(row["label_rank_name"]), len(RANK_ORDER_INDEX)),
            -int(row["assigned_node_count"]),
            str(row["label_qid"]),
        )
    )
    return rows


def build_summary_rows(detail_rows: list[dict[str, str | int]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str | int]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["label_rank_qid"]), str(row["label_rank_name"]))].append(row)

    size_key = "assigned_node_count"
    leaf_size_key = "assigned_leaf_node_count"
    zero_key = "zero_assigned_label_count"

    summary_rows: list[dict[str, str]] = []
    for (rank_qid, rank_name), rows in sorted(
        grouped.items(),
        key=lambda item: RANK_ORDER_INDEX.get(item[0][1], len(RANK_ORDER_INDEX)),
    ):
        size_counts = np.asarray([int(row[size_key]) for row in rows], dtype=np.int64)
        leaf_size_counts = np.asarray([int(row[leaf_size_key]) for row in rows], dtype=np.int64)
        zero_value = int(np.sum(size_counts == 0))

        summary_rows.append(
            {
                "label_rank_qid": rank_qid,
                "label_rank_name": rank_name,
                "label_count": str(len(rows)),
                f"{size_key}_mean": f"{float(size_counts.mean()):.2f}",
                f"{size_key}_median": f"{float(np.median(size_counts)):.2f}",
                f"{size_key}_max": str(int(size_counts.max(initial=0))),
                f"{size_key}_min": str(int(size_counts.min(initial=0))),
                f"{leaf_size_key}_mean": f"{float(leaf_size_counts.mean()):.2f}",
                f"{leaf_size_key}_median": f"{float(np.median(leaf_size_counts)):.2f}",
                f"{leaf_size_key}_max": str(int(leaf_size_counts.max(initial=0))),
                zero_key: str(zero_value),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, str | int]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize assignment label-size distributions for taxon-rank nodes in the bird taxonomy graph."
    )
    parser.add_argument("--graph", default=str(GRAPH_PATH))
    parser.add_argument("--detail-output", default=str(DETAIL_CSV_PATH))
    parser.add_argument("--summary-output", default=str(SUMMARY_CSV_PATH))
    args = parser.parse_args()

    graph = load_graph(Path(args.graph))
    detail_rows = build_detail_rows(graph)
    summary_rows = build_summary_rows(detail_rows)

    size_key = "assigned_node_count"
    leaf_size_key = "assigned_leaf_node_count"
    zero_key = "zero_assigned_label_count"

    write_csv(
        Path(args.detail_output),
        detail_rows,
        [
            "label_rank_qid",
            "label_rank_name",
            "label_qid",
            "label_name",
            size_key,
            leaf_size_key,
            "strict_descendant_count",
            "leaf_descendant_count",
            "is_leaf_label",
        ],
    )
    write_csv(
        Path(args.summary_output),
        summary_rows,
        [
            "label_rank_qid",
            "label_rank_name",
            "label_count",
            f"{size_key}_mean",
            f"{size_key}_median",
            f"{size_key}_max",
            f"{size_key}_min",
            f"{leaf_size_key}_mean",
            f"{leaf_size_key}_median",
            f"{leaf_size_key}_max",
            zero_key,
        ],
    )
    print(f"node_count\t{graph.number_of_nodes()}")
    print(f"label_row_count\t{len(detail_rows)}")
    print(f"rank_count\t{len(summary_rows)}")
    print(f"detail_csv_output\t{args.detail_output}")
    print(f"summary_csv_output\t{args.summary_output}")


if __name__ == "__main__":
    main()
