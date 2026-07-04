from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "data" / "processed" / "graph" / "bird_taxonomy_graph.pkl"
DEFAULT_OUTPUT_PATH = ROOT / "analysis_notebooks" / "taxon_rank_distribution.csv"

# Fallback labels for the taxon-rank QIDs currently present in this graph snapshot.
# The ontology currently carries empty taxon_rank_name fields, so the analysis uses
# these readable labels until the upstream ontology/graph build stores them directly.
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
FIELDNAMES = ["taxon_rank_qid", "taxon_rank_name", "name_source", "node_count"]


def load_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def resolve_rank_name(rank_qid: str, graph_rank_name: str) -> tuple[str, str]:
    cleaned = str(graph_rank_name or "").strip()
    if cleaned:
        return cleaned, "graph"
    fallback = FALLBACK_TAXON_RANK_NAMES.get(rank_qid, "")
    if fallback:
        return fallback, "fallback_wikidata"
    return "unknown", "unknown"


def build_rank_counter(graph) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for _, data in graph.nodes(data=True):
        rank_qid = str(data.get("taxon_rank") or "").strip() or "unknown"
        rank_name, name_source = resolve_rank_name(rank_qid, str(data.get("taxon_rank_name") or ""))
        counts[(rank_qid, rank_name, name_source)] += 1
    return counts


def make_rows(counter: Counter[tuple[str, str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (rank_qid, rank_name, name_source), node_count in sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        rows.append(
            {
                "taxon_rank_qid": rank_qid,
                "taxon_rank_name": rank_name,
                "name_source": name_source,
                "node_count": str(node_count),
            }
        )
    return rows


def write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count graph nodes by taxon rank QID and taxon rank name.")
    parser.add_argument("--graph", default=str(GRAPH_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    graph = load_graph(Path(args.graph))
    counter = build_rank_counter(graph)
    rows = make_rows(counter)
    write_csv(Path(args.output), rows)

    print(f"node_count\t{graph.number_of_nodes()}")
    print(f"distinct_taxon_rank_pairs\t{len(rows)}")
    print(f"output_csv\t{args.output}")
    print("taxon_rank_qid\ttaxon_rank_name\tname_source\tnode_count")
    for row in rows:
        print(
            f"{row['taxon_rank_qid']}\t{row['taxon_rank_name']}\t{row['name_source']}\t{row['node_count']}"
        )
    print(f"wrote_csv\t{args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
