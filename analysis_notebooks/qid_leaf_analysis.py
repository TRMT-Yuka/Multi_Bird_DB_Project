from __future__ import annotations

import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "data" / "processed" / "bird_ontology.pkl"
GRAPH_PATH = ROOT / "data" / "processed" / "graph" / "bird_taxonomy_graph.pkl"


def load_ontology_by_qid(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)
    return {str(row.get("qid") or row.get("id") or ""): row for row in rows}


def load_graph(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def count_leaf_qids(graph) -> list[str]:
    return [node for node in graph.nodes if graph.out_degree(node) == 0]


def main() -> None:
    ontology_by_qid = load_ontology_by_qid(ONTOLOGY_PATH)
    graph = load_graph(GRAPH_PATH)
    leaf_qids = count_leaf_qids(graph)
    leaf_with_xeno_canto = [
        qid
        for qid in leaf_qids
        if str(ontology_by_qid.get(qid, {}).get("xeno_canto_species_id") or "").strip()
    ]

    print(f"leaf_qid_count\t{len(leaf_qids)}")
    print(f"leaf_qid_with_xeno_canto_species_id_count\t{len(leaf_with_xeno_canto)}")


if __name__ == "__main__":
    main()
