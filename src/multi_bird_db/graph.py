from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import networkx as nx

from .config import get_project_paths


def load_ontology(ontology_path: Path) -> list[dict[str, Any]]:
    """Load ontology PKL rows. / ontology PKL の行一覧を読む。"""

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file does not exist: {ontology_path}")
    with ontology_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Ontology PKL must contain a list, got: {type(data).__name__}")
    return data


def build_node_attributes(row: dict[str, Any], qid: str) -> dict[str, Any]:
    """Select node attributes used by the graph. / graph に載せるノード属性を整える。"""

    label_en = str(row.get("en_name") or row.get("taxon_name") or qid).strip() or qid
    label_ja = str(row.get("ja_name") or "").strip()
    return {
        "qid": qid,
        "label_en": label_en,
        "label_ja": label_ja,
        "en_name": str(row.get("en_name") or "").strip(),
        "ja_name": label_ja,
        "taxon_name": str(row.get("taxon_name") or "").strip(),
        "taxon_rank": str(row.get("taxon_rank") or "").strip(),
        "taxon_rank_name": str(row.get("taxon_rank_name") or "").strip(),
        "parent_taxon": str(row.get("parent_taxon") or "").strip(),
        "entity_url": str(row.get("entity_url") or "").strip(),
        "enwiki_url": str(row.get("enwiki_url") or "").strip(),
        "jawiki_url": str(row.get("jawiki_url") or "").strip(),
    }


def build_taxonomy_graph(
    ontology_rows: list[dict[str, Any]],
    root_qid: str = "Q5113",
) -> nx.DiGraph:
    """Build a NetworkX directed graph from ontology rows. / ontology 行から NetworkX の有向グラフを作る。"""

    graph = nx.DiGraph()
    graph.graph["graph_type"] = "taxonomy_digraph"
    graph.graph["root_qid"] = root_qid

    for row in ontology_rows:
        qid = str(row.get("qid") or row.get("id") or "").strip()
        if not qid:
            continue
        graph.add_node(qid, **build_node_attributes(row, qid))

    for row in ontology_rows:
        child_qid = str(row.get("qid") or row.get("id") or "").strip()
        parent_qid = str(row.get("parent_taxon") or "").strip()
        if not child_qid or not parent_qid:
            continue
        if parent_qid not in graph:
            graph.add_node(
                parent_qid,
                qid=parent_qid,
                label_en=parent_qid,
                label_ja="",
                en_name="",
                ja_name="",
                taxon_name="",
                taxon_rank="",
                taxon_rank_name="",
                parent_taxon="",
                entity_url="",
                enwiki_url="",
                jawiki_url="",
            )
        graph.add_edge(parent_qid, child_qid, relation="parent_taxon")

    graph.graph["node_count"] = graph.number_of_nodes()
    graph.graph["edge_count"] = graph.number_of_edges()
    return graph


def build_graph(ontology_path: Path, output_path: Path, root_qid: str = "Q5113") -> None:
    """Create a NetworkX taxonomy graph PKL from bird_ontology.pkl. / bird_ontology.pkl から NetworkX の graph PKL を作る。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_taxonomy_graph(load_ontology(ontology_path), root_qid=root_qid)
    with output_path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for graph generation. / graph 生成コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build Bird taxonomy NetworkX graph PKL from ontology PKL.")
    parser.add_argument("--input", default=str(paths.ontology_pkl))
    parser.add_argument("--output", default=str(paths.taxonomy_graph_pkl))
    parser.add_argument("--root-qid", default="Q5113")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the graph generation command. / graph 生成コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    build_graph(Path(args.input), Path(args.output), root_qid=args.root_qid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
