from __future__ import annotations

import argparse
import pickle
from collections import deque
from pathlib import Path

import networkx as nx

from .config import get_project_paths


def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load a pickled NetworkX graph. / pickle 化された NetworkX グラフを読む。"""

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(f"Graph PKL must contain a networkx.DiGraph, got: {type(graph).__name__}")
    return graph


def collect_descendants(graph: nx.DiGraph, root_qid: str, max_depth: int, max_nodes: int) -> list[str]:
    """Collect descendants in BFS order for a bounded visualization. / 可視化用に深さ制限付きで子孫ノードを集める。"""

    if root_qid not in graph:
        raise KeyError(f"QID not found in graph: {root_qid}")

    ordered: list[str] = []
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(root_qid, 0)])

    while queue and len(ordered) < max_nodes:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        ordered.append(node)
        if depth >= max_depth:
            continue
        for child in sorted(graph.successors(node)):
            if child not in visited:
                queue.append((child, depth + 1))
    return ordered


def make_subgraph(graph: nx.DiGraph, root_qid: str, max_depth: int, max_nodes: int) -> nx.DiGraph:
    """Create the visualization subgraph. / 可視化対象の部分グラフを作る。"""

    nodes = collect_descendants(graph, root_qid=root_qid, max_depth=max_depth, max_nodes=max_nodes)
    return graph.subgraph(nodes).copy()


def summarize_graph(
    graph: nx.DiGraph,
    root_qid: str,
    max_depth: int,
    max_nodes: int,
) -> str:
    """Summarize the bounded subgraph without writing an image. / 画像を書かずに部分グラフの要約を返す。"""

    subgraph = make_subgraph(graph, root_qid=root_qid, max_depth=max_depth, max_nodes=max_nodes)
    rank_counts: dict[str, int] = {}
    for _, data in subgraph.nodes(data=True):
        rank_name = str(data.get("taxon_rank_name") or "").strip().lower() or "other"
        rank_counts[rank_name] = rank_counts.get(rank_name, 0) + 1

    parts = [f"Root QID: {root_qid}", f"Nodes: {subgraph.number_of_nodes()}", f"Edges: {subgraph.number_of_edges()}"]
    if rank_counts:
        rank_summary = ", ".join(f"{rank}={count}" for rank, count in sorted(rank_counts.items()))
        parts.append(f"Ranks: {rank_summary}")
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for graph summary output. / graph 要約出力用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Summarize a bounded Bird taxonomy subgraph from a NetworkX graph PKL.")
    parser.add_argument("--input", default=str(paths.taxonomy_graph_pkl))
    parser.add_argument("--root-qid", default="Q5113")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=150)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the graph summary command. / graph 要約コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    print(summarize_graph(load_graph(Path(args.input)), root_qid=args.root_qid, max_depth=args.max_depth, max_nodes=args.max_nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
