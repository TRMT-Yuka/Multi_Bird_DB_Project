from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_CACHE_ROOT = PROJECT_ROOT / "temp" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_ROOT / "xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp1_similarity_matrices import (  # noqa: E402
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_SELECTED_RUNS_PATH,
    build_qid_vector_maps,
    cosine_similarity_matrix,
    discover_all_runs,
    load_selected_runs,
    select_complete_qids,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp2" / "integrated_embedding_graphs"
DEFAULT_GRAPH_LABEL = "node2vec"
DEFAULT_LANGUAGE_LABEL = "ja"
DEFAULT_AUDIO_LABEL = "perch_07300906"
DEFAULT_TAXONOMY_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "graph" / "bird_taxonomy_graph.pkl"

MODALITY_STYLES = {
    "G": {"color": "#1f77b4", "title": "Graph"},
    "L": {"color": "#ff7f0e", "title": "Language"},
    "A": {"color": "#2ca02c", "title": "Audio"},
    "U": {"color": "#7f7f7f", "title": "Integrated"},
}
def _load_english_labels(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(f"Expected a DiGraph in {path}, got {type(graph).__name__}")
    labels: dict[str, str] = {}
    for qid, data in graph.nodes(data=True):
        label = str(data.get("label_en") or data.get("en_name") or qid).strip()
        labels[str(qid)] = label or str(qid)
    return labels


def _select_shared_qids(common_qids_path: Path, num_recordings_path: Path, top_k: int, qid_maps: dict[str, dict[str, np.ndarray]]) -> list[str]:
    common_qids = _read_qids(common_qids_path)
    num_recordings = _load_num_recordings(num_recordings_path)
    selected: list[tuple[int, str]] = []
    shared_qids = set.intersection(*(set(qid_map) for qid_map in qid_maps.values())) if qid_maps else set()
    for qid in common_qids:
        if qid not in shared_qids:
            continue
        selected.append((num_recordings.get(qid, 0), qid))
    selected.sort(key=lambda item: (-item[0], item[1]))
    return [qid for _, qid in selected[:top_k]]


def _resolve_run(runs: list, modality: str, label: str):
    target_name = f"{modality}__{label}"
    for run in runs:
        if run.name == target_name:
            return run
    raise FileNotFoundError(f"Selected run not found: {target_name}")


def _build_edge_sets(matrix: np.ndarray, qids: list[str]) -> tuple[float, list[tuple[str, str, float]]]:
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    threshold = float(np.quantile(upper, 0.95)) if upper.size else 1.0
    edges: list[tuple[str, str, float]] = []
    for i in range(len(qids)):
        for j in range(i + 1, len(qids)):
            score = float(matrix[i, j])
            if score >= threshold:
                edges.append((qids[i], qids[j], score))
    return threshold, edges


def _edge_weight_quantile(graph: nx.Graph, q: float = 0.95) -> float:
    weights = [float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)]
    if not weights:
        return 1.0
    return float(np.quantile(np.asarray(weights, dtype=np.float32), q))


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value - min_value < 1e-12:
        return [0.5 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def _draw_panel(
    ax,
    graph: nx.Graph,
    pos: dict[str, np.ndarray],
    title: str,
    color: str,
    labels: dict[str, str],
) -> None:
    degrees = dict(graph.degree())
    sizes = [60 + 10 * degrees.get(node, 0) for node in graph.nodes()]
    node_values = [degrees.get(node, 0) for node in graph.nodes()]
    node_colors = plt.cm.Blues(_normalize(node_values)) if node_values else []
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=color, width=1.4, alpha=0.55)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=node_colors if len(node_colors) else "#d9d9d9",
        edgecolors="#222222",
        linewidths=0.5,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels={node: labels.get(node, node) for node in graph.nodes()},
        font_size=4,
        font_color="#111111",
        verticalalignment="center",
        horizontalalignment="center",
        bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )
    ax.set_title(title)
    ax.set_axis_off()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exp2 integrated graph visualization from selected embedding runs.")
    parser.add_argument("--embedding-root", default=str(DEFAULT_EMBEDDING_ROOT))
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument("--taxonomy-graph", default=str(DEFAULT_TAXONOMY_GRAPH_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--graph-label", default=DEFAULT_GRAPH_LABEL)
    parser.add_argument("--language-label", default=DEFAULT_LANGUAGE_LABEL)
    parser.add_argument("--audio-label", default=DEFAULT_AUDIO_LABEL)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--use-selected-runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-discover", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args: argparse.Namespace) -> None:
    embedding_root = Path(args.embedding_root).expanduser()
    if not embedding_root.is_absolute():
        embedding_root = PROJECT_ROOT / embedding_root

    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path

    taxonomy_graph_path = Path(args.taxonomy_graph).expanduser()
    if not taxonomy_graph_path.is_absolute():
        taxonomy_graph_path = PROJECT_ROOT / taxonomy_graph_path

    if args.use_selected_runs and selected_runs_path.exists():
        runs = load_selected_runs(selected_runs_path)
    elif args.auto_discover:
        runs = discover_all_runs(embedding_root)
    else:
        runs = []
    if not runs:
        raise SystemExit("No embedding runs found.")

    selected_runs = [
        _resolve_run(runs, "graph", args.graph_label),
        _resolve_run(runs, "language", args.language_label),
        _resolve_run(runs, "audio", args.audio_label),
    ]

    vector_maps, _ = build_qid_vector_maps(selected_runs)
    qids = select_complete_qids(vector_maps)
    if args.top_k is not None:
        qids = qids[: args.top_k]
    if not qids:
        raise SystemExit("No shared QIDs available for exp2 visualization.")

    english_labels = _load_english_labels(taxonomy_graph_path)

    matrices: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    edge_sets: dict[str, list[tuple[str, str, float]]] = {}
    for run, modality in zip(selected_runs, ["G", "L", "A"], strict=True):
        vectors = np.stack([vector_maps[run.name][qid] for qid in qids], axis=0).astype(np.float32)
        matrix = cosine_similarity_matrix(vectors)
        threshold, edges = _build_edge_sets(matrix, qids)
        matrices[modality] = matrix
        thresholds[modality] = threshold
        edge_sets[modality] = edges

    union_graph = nx.Graph()
    union_graph.add_nodes_from(qids)
    for modality, edges in edge_sets.items():
        for source, target, score in edges:
            if union_graph.has_edge(source, target):
                union_graph[source][target]["modalities"].add(modality)
                union_graph[source][target]["scores"][modality] = score
                union_graph[source][target]["weight"] = max(union_graph[source][target]["weight"], score)
            else:
                union_graph.add_edge(
                    source,
                    target,
                    modalities={modality},
                    scores={modality: score},
                    weight=score,
                )

    union_threshold = _edge_weight_quantile(union_graph, 0.95)
    pos = nx.spring_layout(union_graph, seed=42, weight="weight", k=0.85, iterations=200)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    run_dir = output_dir / f"G_{args.graph_label}__L_{args.language_label}__A_{args.audio_label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for modality in ["G", "L", "A", "U"]:
        if modality == "U":
            graph = union_graph
        else:
            graph = nx.Graph()
            graph.add_nodes_from(qids)
            graph.add_edges_from((u, v, {"weight": score}) for u, v, score in edge_sets[modality])
        fig, ax = plt.subplots(figsize=(16, 14), dpi=180)
        _draw_panel(
            ax,
            graph,
            pos,
            title=(
                f"{MODALITY_STYLES[modality]['title']} (top-{len(qids)}, "
                f"p95={union_threshold:.4f})"
                if modality == "U"
                else f"{MODALITY_STYLES[modality]['title']} (top-{len(qids)}, p95={thresholds.get(modality, 0.0):.4f})"
            ),
            color=MODALITY_STYLES[modality]["color"],
            labels=english_labels,
        )
        fig.suptitle(
            f"EXP2 integrated embedding graph: G={args.graph_label}, L={args.language_label}, A={args.audio_label}",
            fontsize=14,
        )
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.93)
        filename = {
            "G": "graph.svg",
            "L": "language.svg",
            "A": "audio.svg",
            "U": "integrated.svg",
        }[modality]
        fig.savefig(run_dir / filename, bbox_inches="tight", pad_inches=0.4, format="svg")
        plt.close(fig)

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment": "EXP2-integrated-embedding-graphs",
                "graph_label": args.graph_label,
                "language_label": args.language_label,
                "audio_label": args.audio_label,
                "qid_count": len(qids),
                "thresholds": thresholds,
                "union_threshold": union_threshold,
                "edge_counts": {modality: len(edges) for modality, edges in edge_sets.items()},
                "union_edge_count": union_graph.number_of_edges(),
                "selected_runs": [run.name for run in selected_runs],
                "qid_selection_policy": "Same complete-QID intersection and order as exp1_similarity_matrices.py, then capped by top_k.",
                "taxonomy_graph_path": str(taxonomy_graph_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"qid_count": len(qids), "union_edge_count": union_graph.number_of_edges()}, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"output_dir: {run_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
