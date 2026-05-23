from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx
import numpy as np

from .config import get_project_paths
from .embeddings import EmbeddingStore, load_embedding_store, load_graph


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(slots=True)
class ClusteringResult:
    method: str
    n_samples: int
    n_clusters: int
    inertia: float
    nmi: float
    ari: float
    purity: float
    homogeneity: float
    completeness: float
    v_measure: float
    silhouette: float
    labels: list[str]
    predicted: list[int]


def _safe_label(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "other"


def _resolve_labels(graph: nx.DiGraph, qids: list[str], label_field: str) -> list[str]:
    labels: list[str] = []
    fallback_fields = [label_field, "taxon_rank_name", "taxon_rank", "label_en", "en_name"]
    for qid in qids:
        node_data = graph.nodes[qid] if qid in graph else {}
        resolved = "other"
        for field in fallback_fields:
            value = _safe_label(node_data.get(field))
            if value != "other" or field == fallback_fields[-1]:
                resolved = value
                break
        labels.append(resolved)
    return labels


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def _kmeans_plus_plus_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = data.shape[0]
    centers = np.empty((k, data.shape[1]), dtype=np.float64)
    first_index = int(rng.integers(0, n_samples))
    centers[0] = data[first_index]
    closest_dist_sq = np.sum((data - centers[0]) ** 2, axis=1)
    for center_index in range(1, k):
        total = float(np.sum(closest_dist_sq))
        if total <= 0:
            centers[center_index] = data[int(rng.integers(0, n_samples))]
            continue
        probs = closest_dist_sq / total
        next_index = int(rng.choice(n_samples, p=probs))
        centers[center_index] = data[next_index]
        dist_sq = np.sum((data - centers[center_index]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
    return centers


def _kmeans(
    data: np.ndarray,
    k: int,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_inertia = float("inf")
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None

    for _ in range(max(n_init, 1)):
        centers = _kmeans_plus_plus_init(data, k, rng)
        labels = np.zeros(data.shape[0], dtype=np.int64)
        for _ in range(max_iter):
            distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster_index in range(k):
                members = data[new_labels == cluster_index]
                if len(members) == 0:
                    new_centers[cluster_index] = data[int(rng.integers(0, data.shape[0]))]
                    continue
                new_centers[cluster_index] = members.mean(axis=0)
            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            labels = new_labels
            if shift <= tol:
                break
        inertia = float(np.sum((data - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise RuntimeError("K-means failed to converge")
    return best_labels, best_centers, best_inertia


def _comb2(x: int) -> float:
    return float(x * (x - 1) / 2)


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        value -= p * math.log(p)
    return value


def _mutual_information(contingency: np.ndarray) -> float:
    total = float(contingency.sum())
    if total <= 0:
        return 0.0
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            count = contingency[i, j]
            if count <= 0:
                continue
            mi += (count / total) * math.log((count * total) / (row_sums[i] * col_sums[j]))
    return mi


def _contingency_matrix(true_labels: list[str], pred_labels: np.ndarray) -> tuple[np.ndarray, list[str], list[int]]:
    true_index = {label: idx for idx, label in enumerate(sorted(set(true_labels)))}
    pred_index = {label: idx for idx, label in enumerate(sorted(set(int(v) for v in pred_labels.tolist())))}
    matrix = np.zeros((len(true_index), len(pred_index)), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels.tolist()):
        matrix[true_index[true_label], pred_index[int(pred_label)]] += 1
    return matrix, list(true_index.keys()), list(pred_index.keys())


def _nmi(contingency: np.ndarray) -> float:
    mi = _mutual_information(contingency)
    true_counts = contingency.sum(axis=1).tolist()
    pred_counts = contingency.sum(axis=0).tolist()
    h_true = _entropy(true_counts)
    h_pred = _entropy(pred_counts)
    if h_true == 0.0 and h_pred == 0.0:
        return 1.0
    if h_true == 0.0 or h_pred == 0.0:
        return 0.0
    return mi / math.sqrt(h_true * h_pred)


def _homogeneity_completeness_v_measure(contingency: np.ndarray) -> tuple[float, float, float]:
    mi = _mutual_information(contingency)
    true_counts = contingency.sum(axis=1).tolist()
    pred_counts = contingency.sum(axis=0).tolist()
    h_true = _entropy(true_counts)
    h_pred = _entropy(pred_counts)
    homogeneity = 1.0 if h_true == 0.0 else mi / h_true
    completeness = 1.0 if h_pred == 0.0 else mi / h_pred
    if homogeneity + completeness == 0.0:
        v_measure = 0.0
    else:
        v_measure = 2.0 * homogeneity * completeness / (homogeneity + completeness)
    return homogeneity, completeness, v_measure


def _purity(contingency: np.ndarray) -> float:
    total = float(contingency.sum())
    if total == 0:
        return 0.0
    return float(np.sum(np.max(contingency, axis=0)) / total)


def _ari(contingency: np.ndarray) -> float:
    n = int(contingency.sum())
    if n <= 1:
        return 1.0
    sum_comb = float(np.sum([_comb2(int(x)) for x in contingency.flatten()]))
    row_comb = float(np.sum([_comb2(int(x)) for x in contingency.sum(axis=1)]))
    col_comb = float(np.sum([_comb2(int(x)) for x in contingency.sum(axis=0)]))
    total_comb = _comb2(n)
    if total_comb == 0:
        return 0.0
    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denominator = max_index - expected
    if denominator == 0.0:
        return 1.0 if sum_comb == expected else 0.0
    return (sum_comb - expected) / denominator


def _silhouette_score(data: np.ndarray, labels: np.ndarray, sample_size: int = 2000, seed: int = 42) -> float:
    n = data.shape[0]
    if n <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    if n > sample_size:
        indices = rng.choice(n, size=sample_size, replace=False)
        data = data[indices]
        labels = labels[indices]
        n = data.shape[0]

    distances = np.sqrt(np.maximum(
        np.sum(data[:, None, :] ** 2, axis=2)
        + np.sum(data[None, :, :] ** 2, axis=2)
        - 2.0 * np.matmul(data, data.T),
        0.0,
    ))

    silhouettes: list[float] = []
    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    for i in range(n):
        label_i = int(labels[i])
        same_mask = labels == label_i
        same_count = int(np.sum(same_mask))
        if same_count <= 1:
            a_i = 0.0
        else:
            a_i = float(np.sum(distances[i, same_mask]) / max(same_count - 1, 1))

        b_i = float("inf")
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_mask = labels == other_label
            if not np.any(other_mask):
                continue
            b_i = min(b_i, float(np.mean(distances[i, other_mask])))
        if b_i == float("inf"):
            silhouettes.append(0.0)
            continue
        denom = max(a_i, b_i)
        silhouettes.append(0.0 if denom == 0.0 else (b_i - a_i) / denom)
    return float(np.mean(silhouettes)) if silhouettes else 0.0


def evaluate_embedding_store(
    store: EmbeddingStore,
    graph: nx.DiGraph,
    label_field: str,
    seed: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    qids = [qid for qid in store.qids if qid in graph]
    if not qids:
        raise ValueError("No overlapping QIDs between embeddings and graph")
    indices = [store.qid_to_index[qid] for qid in qids]
    embeddings = np.asarray(store.embeddings[indices], dtype=np.float64)
    embeddings = _l2_normalize(embeddings)
    labels = _resolve_labels(graph, qids, label_field)
    n_clusters = len(set(labels))
    if n_clusters < 2:
        raise ValueError("Need at least two unique labels for clustering evaluation")
    predicted, _, inertia = _kmeans(embeddings, n_clusters, seed=seed, n_init=10, max_iter=100)
    contingency, _, _ = _contingency_matrix(labels, predicted)
    homogeneity, completeness, v_measure = _homogeneity_completeness_v_measure(contingency)
    result = {
        "method": store.metadata.get("algorithm") or "unknown",
        "n_samples": len(labels),
        "n_clusters": n_clusters,
        "inertia": inertia,
        "nmi": _nmi(contingency),
        "ari": _ari(contingency),
        "purity": _purity(contingency),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "silhouette": _silhouette_score(embeddings, predicted, sample_size=silhouette_sample_size, seed=seed),
        "labels": labels,
        "predicted": predicted.tolist(),
    }
    return result


def _resolve_latest_embedding_dir(embeddings_root: Path, method: str) -> Path | None:
    method_root = embeddings_root / method
    if (method_root / "embeddings.npy").exists():
        return method_root
    if not method_root.exists() or not method_root.is_dir():
        return None
    candidates = [
        child
        for child in method_root.iterdir()
        if child.is_dir() and (child / "embeddings.npy").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_cluster_assignments(path: Path, qids: list[str], labels: list[str], clusters: list[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("qid\ttrue_label\tcluster_id\n")
        for qid, label, cluster in zip(qids, labels, clusters):
            handle.write(f"{qid}\t{label}\t{cluster}\n")


def _rank_metrics(rows: list[dict[str, Any]], metric_names: list[str]) -> list[dict[str, Any]]:
    ranked_rows = [dict(row) for row in rows]
    for metric in metric_names:
        ordered = sorted(ranked_rows, key=lambda row: float(row[metric]), reverse=True)
        for rank, row in enumerate(ordered, start=1):
            row[f"rank_{metric}"] = rank
    for row in ranked_rows:
        row["mean_rank"] = float(np.mean([row[f"rank_{metric}"] for metric in metric_names]))
        row["best_metric_count"] = int(
            sum(1 for metric in metric_names if int(row[f"rank_{metric}"]) == 1)
        )
    return ranked_rows


def _build_report_markdown(
    summary_rows: list[dict[str, Any]],
    label_field: str,
    graph_path: Path,
    embeddings_root: Path,
) -> str:
    lines = [
        "# Graph Clustering Report",
        "",
        f"- Graph: `{graph_path}`",
        f"- Embeddings root: `{embeddings_root}`",
        f"- Label field: `{label_field}`",
        "",
        "## Summary",
        "",
        "| method | nmi | ari | purity | homogeneity | completeness | v_measure | silhouette | mean_rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {nmi:.4f} | {ari:.4f} | {purity:.4f} | {homogeneity:.4f} | {completeness:.4f} | {v_measure:.4f} | {silhouette:.4f} | {mean_rank:.2f} |".format(
                **row
            )
        )
    lines.extend(["", "## Notes", "", "- Labels are used only for evaluation.", "- Clustering uses k-means with k equal to the number of unique true labels.", "- Silhouette is computed on a bounded sample when the embedding set is large."])
    return "\n".join(lines) + "\n"


def _plot_metrics(rows: list[dict[str, Any]], output_path: Path) -> None:
    methods = [row["method"] for row in rows]
    metric_names = ["nmi", "ari", "purity", "v_measure", "silhouette"]
    x = np.arange(len(methods))
    width = 0.14
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=150)
    for index, metric in enumerate(metric_names):
        offsets = x + (index - (len(metric_names) - 1) / 2.0) * width
        ax.bar(offsets, [float(row[metric]) for row in rows], width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Graph clustering metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_graph_embeddings(
    graph_path: Path,
    embeddings_root: Path,
    output_root: Path,
    label_field: str = "taxon_rank_name",
    methods: list[str] | None = None,
    seed: int = 42,
    silhouette_sample_size: int = 2000,
) -> dict[str, Any]:
    graph = load_graph(graph_path)
    known_methods = methods or ["node2vec", "gcn", "grace", "graphsage", "transe"]
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    logs_dir = output_root / "logs"
    report_dir = output_root / "report"
    for directory in [metrics_dir, plots_dir, logs_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for method in known_methods:
        embedding_dir = _resolve_latest_embedding_dir(embeddings_root, method)
        if embedding_dir is None:
            continue
        store = load_embedding_store(embedding_dir)
        result = evaluate_embedding_store(
            store=store,
            graph=graph,
            label_field=label_field,
            seed=seed,
            silhouette_sample_size=silhouette_sample_size,
        )
        rows.append(result)
        _write_cluster_assignments(
            logs_dir / f"{method}_cluster_assignments.tsv",
            [qid for qid in store.qids if qid in graph],
            result["labels"],
            result["predicted"],
        )

    if not rows:
        raise FileNotFoundError(f"No embedding stores were found under {embeddings_root}")

    metric_names = ["nmi", "ari", "purity", "homogeneity", "completeness", "v_measure", "silhouette"]
    summary_rows = _rank_metrics(rows, metric_names)

    clustering_csv_rows = [
        {key: row[key] for key in [
            "method",
            "n_samples",
            "n_clusters",
            "inertia",
            "nmi",
            "ari",
            "purity",
            "homogeneity",
            "completeness",
            "v_measure",
            "silhouette",
            "mean_rank",
            "best_metric_count",
        ]}
        for row in summary_rows
    ]

    _write_csv(
        metrics_dir / "clustering_metrics.csv",
        clustering_csv_rows,
        [
            "method",
            "n_samples",
            "n_clusters",
            "inertia",
            "nmi",
            "ari",
            "purity",
            "homogeneity",
            "completeness",
            "v_measure",
            "silhouette",
            "mean_rank",
            "best_metric_count",
        ],
    )
    _write_csv(
        metrics_dir / "summary_metrics.csv",
        clustering_csv_rows,
        [
            "method",
            "nmi",
            "ari",
            "purity",
            "homogeneity",
            "completeness",
            "v_measure",
            "silhouette",
            "mean_rank",
            "best_metric_count",
        ],
    )

    _plot_metrics(summary_rows, plots_dir / "clustering_metrics_barplot.png")

    best_by_metric = {
        metric: max(summary_rows, key=lambda row: float(row[metric]))["method"]
        for metric in ["nmi", "ari", "purity", "homogeneity", "completeness", "v_measure", "silhouette"]
    }
    best_by_rank = min(summary_rows, key=lambda row: float(row["mean_rank"]))["method"]
    report = {
        "graph_path": str(graph_path),
        "embeddings_root": str(embeddings_root),
        "label_field": label_field,
        "methods": [row["method"] for row in summary_rows],
        "best_by_metric": best_by_metric,
        "best_by_mean_rank": best_by_rank,
        "rows": summary_rows,
    }
    (report_dir / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / "experiment_report.md").write_text(
        _build_report_markdown(summary_rows, label_field, graph_path, embeddings_root),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Evaluate graph embeddings with clustering metrics.")
    parser.add_argument("--graph-input", default=str(paths.taxonomy_graph_pkl))
    parser.add_argument("--embeddings-root", default=str(paths.graph_embeddings_dir))
    parser.add_argument("--output-root", default=str(paths.graph_embeddings_dir / "evaluation"))
    parser.add_argument("--label-field", default="taxon_rank_name")
    parser.add_argument("--methods", default=None, help="Comma-separated embedding method names.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--silhouette-sample-size", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    methods = [item.strip() for item in args.methods.split(",")] if args.methods else None
    evaluate_graph_embeddings(
        graph_path=Path(args.graph_input),
        embeddings_root=Path(args.embeddings_root),
        output_root=Path(args.output_root),
        label_field=args.label_field,
        methods=methods,
        seed=args.seed,
        silhouette_sample_size=args.silhouette_sample_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
