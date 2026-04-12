from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .config import get_project_paths


RANK_ORDER = {
    "class": 0,
    "order": 1,
    "family": 2,
    "genus": 3,
    "species": 4,
    "other": 5,
}


def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load a pickled NetworkX graph. / pickle 化された NetworkX グラフを読む。"""

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.DiGraph):
        raise ValueError(f"Graph PKL must contain a networkx.DiGraph, got: {type(graph).__name__}")
    return graph


@dataclass(slots=True)
class EmbeddingStore:
    """Store embeddings and O(1) QID lookup metadata. / 埋め込み本体と QID 参照用メタデータを持つ。"""

    node_ids: list[str]
    embeddings: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    qid_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.qid_to_index = {qid: index for index, qid in enumerate(self.node_ids)}
        if self.embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {self.embeddings.shape}")
        if self.embeddings.shape[0] != len(self.node_ids):
            raise ValueError("Embeddings row count must match node count")

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def get(self, qid: str) -> np.ndarray:
        """Return one embedding vector by QID. / QID から 1 本の埋め込みを返す。"""

        return self.embeddings[self.qid_to_index[qid]]

    def get_many(self, qids: list[str]) -> np.ndarray:
        """Return a stacked embedding matrix by QID order. / QID の順で埋め込み行列を返す。"""

        indices = [self.qid_to_index[qid] for qid in qids]
        return self.embeddings[indices]


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_embedding_store(store: EmbeddingStore, output_dir: Path) -> None:
    """Persist an embedding store to disk. / 埋め込みストアをディスクへ保存する。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", store.embeddings)
    (output_dir / "node_ids.json").write_text(json.dumps(store.node_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(store.metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_embedding_store(output_dir: Path) -> EmbeddingStore:
    """Load an embedding store from disk. / 埋め込みストアをディスクから読む。"""

    node_ids = json.loads((output_dir / "node_ids.json").read_text(encoding="utf-8"))
    embeddings = np.load(output_dir / "embeddings.npy", mmap_mode="r")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return EmbeddingStore(node_ids=node_ids, embeddings=embeddings, metadata=metadata)


def _normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return matrix / norms


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _build_neighbor_map(graph: nx.DiGraph, undirected: bool = True) -> dict[str, list[str]]:
    if undirected:
        working_graph = graph.to_undirected(as_view=True)
    else:
        working_graph = graph
    return {node: sorted(working_graph.neighbors(node)) for node in graph.nodes()}


def _node_weights(graph: nx.DiGraph, node_ids: list[str]) -> np.ndarray:
    degrees = np.array([max(graph.degree(node), 1) for node in node_ids], dtype=np.float64)
    weights = np.power(degrees, 0.75)
    return weights / weights.sum()


def _sample_negative_indices(
    rng: np.random.Generator,
    node_count: int,
    probs: np.ndarray,
    k: int,
    forbidden: set[int],
) -> list[int]:
    negatives: list[int] = []
    while len(negatives) < k:
        candidate = int(rng.choice(node_count, p=probs))
        if candidate in forbidden:
            continue
        negatives.append(candidate)
    return negatives


def _generate_node2vec_walks(
    node_ids: list[str],
    neighbors: dict[str, list[str]],
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    walks: list[list[str]] = []
    shuffled_nodes = node_ids[:]

    for _ in range(num_walks):
        rng.shuffle(shuffled_nodes)
        for start_node in shuffled_nodes:
            walk = [start_node]
            previous = None
            current = start_node
            while len(walk) < walk_length:
                candidates = neighbors.get(current, [])
                if not candidates:
                    break
                if previous is None or len(candidates) == 1:
                    next_node = rng.choice(candidates)
                else:
                    candidate_weights = []
                    prev_neighbors = set(neighbors.get(previous, []))
                    for candidate in candidates:
                        if candidate == previous:
                            weight = 1.0 / p
                        elif candidate in prev_neighbors:
                            weight = 1.0
                        else:
                            weight = 1.0 / q
                        candidate_weights.append(weight)
                    total = sum(candidate_weights)
                    threshold = rng.random() * total
                    cumulative = 0.0
                    next_node = candidates[-1]
                    for candidate, weight in zip(candidates, candidate_weights):
                        cumulative += weight
                        if cumulative >= threshold:
                            next_node = candidate
                            break
                walk.append(next_node)
                previous, current = current, next_node
            walks.append(walk)
    return walks


def _train_skipgram_negative_sampling(
    walks: list[list[str]],
    node_ids: list[str],
    dim: int,
    window_size: int,
    negative_samples: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    graph: nx.DiGraph,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    node_to_index = {qid: index for index, qid in enumerate(node_ids)}
    noise_probs = _node_weights(graph, node_ids)
    input_vectors = rng.normal(0.0, 0.1 / max(dim, 1), size=(len(node_ids), dim)).astype(np.float32)
    output_vectors = np.zeros((len(node_ids), dim), dtype=np.float32)

    for _ in range(epochs):
        rng.shuffle(walks)
        for walk in walks:
            walk_indices = [node_to_index[qid] for qid in walk]
            for center_pos, center_index in enumerate(walk_indices):
                left = max(0, center_pos - window_size)
                right = min(len(walk_indices), center_pos + window_size + 1)
                for context_pos in range(left, right):
                    if context_pos == center_pos:
                        continue
                    context_index = walk_indices[context_pos]

                    center_vector = input_vectors[center_index].copy()
                    context_vector = output_vectors[context_index].copy()
                    score = float(np.dot(center_vector, context_vector))
                    grad = learning_rate * (1.0 - _sigmoid(score))
                    input_vectors[center_index] += grad * context_vector
                    output_vectors[context_index] += grad * center_vector

                    negatives = _sample_negative_indices(
                        rng=rng,
                        node_count=len(node_ids),
                        probs=noise_probs,
                        k=negative_samples,
                        forbidden={center_index, context_index},
                    )
                    for negative_index in negatives:
                        negative_vector = output_vectors[negative_index].copy()
                        score_neg = float(np.dot(center_vector, negative_vector))
                        grad_neg = learning_rate * (0.0 - _sigmoid(score_neg))
                        input_vectors[center_index] += grad_neg * negative_vector
                        output_vectors[negative_index] += grad_neg * center_vector

    return (input_vectors + output_vectors) / 2.0


def build_node2vec_embeddings(
    graph: nx.DiGraph,
    dim: int = 64,
    walk_length: int = 20,
    num_walks: int = 10,
    window_size: int = 5,
    negative_samples: int = 5,
    epochs: int = 2,
    learning_rate: float = 0.025,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
    undirected: bool = True,
) -> EmbeddingStore:
    """Train a node2vec-style embedding using walk-based skip-gram. / node2vec 風の walk ベース埋め込みを学習する。"""

    node_ids = sorted(str(node) for node in graph.nodes())
    neighbors = _build_neighbor_map(graph, undirected=undirected)
    walks = _generate_node2vec_walks(
        node_ids=node_ids,
        neighbors=neighbors,
        num_walks=num_walks,
        walk_length=walk_length,
        p=p,
        q=q,
        seed=seed,
    )
    embeddings = _train_skipgram_negative_sampling(
        walks=walks,
        node_ids=node_ids,
        dim=dim,
        window_size=window_size,
        negative_samples=negative_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        graph=graph,
    )

    metadata = {
        "algorithm": "node2vec",
        "created_at_utc": _timestamp_utc(),
        "graph_type": graph.graph.get("graph_type"),
        "root_qid": graph.graph.get("root_qid"),
        "parameters": {
            "dim": dim,
            "walk_length": walk_length,
            "num_walks": num_walks,
            "window_size": window_size,
            "negative_samples": negative_samples,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "p": p,
            "q": q,
            "seed": seed,
            "undirected": undirected,
        },
    }
    return EmbeddingStore(node_ids=node_ids, embeddings=embeddings.astype(np.float32), metadata=metadata)


def _rank_feature(rank_name: str) -> float:
    return float(RANK_ORDER.get(rank_name.strip().lower() or "other", RANK_ORDER["other"])) / float(
        RANK_ORDER["other"]
    )


def _node_depths(graph: nx.DiGraph, root_qid: str | None) -> dict[str, int]:
    if root_qid and root_qid in graph:
        return dict(nx.single_source_shortest_path_length(graph, root_qid))
    if graph.number_of_nodes() == 0:
        return {}
    candidate_root = next(iter(sorted(graph.nodes())))
    return dict(nx.single_source_shortest_path_length(graph.to_undirected(as_view=True), candidate_root))


def _stable_noise(qid: str, dim: int, seed: int) -> np.ndarray:
    digest = qid.encode("utf-8")
    value = seed
    for byte in digest:
        value = (value * 131 + byte) % (2**32)
    rng = np.random.default_rng(value)
    return rng.normal(0.0, 0.05, size=dim).astype(np.float32)


def _initial_hyperbolic_features(
    graph: nx.DiGraph,
    node_ids: list[str],
    dim: int,
    seed: int,
    root_qid: str | None,
) -> np.ndarray:
    depths = _node_depths(graph, root_qid=root_qid)
    max_depth = max(depths.values(), default=1)
    max_degree = max((graph.degree(node) for node in graph.nodes()), default=1)
    max_degree = max(max_degree, 1)

    features = np.zeros((len(node_ids), dim), dtype=np.float32)
    for index, qid in enumerate(node_ids):
        node_data = graph.nodes[qid]
        depth = float(depths.get(qid, max_depth))
        degree = float(graph.degree(qid))
        out_degree = float(graph.out_degree(qid)) if isinstance(graph, nx.DiGraph) else degree
        in_degree = float(graph.in_degree(qid)) if isinstance(graph, nx.DiGraph) else degree
        rank_name = str(node_data.get("taxon_rank_name") or "other")

        base = np.array(
            [
                depth / max(float(max_depth), 1.0),
                math.log1p(degree) / math.log1p(max_degree),
                math.log1p(out_degree) / math.log1p(max_degree),
                math.log1p(in_degree) / math.log1p(max_degree),
                _rank_feature(rank_name),
            ],
            dtype=np.float32,
        )
        if dim <= len(base):
            vector = base[:dim]
        else:
            vector = np.zeros(dim, dtype=np.float32)
            vector[: len(base)] = base
            vector[len(base) :] = _stable_noise(qid, dim - len(base), seed)
        features[index] = vector

    return _normalize_rows(features)


def _smooth_graph_features(
    graph: nx.DiGraph,
    node_ids: list[str],
    features: np.ndarray,
    layers: int,
    residual: float,
    undirected: bool = True,
) -> np.ndarray:
    neighbors = _build_neighbor_map(graph, undirected=undirected)
    node_to_index = {qid: index for index, qid in enumerate(node_ids)}
    smoothed = features.copy()

    for _ in range(layers):
        next_features = np.empty_like(smoothed)
        for index, qid in enumerate(node_ids):
            neighbor_ids = neighbors.get(qid, [])
            if neighbor_ids:
                neighbor_indices = [node_to_index[neighbor] for neighbor in neighbor_ids]
                neighborhood = np.vstack([smoothed[index], smoothed[neighbor_indices]])
                aggregated = neighborhood.mean(axis=0)
            else:
                aggregated = smoothed[index]
            next_features[index] = residual * smoothed[index] + (1.0 - residual) * aggregated
        smoothed = np.tanh(next_features)
    return smoothed


def _expmap0(x: np.ndarray, curvature: float) -> np.ndarray:
    if curvature <= 0:
        raise ValueError("Curvature must be positive")
    sqrt_c = math.sqrt(curvature)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.ones_like(norms)
    nonzero = norms > 0
    scale[nonzero] = np.tanh(sqrt_c * norms[nonzero]) / (sqrt_c * norms[nonzero])
    return (scale * x).astype(np.float32)


def build_hgcn_embeddings(
    graph: nx.DiGraph,
    dim: int = 16,
    layers: int = 2,
    residual: float = 0.5,
    curvature: float = 1.0,
    seed: int = 42,
    root_qid: str | None = None,
    undirected: bool = True,
) -> EmbeddingStore:
    """Build a hyperbolic message-passing embedding inspired by HGCN. / HGCN に着想を得たハイパボリック埋め込みを作る。"""

    node_ids = sorted(str(node) for node in graph.nodes())
    features = _initial_hyperbolic_features(graph, node_ids=node_ids, dim=dim, seed=seed, root_qid=root_qid)
    features = _smooth_graph_features(
        graph=graph,
        node_ids=node_ids,
        features=features,
        layers=layers,
        residual=residual,
        undirected=undirected,
    )
    embeddings = _expmap0(features, curvature=curvature)

    metadata = {
        "algorithm": "hgcn",
        "implementation": "hyperbolic_message_passing_baseline",
        "created_at_utc": _timestamp_utc(),
        "graph_type": graph.graph.get("graph_type"),
        "root_qid": graph.graph.get("root_qid"),
        "parameters": {
            "dim": dim,
            "layers": layers,
            "residual": residual,
            "curvature": curvature,
            "seed": seed,
            "root_qid": root_qid,
            "undirected": undirected,
        },
    }
    return EmbeddingStore(node_ids=node_ids, embeddings=embeddings, metadata=metadata)


def build_embeddings(graph: nx.DiGraph, algorithm: str, **kwargs: Any) -> EmbeddingStore:
    """Dispatch to the requested embedding algorithm. / 指定アルゴリズムへ振り分ける。"""

    normalized = algorithm.strip().lower()
    if normalized == "node2vec":
        return build_node2vec_embeddings(graph, **kwargs)
    if normalized == "hgcn":
        return build_hgcn_embeddings(graph, **kwargs)
    raise ValueError(f"Unsupported embedding algorithm: {algorithm}")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for graph embeddings. / graph 埋め込み用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build graph embeddings from a taxonomy NetworkX graph PKL.")
    parser.add_argument("--input", default=str(paths.taxonomy_graph_pkl))
    parser.add_argument("--output-dir", default=str(paths.graph_embeddings_dir / "taxonomy"))
    parser.add_argument("--algorithm", choices=["node2vec", "hgcn", "both"], default="node2vec")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--undirected", action="store_true", default=True)

    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--residual", type=float, default=0.5)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--root-qid", default=None)
    return parser


def _write_summary(store: EmbeddingStore, output_dir: Path) -> None:
    summary = {
        "node_count": len(store.node_ids),
        "dimension": store.dim,
        "metadata": store.metadata,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the graph embedding command. / graph 埋め込みコマンドを実行する。"""

    args = build_parser().parse_args(argv)
    graph = load_graph(Path(args.input))
    output_root = Path(args.output_dir)

    common_kwargs: dict[str, Any] = {
        "dim": args.dim,
        "seed": args.seed,
    }

    if args.algorithm in {"node2vec", "both"}:
        node2vec_store = build_node2vec_embeddings(
            graph,
            walk_length=args.walk_length,
            num_walks=args.num_walks,
            window_size=args.window_size,
            negative_samples=args.negative_samples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            p=args.p,
            q=args.q,
            undirected=args.undirected,
            **common_kwargs,
        )
        node2vec_dir = output_root / "node2vec"
        save_embedding_store(node2vec_store, node2vec_dir)
        _write_summary(node2vec_store, node2vec_dir)

    if args.algorithm in {"hgcn", "both"}:
        hgcn_store = build_hgcn_embeddings(
            graph,
            layers=args.layers,
            residual=args.residual,
            curvature=args.curvature,
            undirected=args.undirected,
            root_qid=args.root_qid,
            **common_kwargs,
        )
        hgcn_dir = output_root / "hgcn"
        save_embedding_store(hgcn_store, hgcn_dir)
        _write_summary(hgcn_store, hgcn_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
