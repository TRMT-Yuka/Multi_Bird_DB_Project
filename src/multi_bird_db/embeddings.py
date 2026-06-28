from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .config import get_project_paths

try:  # Optional dependency for trainable graph encoders.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency guard
    torch = None
    nn = None
    F = None


RANK_ORDER = {
    "class": 0,
    "order": 1,
    "family": 2,
    "genus": 3,
    "species": 4,
    "other": 5,
}


def _render_progress_line(message: str) -> None:
    """Render one in-place progress line to stderr. / stderr に進捗を 1 行表示する。"""

    sys.stderr.write(f"\r{message}")
    sys.stderr.flush()


def _finish_progress_line(message: str | None = None) -> None:
    """Finish an in-place progress line. / 進捗行を確定する。"""

    if message is not None:
        sys.stderr.write(f"\r{message}\n")
    else:
        sys.stderr.write("\n")
    sys.stderr.flush()


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

    qids: list[str]
    embeddings: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    qid_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.qid_to_index = {qid: index for index, qid in enumerate(self.qids)}
        if self.embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {self.embeddings.shape}")
        if self.embeddings.shape[0] != len(self.qids):
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


def _run_timestamp_mmddhhmm() -> str:
    return datetime.now().astimezone().strftime("%m%d%H%M")


def save_embedding_store(store: EmbeddingStore, output_dir: Path) -> None:
    """Persist an embedding store to disk. / 埋め込みストアをディスクへ保存する。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", store.embeddings)
    (output_dir / "qids.json").write_text(json.dumps(store.qids, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy_node_ids_path = output_dir / "node_ids.json"
    if legacy_node_ids_path.exists():
        legacy_node_ids_path.unlink()
    (output_dir / "metadata.json").write_text(
        json.dumps(store.metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_training_curve_png(output_dir, store.metadata)


def _write_training_curve_png(output_dir: Path, metadata: dict[str, Any]) -> None:
    """Render an average-loss curve when trace data is available. / trace があれば平均 loss 曲線を描画する。"""

    trace = metadata.get("training_trace")
    if not isinstance(trace, list) or not trace:
        return

    epochs: list[float] = []
    average_losses: list[float] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        try:
            epochs.append(float(item.get("epoch", len(epochs) + 1)))
            average_losses.append(float(item.get("average_loss", 0.0)))
        except (TypeError, ValueError):
            continue

    if not epochs:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    ax.plot(epochs, average_losses, marker="o", label="average_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(str(metadata.get("algorithm", "embedding")) + " training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", bbox_inches="tight")
    plt.close(fig)


def load_embedding_store(output_dir: Path) -> EmbeddingStore:
    """Load an embedding store from disk. / 埋め込みストアをディスクから読む。"""

    qids_path = output_dir / "qids.json"
    if not qids_path.exists():
        raise FileNotFoundError(f"Embedding QID manifest does not exist: {qids_path}")
    qids = json.loads(qids_path.read_text(encoding="utf-8"))
    embeddings = np.load(output_dir / "embeddings.npy", mmap_mode="r")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return EmbeddingStore(qids=qids, embeddings=embeddings, metadata=metadata)


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


def _softplus(x: float) -> float:
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _build_neighbor_map(graph: nx.DiGraph, undirected: bool = True) -> dict[str, list[str]]:
    if undirected:
        working_graph = graph.to_undirected(as_view=True)
    else:
        working_graph = graph
    return {node: sorted(working_graph.neighbors(node)) for node in graph.nodes()}


def _node_weights(graph: nx.DiGraph, qids: list[str]) -> np.ndarray:
    degrees = np.array([max(graph.degree(node), 1) for node in qids], dtype=np.float64)
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


def _require_torch() -> tuple[Any, Any, Any]:
    if torch is None or nn is None or F is None:
        raise RuntimeError(
            "PyTorch is required for trainable graph embeddings. Install torch and retry the graph embedding command."
        )
    return torch, nn, F


def _resolve_torch_device(requested_device: str, *, context: str) -> tuple[Any, str]:
    torch_mod, _, _ = _require_torch()

    normalized = str(requested_device).strip().lower()
    if normalized == "auto":
        if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
            return torch_mod.device("cuda"), "cuda"
        return torch_mod.device("cpu"), "cpu"
    if normalized == "cuda":
        if not (hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available()):
            raise RuntimeError(f"CUDA was requested for {context}, but torch.cuda.is_available() is false.")
        return torch_mod.device("cuda"), "cuda"
    return torch_mod.device("cpu"), "cpu"


def _gradient_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        total += float(torch.linalg.norm(grad.detach()).cpu().item()) ** 2
    return math.sqrt(total)


def _normalize_rows_torch(matrix: Any, eps: float = 1e-12) -> Any:
    norms = torch.linalg.norm(matrix, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=eps)
    return matrix / norms


def _build_torch_graph_data(
    graph: nx.DiGraph,
    qids: list[str],
    undirected: bool = True,
) -> tuple[dict[str, int], list[list[int]], Any, set[tuple[int, int]], Any]:
    torch_mod, _, _ = _require_torch()

    node_to_index = {qid: index for index, qid in enumerate(qids)}
    rows: list[int] = []
    cols: list[int] = []
    neighbors: list[list[int]] = [[] for _ in qids]
    edge_set: set[tuple[int, int]] = set()

    for source, target in graph.edges():
        source_qid = str(source)
        target_qid = str(target)
        if source_qid not in node_to_index or target_qid not in node_to_index:
            continue
        source_index = node_to_index[source_qid]
        target_index = node_to_index[target_qid]
        edge_set.add((source_index, target_index))
        rows.append(source_index)
        cols.append(target_index)
        neighbors[source_index].append(target_index)
        if undirected and source_index != target_index:
            rows.append(target_index)
            cols.append(source_index)
            neighbors[target_index].append(source_index)

    for index in range(len(qids)):
        rows.append(index)
        cols.append(index)
        neighbors[index].append(index)

    neighbors = [sorted(set(items)) for items in neighbors]
    if rows:
        row_tensor = torch_mod.tensor(rows, dtype=torch.long)
        col_tensor = torch_mod.tensor(cols, dtype=torch.long)
        values = torch_mod.ones(len(rows), dtype=torch.float32)
        degree = torch_mod.bincount(row_tensor, minlength=len(qids)).to(torch.float32).clamp_min(1.0)
        normalized_values = values / degree[row_tensor]
        adjacency = torch_mod.sparse_coo_tensor(
            torch_mod.stack([row_tensor, col_tensor]),
            normalized_values,
            size=(len(qids), len(qids)),
        ).coalesce()
    else:
        degree = torch_mod.ones(len(qids), dtype=torch.float32)
        adjacency = torch_mod.sparse_coo_tensor(
            torch_mod.zeros((2, 0), dtype=torch.long),
            torch_mod.tensor([], dtype=torch.float32),
            size=(len(qids), len(qids)),
        ).coalesce()
    return node_to_index, neighbors, adjacency, edge_set, degree


def _edge_pairs_from_graph(
    graph: nx.DiGraph,
    qids: list[str],
    undirected: bool = True,
) -> tuple[dict[str, int], list[tuple[int, int]]]:
    node_to_index = {qid: index for index, qid in enumerate(qids)}
    edge_pairs: list[tuple[int, int]] = []
    for source, target in graph.edges():
        source_qid = str(source)
        target_qid = str(target)
        if source_qid not in node_to_index or target_qid not in node_to_index:
            continue
        source_index = node_to_index[source_qid]
        target_index = node_to_index[target_qid]
        edge_pairs.append((source_index, target_index))
        if undirected and source_index != target_index:
            edge_pairs.append((target_index, source_index))
    return node_to_index, edge_pairs


def _sample_negative_edge(
    rng: np.random.Generator,
    node_count: int,
    head_index: int,
    tail_index: int,
    edge_set: set[tuple[int, int]],
) -> tuple[int, int]:
    if node_count <= 2:
        return head_index, tail_index
    corrupt_head = bool(rng.integers(0, 2))
    if corrupt_head:
        while True:
            candidate = int(rng.integers(0, node_count))
            if candidate != head_index and (candidate, tail_index) not in edge_set:
                return candidate, tail_index
    while True:
        candidate = int(rng.integers(0, node_count))
        if candidate != tail_index and (head_index, candidate) not in edge_set:
            return head_index, candidate


def _sparse_adjacency_from_edges(
    node_count: int,
    edge_pairs: list[tuple[int, int]],
    device: Any,
) -> Any:
    torch_mod, _, _ = _require_torch()

    rows: list[int] = []
    cols: list[int] = []
    for source_index, target_index in edge_pairs:
        rows.append(source_index)
        cols.append(target_index)
    for index in range(node_count):
        rows.append(index)
        cols.append(index)

    if not rows:
        return torch_mod.sparse_coo_tensor(
            torch_mod.zeros((2, 0), dtype=torch.long, device=device),
            torch_mod.tensor([], dtype=torch.float32, device=device),
            size=(node_count, node_count),
            device=device,
        ).coalesce()

    row_tensor = torch_mod.tensor(rows, dtype=torch.long, device=device)
    col_tensor = torch_mod.tensor(cols, dtype=torch.long, device=device)
    degree = torch_mod.zeros(node_count, dtype=torch.float32, device=device)
    degree.index_add_(0, row_tensor, torch_mod.ones(len(rows), dtype=torch.float32, device=device))
    degree = degree.clamp_min(1.0)
    normalized_values = 1.0 / torch_mod.sqrt(degree[row_tensor] * degree[col_tensor])
    return torch_mod.sparse_coo_tensor(
        torch_mod.stack([row_tensor, col_tensor]),
        normalized_values,
        size=(node_count, node_count),
        device=device,
    ).coalesce()


def _drop_edge_pairs(
    edge_pairs: list[tuple[int, int]],
    drop_rate: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if not edge_pairs or drop_rate <= 0.0:
        return edge_pairs[:]
    keep_mask = rng.random(len(edge_pairs)) >= drop_rate
    kept = [edge for edge, keep in zip(edge_pairs, keep_mask) if keep]
    if kept:
        return kept
    return [edge_pairs[int(rng.integers(0, len(edge_pairs)))]]


def _drop_feature_matrix(features: Any, drop_rate: float, rng: np.random.Generator) -> Any:
    torch_mod, _, _ = _require_torch()
    if drop_rate <= 0.0:
        return features
    mask = torch_mod.from_numpy((rng.random(features.shape) >= drop_rate).astype(np.float32)).to(features.device)
    return features * mask


def _drop_feature_columns(features: Any, drop_rate: float, rng: np.random.Generator) -> Any:
    torch_mod, _, _ = _require_torch()
    if drop_rate <= 0.0:
        return features
    drop_mask = torch_mod.from_numpy((rng.random(features.shape[1]) < drop_rate)).to(features.device)
    dropped = features.clone()
    dropped[:, drop_mask] = 0.0
    return dropped


def _sample_neighbor_layers(
    neighbors: list[list[int]],
    num_neighbors: list[int],
    rng: np.random.Generator,
) -> list[list[list[int]]]:
    sampled_layers: list[list[list[int]]] = []
    current_neighbors = neighbors
    for fanout in num_neighbors:
        layer_neighbors: list[list[int]] = []
        for node_index, node_neighbors in enumerate(current_neighbors):
            candidates = [index for index in node_neighbors if index != node_index]
            if not candidates:
                sampled = [node_index]
            elif fanout <= 0 or len(candidates) <= fanout:
                sampled = candidates[:]
            else:
                sampled = rng.choice(candidates, size=fanout, replace=False).tolist()
            if node_index not in sampled:
                sampled.append(node_index)
            layer_neighbors.append(sorted(set(int(index) for index in sampled)))
        sampled_layers.append(layer_neighbors)
        current_neighbors = layer_neighbors
    return sampled_layers


def _contrastive_loss(z1: Any, z2: Any, tau: float) -> Any:
    torch_mod, _, F_mod = _require_torch()
    z1 = F_mod.normalize(z1, p=2, dim=1)
    z2 = F_mod.normalize(z2, p=2, dim=1)
    logits = torch_mod.matmul(z1, z2.T) / max(tau, 1e-6)
    labels = torch_mod.arange(z1.shape[0], device=z1.device)
    loss_a = F_mod.cross_entropy(logits, labels)
    loss_b = F_mod.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a + loss_b)


def _grace_contrastive_loss(z1: Any, z2: Any, tau: float, batch_size: int = 0) -> Any:
    torch_mod, _, _ = _require_torch()
    if batch_size <= 0 or batch_size >= z1.shape[0]:
        return _contrastive_loss(z1, z2, tau)
    losses: list[Any] = []
    for start in range(0, z1.shape[0], batch_size):
        stop = min(start + batch_size, z1.shape[0])
        if stop - start < 2:
            continue
        losses.append(_contrastive_loss(z1[start:stop], z2[start:stop], tau))
    if not losses:
        return _contrastive_loss(z1, z2, tau)
    return torch_mod.stack(losses).mean()


class _GCNConv((nn.Module if nn is not None else object)):
    """Lightweight GCNConv-style layer backed by PyTorch only."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: Any, adjacency: Any) -> Any:
        support = self.linear(x)
        return torch.sparse.mm(adjacency, support)


class _GraphSAGELayer((nn.Module if nn is not None else object)):
    """Lightweight mean-aggregator GraphSAGE layer backed by PyTorch only."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels * 2, channels, bias=True)

    def forward(self, h: Any, neighbor_ids: list[list[int]]) -> Any:
        torch_mod, _, _ = _require_torch()
        aggregated: list[Any] = []
        for ids in neighbor_ids:
            if not ids:
                aggregated.append(h.new_zeros(h.shape[1]))
                continue
            neighbor_tensor = torch_mod.tensor(ids, dtype=torch_mod.long, device=h.device)
            neighbor_vectors = h.index_select(0, neighbor_tensor)
            aggregated.append(neighbor_vectors.mean(dim=0))
        aggregated_tensor = torch_mod.stack(aggregated, dim=0)
        return self.linear(torch_mod.cat([h, aggregated_tensor], dim=1))


def _generate_node2vec_walks(
    qids: list[str],
    neighbors: dict[str, list[str]],
    num_walks: int,
    walk_length: int,
    p: float,
    q: float,
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    walks: list[list[str]] = []
    shuffled_nodes = qids[:]

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


def _train_skipgram_negative_sampling_numpy(
    walks: list[list[str]],
    qids: list[str],
    dim: int,
    window_size: int,
    negative_samples: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    graph: nx.DiGraph,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    node_to_index = {qid: index for index, qid in enumerate(qids)}
    noise_probs = _node_weights(graph, qids)
    input_vectors = rng.normal(0.0, 0.1 / max(dim, 1), size=(len(qids), dim)).astype(np.float32)
    output_vectors = np.zeros((len(qids), dim), dtype=np.float32)
    trace: list[dict[str, float]] = []

    for epoch in range(epochs):
        rng.shuffle(walks)
        total_loss = 0.0
        pair_count = 0
        walk_progress_interval = max(len(walks) // 20, 1)
        last_progress_time = time.monotonic()
        _render_progress_line(
            f"node2vec epoch {epoch + 1}/{epochs} | walks 0/{len(walks)} | pairs 0 | loss 0.0000"
        )
        for walk_index, walk in enumerate(walks, start=1):
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
                    total_loss += _softplus(-score)
                    pair_count += 1
                    grad = learning_rate * (1.0 - _sigmoid(score))
                    input_vectors[center_index] += grad * context_vector
                    output_vectors[context_index] += grad * center_vector

                    negatives = _sample_negative_indices(
                        rng=rng,
                        node_count=len(qids),
                        probs=noise_probs,
                        k=negative_samples,
                        forbidden={center_index, context_index},
                    )
                    for negative_index in negatives:
                        negative_vector = output_vectors[negative_index].copy()
                        score_neg = float(np.dot(center_vector, negative_vector))
                        total_loss += _softplus(score_neg)
                        grad_neg = learning_rate * (0.0 - _sigmoid(score_neg))
                        input_vectors[center_index] += grad_neg * negative_vector
                        output_vectors[negative_index] += grad_neg * center_vector
            now = time.monotonic()
            if walk_index % walk_progress_interval == 0 or now - last_progress_time >= 2.0:
                _render_progress_line(
                    f"node2vec epoch {epoch + 1}/{epochs} | walks {walk_index}/{len(walks)} | "
                    f"pairs {pair_count} | loss {total_loss:.4f}"
                )
                last_progress_time = now

        trace.append(
            {
                "epoch": float(epoch + 1),
                "pair_count": float(pair_count),
                "average_loss": float(total_loss / max(pair_count, 1)),
            }
        )
        _finish_progress_line(
            f"node2vec epoch {epoch + 1}/{epochs} done | walks {len(walks)}/{len(walks)} | "
            f"pairs {pair_count} | loss {total_loss:.4f}"
        )

    return (input_vectors + output_vectors) / 2.0, trace


def _sample_negative_indices_torch(
    noise_probs: Any,
    center_batch: Any,
    context_batch: Any,
    negative_samples: int,
) -> Any:
    torch_mod, _, _ = _require_torch()
    if negative_samples <= 0 or center_batch.numel() == 0:
        return torch_mod.empty((center_batch.shape[0], 0), dtype=torch_mod.long, device=center_batch.device)

    negatives = torch_mod.multinomial(
        noise_probs,
        center_batch.shape[0] * negative_samples,
        replacement=True,
    ).reshape(center_batch.shape[0], negative_samples)
    invalid = (negatives == center_batch.unsqueeze(1)) | (negatives == context_batch.unsqueeze(1))
    while bool(invalid.any().item()):
        replacement = torch_mod.multinomial(noise_probs, int(invalid.sum().item()), replacement=True)
        negatives[invalid] = replacement
        invalid = (negatives == center_batch.unsqueeze(1)) | (negatives == context_batch.unsqueeze(1))
    return negatives


def _train_skipgram_negative_sampling_torch(
    walks: list[list[str]],
    qids: list[str],
    dim: int,
    window_size: int,
    negative_samples: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    graph: nx.DiGraph,
    device_obj: Any,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    torch_mod, nn_mod, F_mod = _require_torch()
    torch_mod.manual_seed(seed)
    if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)
    node_to_index = {qid: index for index, qid in enumerate(qids)}
    noise_probs = torch_mod.tensor(_node_weights(graph, qids), dtype=torch_mod.float32, device=device_obj)

    input_embeddings = nn_mod.Embedding(len(qids), dim, device=device_obj)
    output_embeddings = nn_mod.Embedding(len(qids), dim, device=device_obj)
    with torch_mod.no_grad():
        input_embeddings.weight.normal_(0.0, 0.1 / max(dim, 1))
        output_embeddings.weight.zero_()

    optimizer = torch_mod.optim.Adam(
        list(input_embeddings.parameters()) + list(output_embeddings.parameters()),
        lr=learning_rate,
    )
    batch_size = 1024
    trace: list[dict[str, float]] = []

    for epoch in range(epochs):
        rng.shuffle(walks)
        total_loss = 0.0
        pair_count = 0
        walk_progress_interval = max(len(walks) // 20, 1)
        last_progress_time = time.monotonic()
        _render_progress_line(
            f"node2vec epoch {epoch + 1}/{epochs} | walks 0/{len(walks)} | pairs 0 | loss 0.0000"
        )
        for walk_index, walk in enumerate(walks, start=1):
            walk_indices = [node_to_index[qid] for qid in walk]
            positive_pairs: list[tuple[int, int]] = []
            for center_pos, center_index in enumerate(walk_indices):
                left = max(0, center_pos - window_size)
                right = min(len(walk_indices), center_pos + window_size + 1)
                for context_pos in range(left, right):
                    if context_pos == center_pos:
                        continue
                    positive_pairs.append((center_index, walk_indices[context_pos]))

            pair_count += len(positive_pairs)
            for batch_start in range(0, len(positive_pairs), batch_size):
                batch_pairs = positive_pairs[batch_start : batch_start + batch_size]
                center_batch = torch_mod.tensor(
                    [center_index for center_index, _ in batch_pairs],
                    dtype=torch_mod.long,
                    device=device_obj,
                )
                context_batch = torch_mod.tensor(
                    [context_index for _, context_index in batch_pairs],
                    dtype=torch_mod.long,
                    device=device_obj,
                )
                negative_batch = _sample_negative_indices_torch(
                    noise_probs=noise_probs,
                    center_batch=center_batch,
                    context_batch=context_batch,
                    negative_samples=negative_samples,
                )

                center_vectors = input_embeddings(center_batch)
                context_vectors = output_embeddings(context_batch)
                pos_scores = torch_mod.sum(center_vectors * context_vectors, dim=1)
                pos_loss = F_mod.softplus(-pos_scores).sum()

                if negative_batch.numel():
                    negative_vectors = output_embeddings(negative_batch)
                    neg_scores = torch_mod.sum(negative_vectors * center_vectors.unsqueeze(1), dim=2)
                    neg_loss = F_mod.softplus(neg_scores).sum()
                else:
                    neg_loss = torch_mod.tensor(0.0, dtype=torch_mod.float32, device=device_obj)

                batch_loss = pos_loss + neg_loss
                optimizer.zero_grad()
                (batch_loss / max(center_batch.shape[0], 1)).backward()
                optimizer.step()
                total_loss += float(batch_loss.detach().cpu().item())

            now = time.monotonic()
            if walk_index % walk_progress_interval == 0 or now - last_progress_time >= 2.0:
                _render_progress_line(
                    f"node2vec epoch {epoch + 1}/{epochs} | walks {walk_index}/{len(walks)} | "
                    f"pairs {pair_count} | loss {total_loss:.4f}"
                )
                last_progress_time = now

        trace.append(
            {
                "epoch": float(epoch + 1),
                "pair_count": float(pair_count),
                "average_loss": float(total_loss / max(pair_count, 1)),
            }
        )
        _finish_progress_line(
            f"node2vec epoch {epoch + 1}/{epochs} done | walks {len(walks)}/{len(walks)} | "
            f"pairs {pair_count} | loss {total_loss:.4f}"
        )

    with torch_mod.no_grad():
        embeddings = (input_embeddings.weight + output_embeddings.weight) / 2.0
    return embeddings.detach().cpu().numpy().astype(np.float32), trace


def _train_skipgram_negative_sampling(
    walks: list[list[str]],
    qids: list[str],
    dim: int,
    window_size: int,
    negative_samples: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    graph: nx.DiGraph,
    device: str,
) -> tuple[np.ndarray, list[dict[str, float]], str]:
    normalized = str(device).strip().lower()
    if torch is None:
        if normalized == "cuda":
            raise RuntimeError("CUDA was requested for node2vec, but PyTorch is not installed.")
        embeddings, trace = _train_skipgram_negative_sampling_numpy(
            walks=walks,
            qids=qids,
            dim=dim,
            window_size=window_size,
            negative_samples=negative_samples,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            graph=graph,
        )
        return embeddings, trace, "cpu"

    device_obj, resolved_device = _resolve_torch_device(device, context="node2vec")
    embeddings, trace = _train_skipgram_negative_sampling_torch(
        walks=walks,
        qids=qids,
        dim=dim,
        window_size=window_size,
        negative_samples=negative_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        graph=graph,
        device_obj=device_obj,
    )
    return embeddings, trace, resolved_device


def build_node2vec_embeddings(
    graph: nx.DiGraph,
    dim: int = 128,
    walk_length: int = 40,
    num_walks: int = 10,
    window_size: int = 10,
    negative_samples: int = 5,
    epochs: int = 200,
    learning_rate: float = 0.001,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
    undirected: bool = False,
    device: str = "auto",
) -> EmbeddingStore:
    """Train a node2vec-style embedding using walk-based skip-gram. / node2vec 風の walk ベース埋め込みを学習する。"""

    qids = sorted(str(node) for node in graph.nodes())
    neighbors = _build_neighbor_map(graph, undirected=undirected)
    walks = _generate_node2vec_walks(
        qids=qids,
        neighbors=neighbors,
        num_walks=num_walks,
        walk_length=walk_length,
        p=p,
        q=q,
        seed=seed,
    )
    embeddings, trace, resolved_device = _train_skipgram_negative_sampling(
        walks=walks,
        qids=qids,
        dim=dim,
        window_size=window_size,
        negative_samples=negative_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        graph=graph,
        device=device,
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
            "device": device,
            "resolved_device": resolved_device,
        },
        "training_trace": trace,
    }
    return EmbeddingStore(qids=qids, embeddings=embeddings.astype(np.float32), metadata=metadata)


def _build_structural_features(
    graph: nx.DiGraph,
    qids: list[str],
    feature_mode: str,
    seed: int,
    dim: int,
) -> np.ndarray:
    """Build structural node features without label leakage. / ラベル漏洩のない構造特徴を作る。"""

    mode = feature_mode.strip().lower()
    rng = np.random.default_rng(seed)
    node_count = len(qids)

    if node_count == 0:
        return np.zeros((0, 1), dtype=np.float32)

    if mode == "degree":
        max_degree = max((graph.degree(node) for node in qids), default=1)
        max_degree = max(max_degree, 1)
        max_in_degree = max((graph.in_degree(node) for node in qids), default=1)
        max_out_degree = max((graph.out_degree(node) for node in qids), default=1)
        feature_rows = []
        for qid in qids:
            degree = float(graph.degree(qid))
            in_degree = float(graph.in_degree(qid))
            out_degree = float(graph.out_degree(qid))
            feature_rows.append(
                [
                    degree / float(max_degree),
                    in_degree / float(max_in_degree),
                    out_degree / float(max_out_degree),
                    math.log1p(degree) / math.log1p(float(max_degree)),
                ]
            )
        return np.asarray(feature_rows, dtype=np.float32)

    if mode == "one_hot":
        return np.eye(node_count, dtype=np.float32)

    if mode == "constant":
        return np.ones((node_count, 1), dtype=np.float32)

    if mode == "random":
        random_dim = max(int(dim), 1)
        random_features = rng.normal(0.0, 1.0 / math.sqrt(max(random_dim, 1)), size=(node_count, random_dim))
        return _normalize_rows(random_features.astype(np.float32))

    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def _initial_entity_embeddings(qids: list[str], dim: int, seed: int) -> np.ndarray:
    """Build unlabeled initial entity embeddings for TransE. / TransE 用のラベル非依存初期埋め込みを作る。"""

    rng = np.random.default_rng(seed)
    embeddings = rng.normal(0.0, 0.1, size=(len(qids), dim)).astype(np.float32)
    return _normalize_rows(embeddings)


def _train_gcn_embeddings(
    graph: nx.DiGraph,
    qids: list[str],
    dim: int,
    layers: int,
    residual: float,
    epochs: int,
    learning_rate: float,
    negative_samples: int,
    weight_decay: float,
    seed: int,
    undirected: bool,
    feature_mode: str,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    torch_mod, nn_mod, F_mod = _require_torch()
    rng = np.random.default_rng(seed)
    node_to_index, _, adjacency, edge_set, _ = _build_torch_graph_data(graph, qids=qids, undirected=undirected)

    if graph.number_of_edges() == 0:
        features = _build_structural_features(graph, qids=qids, feature_mode=feature_mode, seed=seed, dim=dim)
        return features.astype(np.float32), []

    initial_features = _build_structural_features(graph, qids=qids, feature_mode=feature_mode, seed=seed, dim=dim)
    input_dim = int(initial_features.shape[1])
    edge_pairs = [
        (node_to_index[str(source)], node_to_index[str(target)])
        for source, target in graph.edges()
        if str(source) in node_to_index and str(target) in node_to_index
    ]
    if not edge_pairs:
        return initial_features.astype(np.float32), []

    class GCNEncoder(nn_mod.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn_mod.Linear(input_dim, dim)
            self.layers = nn_mod.ModuleList([_GCNConv(dim, dim) for _ in range(max(layers - 1, 0))])

        def forward(self) -> Any:
            h = self.input(torch_mod.tensor(initial_features, dtype=torch_mod.float32, device=adjacency.device))
            h = F_mod.relu(h)
            h = F_mod.normalize(h, p=2, dim=1)
            for layer in self.layers:
                aggregated = layer(h, adjacency)
                h = residual * h + (1.0 - residual) * aggregated
                h = F_mod.relu(h)
                h = F_mod.normalize(h, p=2, dim=1)
            return h

    encoder = GCNEncoder().to(adjacency.device)
    optimizer = torch_mod.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_heads = torch_mod.tensor([head for head, _ in edge_pairs], dtype=torch_mod.long, device=adjacency.device)
    pos_tails = torch_mod.tensor([tail for _, tail in edge_pairs], dtype=torch_mod.long, device=adjacency.device)

    trace: list[dict[str, float]] = []
    final_embeddings = torch_mod.tensor(initial_features, dtype=torch_mod.float32, device=adjacency.device)
    for epoch in range(epochs):
        encoder.train()
        optimizer.zero_grad()
        embeddings = encoder()

        negative_pairs: list[tuple[int, int]] = []
        edge_progress_interval = max(len(edge_pairs) // 20, 1)
        last_progress_time = time.monotonic()
        _render_progress_line(f"gcn epoch {epoch + 1}/{epochs} | edges 0/{len(edge_pairs)} | loss pending")
        for edge_index, (head_index, tail_index) in enumerate(edge_pairs, start=1):
            for _ in range(negative_samples):
                negative_pairs.append(_sample_negative_edge(rng, len(qids), head_index, tail_index, edge_set))
            now = time.monotonic()
            if edge_index % edge_progress_interval == 0 or now - last_progress_time >= 2.0:
                _render_progress_line(
                    f"gcn epoch {epoch + 1}/{epochs} | edges {edge_index}/{len(edge_pairs)} | "
                    f"negatives {len(negative_pairs)} | loss pending"
                )
                last_progress_time = now

        pos_scores = torch_mod.sum(embeddings.index_select(0, pos_heads) * embeddings.index_select(0, pos_tails), dim=1)
        pos_loss = F_mod.softplus(-pos_scores).mean()

        if negative_pairs:
            neg_heads = torch_mod.tensor([head for head, _ in negative_pairs], dtype=torch_mod.long, device=adjacency.device)
            neg_tails = torch_mod.tensor([tail for _, tail in negative_pairs], dtype=torch_mod.long, device=adjacency.device)
            neg_scores = torch_mod.sum(embeddings.index_select(0, neg_heads) * embeddings.index_select(0, neg_tails), dim=1)
            neg_loss = F_mod.softplus(neg_scores).mean()
            total_loss = pos_loss + neg_loss
        else:
            neg_scores = torch_mod.tensor([], dtype=torch_mod.float32, device=adjacency.device)
            neg_loss = torch_mod.tensor(0.0, dtype=torch_mod.float32, device=adjacency.device)
            total_loss = pos_loss

        total_loss.backward()
        grad_norm = _gradient_norm(encoder.parameters())
        optimizer.step()

        with torch_mod.no_grad():
            encoder.eval()
            final_embeddings = encoder()
            total_loss_value = float(total_loss.detach().cpu().item())
            pos_score_value = float(pos_scores.detach().mean().cpu().item()) if pos_scores.numel() else 0.0
            neg_score_value = float(neg_scores.detach().mean().cpu().item()) if neg_scores.numel() else 0.0
            embedding_norms = torch_mod.linalg.norm(final_embeddings, dim=1)
            embedding_mean = float(embedding_norms.mean().cpu().item())
            embedding_std = float(embedding_norms.std(unbiased=False).cpu().item())
            embedding_var = float(final_embeddings.var(dim=0, unbiased=False).mean().cpu().item())
        trace.append(
            {
                "epoch": float(epoch + 1),
                "positive_pairs": float(len(edge_pairs)),
                "negative_pairs": float(len(negative_pairs)),
                "train_loss": total_loss_value,
                "average_loss": total_loss_value,
                "positive_score_mean": pos_score_value,
                "negative_score_mean": neg_score_value,
                "score_gap": float(pos_score_value - neg_score_value),
                "embedding_norm_mean": embedding_mean,
                "embedding_norm_std": embedding_std,
                "embedding_variance_mean": embedding_var,
                "gradient_norm": float(grad_norm),
                "learning_rate": float(learning_rate),
            }
        )
        _finish_progress_line(
            f"gcn epoch {epoch + 1}/{epochs} done | edges {len(edge_pairs)}/{len(edge_pairs)} | "
            f"loss {total_loss_value:.4f}"
        )

    embeddings = final_embeddings.detach().cpu().numpy().astype(np.float32)
    return embeddings, trace


def build_gcn_embeddings(
    graph: nx.DiGraph,
    dim: int = 128,
    layers: int = 1,
    residual: float = 0.0,
    epochs: int = 300,
    learning_rate: float = 0.01,
    negative_samples: int = 20,
    feature_mode: str = "degree",
    weight_decay: float = 0.0,
    seed: int = 42,
    root_qid: str | None = None,
    undirected: bool = False,
) -> EmbeddingStore:
    """Train a self-supervised GCN autoencoder. / 自己教師あり GCN を学習する。"""

    qids = sorted(str(node) for node in graph.nodes())
    embeddings, trace = _train_gcn_embeddings(
        graph=graph,
        qids=qids,
        dim=dim,
        layers=layers,
        residual=residual,
        epochs=epochs,
        learning_rate=learning_rate,
        negative_samples=negative_samples,
        weight_decay=weight_decay,
        seed=seed,
        undirected=undirected,
        feature_mode=feature_mode,
    )
    metadata = {
        "algorithm": "gcn",
        "implementation": "self_supervised_gcn_autoencoder",
        "created_at_utc": _timestamp_utc(),
        "graph_type": graph.graph.get("graph_type"),
        "root_qid": graph.graph.get("root_qid"),
        "relation": "parent_taxon",
        "objective": "link_prediction",
        "parameters": {
            "dim": dim,
            "layers": layers,
            "residual": residual,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "negative_samples": negative_samples,
            "feature_mode": feature_mode,
            "weight_decay": weight_decay,
            "seed": seed,
            "root_qid": root_qid,
            "undirected": undirected,
        },
        "training_trace": trace,
    }
    return EmbeddingStore(qids=qids, embeddings=embeddings.astype(np.float32), metadata=metadata)


def _train_graphsage_embeddings(
    graph: nx.DiGraph,
    qids: list[str],
    dim: int,
    layers: int,
    residual: float,
    epochs: int,
    learning_rate: float,
    negative_samples: int,
    num_neighbors_1: int,
    num_neighbors_2: int,
    feature_mode: str,
    seed: int,
    undirected: bool,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    torch_mod, nn_mod, F_mod = _require_torch()
    rng = np.random.default_rng(seed)
    node_to_index, neighbors, adjacency, edge_set, _ = _build_torch_graph_data(graph, qids=qids, undirected=undirected)
    features = _build_structural_features(graph, qids=qids, feature_mode=feature_mode, seed=seed, dim=dim)
    input_dim = int(features.shape[1])
    edge_pairs = [
        (node_to_index[str(source)], node_to_index[str(target)])
        for source, target in graph.edges()
        if str(source) in node_to_index and str(target) in node_to_index
    ]

    if not edge_pairs:
        return features.astype(np.float32), []

    num_neighbor_layers = [max(int(num_neighbors_1), 1), max(int(num_neighbors_2), 1)]

    class GraphSAGEEncoder(nn_mod.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn_mod.Linear(input_dim, dim)
            self.layers = nn_mod.ModuleList([_GraphSAGELayer(dim) for _ in range(max(layers, 1))])

        def forward(self, sampled_neighbor_layers: list[list[list[int]]]) -> Any:
            h = torch_mod.tensor(features, dtype=torch_mod.float32, device=adjacency.device)
            h = self.input(h)
            h = F_mod.relu(h)
            h = F_mod.normalize(h, p=2, dim=1)
            for layer_index, layer in enumerate(self.layers):
                neighbor_layer_index = min(layer_index, len(sampled_neighbor_layers) - 1)
                aggregated = layer(h, sampled_neighbor_layers[neighbor_layer_index])
                h = residual * h + (1.0 - residual) * aggregated
                h = F_mod.relu(h)
                h = F_mod.normalize(h, p=2, dim=1)
            return h

    encoder = GraphSAGEEncoder().to(adjacency.device)
    optimizer = torch_mod.optim.Adam(encoder.parameters(), lr=learning_rate)
    noise_probs = _node_weights(graph, qids)
    trace: list[dict[str, float]] = []
    final_embeddings = torch_mod.tensor(features, dtype=torch_mod.float32, device=adjacency.device)

    for epoch in range(epochs):
        encoder.train()
        optimizer.zero_grad()
        sampled_neighbor_layers = _sample_neighbor_layers(neighbors, num_neighbor_layers, rng)
        embeddings = encoder(sampled_neighbor_layers)

        positive_pairs = edge_pairs[:]
        rng.shuffle(positive_pairs)
        negative_pairs: list[tuple[int, int]] = []
        edge_progress_interval = max(len(positive_pairs) // 20, 1)
        last_progress_time = time.monotonic()
        _render_progress_line(
            f"graphsage epoch {epoch + 1}/{epochs} | edges 0/{len(positive_pairs)} | loss pending"
        )
        for edge_index, (head_index, tail_index) in enumerate(positive_pairs, start=1):
            for _ in range(negative_samples):
                negative_pairs.append(_sample_negative_edge(rng, len(qids), head_index, tail_index, edge_set))
            now = time.monotonic()
            if edge_index % edge_progress_interval == 0 or now - last_progress_time >= 2.0:
                _render_progress_line(
                    f"graphsage epoch {epoch + 1}/{epochs} | edges {edge_index}/{len(positive_pairs)} | "
                    f"negatives {len(negative_pairs)} | loss pending"
                )
                last_progress_time = now

        pos_heads = torch_mod.tensor([head for head, _ in positive_pairs], dtype=torch_mod.long, device=adjacency.device)
        pos_tails = torch_mod.tensor([tail for _, tail in positive_pairs], dtype=torch_mod.long, device=adjacency.device)
        pos_scores = torch_mod.sum(embeddings.index_select(0, pos_heads) * embeddings.index_select(0, pos_tails), dim=1)
        pos_loss = F_mod.softplus(-pos_scores).mean()

        if negative_pairs:
            neg_heads = torch_mod.tensor([head for head, _ in negative_pairs], dtype=torch_mod.long, device=adjacency.device)
            neg_tails = torch_mod.tensor([tail for _, tail in negative_pairs], dtype=torch_mod.long, device=adjacency.device)
            neg_scores = torch_mod.sum(embeddings.index_select(0, neg_heads) * embeddings.index_select(0, neg_tails), dim=1)
            neg_loss = F_mod.softplus(neg_scores).mean()
            total_loss = pos_loss + neg_loss
        else:
            neg_scores = torch_mod.tensor([], dtype=torch_mod.float32, device=adjacency.device)
            neg_loss = torch_mod.tensor(0.0, dtype=torch_mod.float32, device=adjacency.device)
            total_loss = pos_loss

        total_loss.backward()
        grad_norm = _gradient_norm(encoder.parameters())
        optimizer.step()

        with torch_mod.no_grad():
            encoder.eval()
            final_embeddings = encoder(_sample_neighbor_layers(neighbors, num_neighbor_layers, rng))
            total_loss_value = float(total_loss.detach().cpu().item())
            pos_score_value = float(pos_scores.detach().mean().cpu().item()) if pos_scores.numel() else 0.0
            neg_score_value = float(neg_scores.detach().mean().cpu().item()) if neg_scores.numel() else 0.0
            embedding_norms = torch_mod.linalg.norm(final_embeddings, dim=1)
            embedding_mean = float(embedding_norms.mean().cpu().item())
            embedding_std = float(embedding_norms.std(unbiased=False).cpu().item())
            embedding_var = float(final_embeddings.var(dim=0, unbiased=False).mean().cpu().item())
        trace.append(
            {
                "epoch": float(epoch + 1),
                "positive_pairs": float(len(positive_pairs)),
                "negative_pairs": float(len(negative_pairs)),
                "train_loss": total_loss_value,
                "average_loss": total_loss_value,
                "positive_score_mean": pos_score_value,
                "negative_score_mean": neg_score_value,
                "score_gap": float(pos_score_value - neg_score_value),
                "embedding_norm_mean": embedding_mean,
                "embedding_norm_std": embedding_std,
                "embedding_variance_mean": embedding_var,
                "gradient_norm": float(grad_norm),
                "learning_rate": float(learning_rate),
                "num_neighbors_1": float(num_neighbor_layers[0]),
                "num_neighbors_2": float(num_neighbor_layers[1]),
            }
        )
        _finish_progress_line(
            f"graphsage epoch {epoch + 1}/{epochs} done | edges {len(positive_pairs)}/{len(positive_pairs)} | "
            f"loss {total_loss_value:.4f}"
        )

    return final_embeddings.detach().cpu().numpy().astype(np.float32), trace


def build_graphsage_embeddings(
    graph: nx.DiGraph,
    dim: int = 128,
    layers: int = 2,
    residual: float = 0.0,
    epochs: int = 200,
    learning_rate: float = 0.001,
    negative_samples: int = 5,
    num_neighbors_1: int = 25,
    num_neighbors_2: int = 10,
    feature_mode: str = "degree",
    seed: int = 42,
    root_qid: str | None = None,
    undirected: bool = False,
) -> EmbeddingStore:
    """Train an unsupervised GraphSAGE embedding. / 自己教師あり GraphSAGE を学習する。"""

    qids = sorted(str(node) for node in graph.nodes())
    embeddings, trace = _train_graphsage_embeddings(
        graph=graph,
        qids=qids,
        dim=dim,
        layers=layers,
        residual=residual,
        epochs=epochs,
        learning_rate=learning_rate,
        negative_samples=negative_samples,
        num_neighbors_1=num_neighbors_1,
        num_neighbors_2=num_neighbors_2,
        feature_mode=feature_mode,
        seed=seed,
        undirected=undirected,
    )
    metadata = {
        "algorithm": "graphsage",
        "implementation": "unsupervised_graphsage_negative_sampling",
        "created_at_utc": _timestamp_utc(),
        "graph_type": graph.graph.get("graph_type"),
        "root_qid": graph.graph.get("root_qid"),
        "relation": "parent_taxon",
        "objective": "link_prediction",
        "parameters": {
            "dim": dim,
            "layers": layers,
            "residual": residual,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "negative_samples": negative_samples,
            "num_neighbors": [num_neighbors_1, num_neighbors_2],
            "feature_mode": feature_mode,
            "seed": seed,
            "root_qid": root_qid,
            "undirected": undirected,
        },
        "training_trace": trace,
    }
    return EmbeddingStore(qids=qids, embeddings=embeddings.astype(np.float32), metadata=metadata)


def build_grace_embeddings(
    graph: nx.DiGraph,
    dim: int = 128,
    proj_dim: int | None = 128,
    layers: int = 2,
    residual: float = 0.0,
    epochs: int = 200,
    learning_rate: float = 0.001,
    negative_samples: int = 5,
    tau: float = 0.5,
    drop_edge_rate_1: float = 0.2,
    drop_edge_rate_2: float = 0.4,
    drop_feature_rate_1: float = 0.0,
    drop_feature_rate_2: float = 0.0,
    batch_size: int = 256,
    encoder_type: str = "gcn",
    feature_mode: str = "degree",
    weight_decay: float = 1e-5,
    seed: int = 42,
    root_qid: str | None = None,
    undirected: bool = False,
    device: str = "auto",
) -> EmbeddingStore:
    """Train a GRACE-style contrastive embedding. / GRACE 風 contrastive 埋め込みを学習する。"""

    torch_mod, nn_mod, F_mod = _require_torch()
    rng = np.random.default_rng(seed)
    qids = sorted(str(node) for node in graph.nodes())
    node_to_index, neighbors, _, edge_set, _ = _build_torch_graph_data(graph, qids=qids, undirected=undirected)
    edge_pairs = [
        (node_to_index[str(source)], node_to_index[str(target)])
        for source, target in graph.edges()
        if str(source) in node_to_index and str(target) in node_to_index
    ]
    features = _build_structural_features(graph, qids=qids, feature_mode=feature_mode, seed=seed, dim=dim)
    projector_dim = int(proj_dim if proj_dim is not None else dim)
    device_obj, resolved_device = _resolve_torch_device(device, context="grace")

    if not qids:
        empty = np.zeros((0, dim), dtype=np.float32)
        return EmbeddingStore(
            qids=[],
            embeddings=empty,
            metadata={
                "algorithm": "grace",
                "implementation": "official_model_based_grace_baseline",
                "created_at_utc": _timestamp_utc(),
                "graph_type": graph.graph.get("graph_type"),
                "root_qid": graph.graph.get("root_qid"),
                "objective": "contrastive_learning",
                "parameters": {
                    "dim": dim,
                    "proj_dim": projector_dim,
                    "layers": layers,
                    "residual": residual,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "tau": tau,
                    "drop_edge_rate_1": drop_edge_rate_1,
                    "drop_edge_rate_2": drop_edge_rate_2,
                    "drop_feature_rate_1": drop_feature_rate_1,
                    "drop_feature_rate_2": drop_feature_rate_2,
                    "batch_size": batch_size,
                    "encoder_type": encoder_type,
                    "feature_mode": feature_mode,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "root_qid": root_qid,
                    "undirected": undirected,
                    "device": device,
                    "resolved_device": resolved_device,
                },
                "training_trace": [],
            },
        )

    base_features = torch_mod.tensor(features, dtype=torch_mod.float32, device=device_obj)
    base_adjacency = _sparse_adjacency_from_edges(len(qids), edge_pairs, device=device_obj)

    class GraceEncoder(nn_mod.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_dim = dim
            self.input = nn_mod.Linear(int(features.shape[1]), dim)
            self.encoder_type = encoder_type.strip().lower()
            if self.encoder_type == "graphsage":
                self.layers = nn_mod.ModuleList([_GraphSAGELayer(dim) for _ in range(max(layers - 1, 0))])
            else:
                self.layers = nn_mod.ModuleList([_GCNConv(dim, dim) for _ in range(max(layers - 1, 0))])
            self.projector = nn_mod.Sequential(
                nn_mod.Linear(dim, projector_dim),
                nn_mod.ReLU(),
                nn_mod.Linear(projector_dim, projector_dim),
            )

        def encode(self, x: Any, adjacency: Any) -> Any:
            h = F_mod.relu(self.input(x))
            h = F_mod.normalize(h, p=2, dim=1)
            for layer in self.layers:
                if self.encoder_type == "graphsage":
                    aggregated = layer(h, neighbors)
                else:
                    aggregated = layer(h, adjacency)
                h = residual * h + (1.0 - residual) * aggregated
                h = F_mod.relu(h)
                h = F_mod.normalize(h, p=2, dim=1)
            return h

        def project(self, h: Any) -> Any:
            return self.projector(h)

    encoder = GraceEncoder().to(device_obj)
    optimizer = torch_mod.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trace: list[dict[str, float]] = []
    final_embeddings = torch_mod.zeros((len(qids), dim), dtype=torch_mod.float32, device=device_obj)
    effective_batch_size = max(2, min(int(max(batch_size, 1)), len(qids)))

    for epoch in range(epochs):
        encoder.train()
        optimizer.zero_grad()

        view1_features = _drop_feature_columns(base_features, drop_feature_rate_1, rng)
        view2_features = _drop_feature_columns(base_features, drop_feature_rate_2, rng)
        view1_edges = _drop_edge_pairs(edge_pairs, drop_edge_rate_1, rng)
        view2_edges = _drop_edge_pairs(edge_pairs, drop_edge_rate_2, rng)
        view1_adjacency = _sparse_adjacency_from_edges(len(qids), view1_edges, device=device_obj)
        view2_adjacency = _sparse_adjacency_from_edges(len(qids), view2_edges, device=device_obj)

        h1 = encoder.encode(view1_features, view1_adjacency)
        h2 = encoder.encode(view2_features, view2_adjacency)
        z1 = F_mod.normalize(encoder.project(h1), p=2, dim=1)
        z2 = F_mod.normalize(encoder.project(h2), p=2, dim=1)

        permutation = rng.permutation(len(qids))
        batch_count = 0
        node_count = 0
        contrastive_losses: list[Any] = []
        batch_progress_interval = max(len(permutation) // max(effective_batch_size * 4, 1), 1)
        last_progress_time = time.monotonic()
        _render_progress_line(
            f"grace epoch {epoch + 1}/{epochs} | batches 0 | nodes 0/{len(qids)} | loss pending"
        )
        for start in range(0, len(permutation), effective_batch_size):
            batch_ids = permutation[start : start + effective_batch_size]
            if len(batch_ids) < 2:
                continue
            batch_index = torch_mod.tensor(batch_ids, dtype=torch_mod.long, device=device_obj)
            batch_loss = _grace_contrastive_loss(
                z1.index_select(0, batch_index),
                z2.index_select(0, batch_index),
                tau=tau,
                batch_size=effective_batch_size,
            )
            contrastive_losses.append(batch_loss)
            batch_count += 1
            node_count += len(batch_ids)
            now = time.monotonic()
            if batch_count % batch_progress_interval == 0 or now - last_progress_time >= 2.0:
                _render_progress_line(
                    f"grace epoch {epoch + 1}/{epochs} | batches {batch_count} | nodes {node_count}/{len(qids)} | "
                    f"loss pending"
                )
                last_progress_time = now

        if batch_count == 0:
            continue

        total_loss = torch_mod.stack(contrastive_losses).mean()
        total_loss.backward()
        grad_norm = _gradient_norm(encoder.parameters())
        optimizer.step()

        with torch_mod.no_grad():
            encoder.eval()
            final_embeddings = encoder.encode(base_features, base_adjacency)
            total_loss_value = float(total_loss.detach().cpu().item())
            pos_similarity = float(torch_mod.sum(F_mod.normalize(z1, p=2, dim=1) * F_mod.normalize(z2, p=2, dim=1), dim=1).mean().cpu().item())
            neg_inter = float(torch_mod.sum(F_mod.normalize(z1, p=2, dim=1) * F_mod.normalize(torch_mod.roll(z2, shifts=1, dims=0), p=2, dim=1), dim=1).mean().cpu().item())
            neg_intra = float(torch_mod.sum(F_mod.normalize(z1, p=2, dim=1) * F_mod.normalize(torch_mod.roll(z1, shifts=1, dims=0), p=2, dim=1), dim=1).mean().cpu().item())
        trace.append(
            {
                "epoch": float(epoch + 1),
                "batch_count": float(batch_count),
                "node_count": float(node_count),
                "train_loss": total_loss_value,
                "contrastive_loss": total_loss_value,
                "average_loss": total_loss_value,
                "positive_similarity_mean": pos_similarity,
                "negative_inter_similarity_mean": neg_inter,
                "negative_intra_similarity_mean": neg_intra,
                "positive_negative_gap": float(pos_similarity - max(neg_inter, neg_intra)),
                "embedding_norm_mean": float(torch_mod.linalg.norm(final_embeddings, dim=1).mean().cpu().item()),
                "embedding_norm_std": float(torch_mod.linalg.norm(final_embeddings, dim=1).std(unbiased=False).cpu().item()),
                "embedding_variance_mean": float(final_embeddings.var(dim=0, unbiased=False).mean().cpu().item()),
                "gradient_norm": float(grad_norm),
                "learning_rate": float(learning_rate),
                "tau": float(tau),
                "drop_edge_rate_1": float(drop_edge_rate_1),
                "drop_edge_rate_2": float(drop_edge_rate_2),
                "drop_feature_rate_1": float(drop_feature_rate_1),
                "drop_feature_rate_2": float(drop_feature_rate_2),
            }
        )
        _finish_progress_line(
            f"grace epoch {epoch + 1}/{epochs} done | batches {batch_count} | loss {total_loss_value:.4f}"
        )

    embeddings = final_embeddings.detach().cpu().numpy().astype(np.float32)
    metadata = {
        "algorithm": "grace",
        "implementation": "official_model_based_grace_baseline",
        "created_at_utc": _timestamp_utc(),
        "graph_type": graph.graph.get("graph_type"),
        "root_qid": graph.graph.get("root_qid"),
        "objective": "contrastive_learning",
        "parameters": {
            "dim": dim,
            "proj_dim": projector_dim,
            "layers": layers,
            "residual": residual,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "negative_samples": negative_samples,
            "tau": tau,
            "drop_edge_rate_1": drop_edge_rate_1,
            "drop_edge_rate_2": drop_edge_rate_2,
            "drop_feature_rate_1": drop_feature_rate_1,
            "drop_feature_rate_2": drop_feature_rate_2,
            "batch_size": batch_size,
            "encoder_type": encoder_type,
            "feature_mode": feature_mode,
            "weight_decay": weight_decay,
            "seed": seed,
            "root_qid": root_qid,
            "undirected": undirected,
            "device": device,
        },
        "training_trace": trace,
    }
    return EmbeddingStore(qids=qids, embeddings=embeddings.astype(np.float32), metadata=metadata)


def _transe_negative_sample(
    rng: np.random.Generator,
    node_count: int,
    head_index: int,
    tail_index: int,
) -> tuple[int, int]:
    if node_count <= 2:
        return head_index, tail_index
    corrupt_head = bool(rng.integers(0, 2))
    if corrupt_head:
        while True:
            candidate = int(rng.integers(0, node_count))
            if candidate != head_index and candidate != tail_index:
                return candidate, tail_index
    while True:
        candidate = int(rng.integers(0, node_count))
        if candidate != head_index and candidate != tail_index:
            return head_index, candidate


def build_transe_embeddings(
    graph: nx.DiGraph,
    dim: int = 128,
    epochs: int = 200,
    learning_rate: float = 0.001,
    margin: float = 1.0,
    negative_samples: int = 10,
    p_norm: int = 1,
    weight_decay: float = 1e-5,
    seed: int = 42,
    root_qid: str | None = None,
) -> EmbeddingStore:
    """Train a TransE-style embedding on graph edges. / graph の辺で TransE 風埋め込みを学習する。"""

    qids = sorted(str(node) for node in graph.nodes())
    if not qids:
        empty = np.zeros((0, dim), dtype=np.float32)
        return EmbeddingStore(
            qids=[],
            embeddings=empty,
            metadata={
                "algorithm": "transe",
                "implementation": "knowledge_graph_embedding_baseline",
                "created_at_utc": _timestamp_utc(),
                "graph_type": graph.graph.get("graph_type"),
                "root_qid": graph.graph.get("root_qid"),
                "parameters": {
                    "dim": dim,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "margin": margin,
                    "negative_samples": negative_samples,
                    "p_norm": p_norm,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "root_qid": root_qid,
                },
            },
        )

    node_to_index = {qid: index for index, qid in enumerate(qids)}
    edges = [(str(source), str(target)) for source, target in graph.edges()]
    if not edges:
        embeddings = _initial_entity_embeddings(qids=qids, dim=dim, seed=seed)
        metadata = {
            "algorithm": "transe",
            "implementation": "knowledge_graph_embedding_baseline",
            "created_at_utc": _timestamp_utc(),
            "graph_type": graph.graph.get("graph_type"),
            "root_qid": graph.graph.get("root_qid"),
                "parameters": {
                    "dim": dim,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "margin": margin,
                    "negative_samples": negative_samples,
                    "p_norm": p_norm,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "root_qid": root_qid,
                },
            }
        return EmbeddingStore(qids=qids, embeddings=embeddings, metadata=metadata)

    rng = np.random.default_rng(seed)
    entity_vectors = _initial_entity_embeddings(qids=qids, dim=dim, seed=seed).astype(np.float32, copy=True)
    relation_vector = rng.normal(0.0, 0.1, size=dim).astype(np.float32)
    edge_indices = [(node_to_index[source], node_to_index[target]) for source, target in edges if source in node_to_index and target in node_to_index]
    eps = 1e-12
    trace: list[dict[str, float]] = []

    def _distance_and_grad(diff: np.ndarray) -> tuple[float, np.ndarray]:
        if int(p_norm) == 1:
            distance = float(np.abs(diff).sum())
            grad = np.sign(diff).astype(np.float32)
        else:
            distance = float(np.linalg.norm(diff))
            grad = (diff / max(distance, eps)).astype(np.float32)
        return distance, grad

    for epoch in range(epochs):
        rng.shuffle(edge_indices)
        total_loss = 0.0
        margin_hits = 0
        positive_distances: list[float] = []
        negative_distances: list[float] = []
        edge_progress_interval = max(len(edge_indices) // 20, 1)
        last_progress_time = time.monotonic()
        _render_progress_line(f"transe epoch {epoch + 1}/{epochs} | edges 0/{len(edge_indices)} | loss 0.0000")
        for edge_index, (head_index, tail_index) in enumerate(edge_indices, start=1):
            head_vector = entity_vectors[head_index].copy()
            tail_vector = entity_vectors[tail_index].copy()
            positive_diff = head_vector + relation_vector - tail_vector
            positive_distance, positive_grad = _distance_and_grad(positive_diff)
            positive_distances.append(positive_distance)

            for _ in range(negative_samples):
                negative_head_index, negative_tail_index = _transe_negative_sample(
                    rng=rng,
                    node_count=len(qids),
                    head_index=head_index,
                    tail_index=tail_index,
                )
                if negative_head_index == head_index and negative_tail_index == tail_index:
                    continue
                negative_head = entity_vectors[negative_head_index].copy()
                negative_tail = entity_vectors[negative_tail_index].copy()
                negative_diff = negative_head + relation_vector - negative_tail
                negative_distance, negative_grad = _distance_and_grad(negative_diff)
                negative_distances.append(negative_distance)
                hinge = margin + positive_distance - negative_distance
                if hinge <= 0:
                    continue
                total_loss += hinge
                margin_hits += 1

                entity_vectors[head_index] -= learning_rate * positive_grad
                relation_vector -= learning_rate * positive_grad
                entity_vectors[tail_index] += learning_rate * positive_grad

                entity_vectors[negative_head_index] += learning_rate * negative_grad
                relation_vector += learning_rate * negative_grad
                entity_vectors[negative_tail_index] -= learning_rate * negative_grad
                if weight_decay > 0.0:
                    decay = max(0.0, 1.0 - learning_rate * weight_decay)
                    entity_vectors[head_index] *= decay
                    entity_vectors[tail_index] *= decay
                    entity_vectors[negative_head_index] *= decay
                    entity_vectors[negative_tail_index] *= decay
                    relation_vector *= decay
            now = time.monotonic()
            if edge_index % edge_progress_interval == 0 or now - last_progress_time >= 2.0:
                _render_progress_line(
                    f"transe epoch {epoch + 1}/{epochs} | edges {edge_index}/{len(edge_indices)} | "
                    f"margin_hits {margin_hits} | loss {total_loss:.4f}"
                )
                last_progress_time = now

        entity_vectors = _normalize_rows(entity_vectors)
        relation_norm = float(np.linalg.norm(relation_vector))
        if relation_norm > eps:
            relation_vector = relation_vector / relation_norm
        trace.append(
            {
                "epoch": float(epoch + 1),
                "margin_hits": float(margin_hits),
                "average_loss": float(total_loss / max(margin_hits, 1)),
                "positive_distance_mean": float(np.mean(positive_distances)) if positive_distances else 0.0,
                "negative_distance_mean": float(np.mean(negative_distances)) if negative_distances else 0.0,
                "margin_gap": float((np.mean(negative_distances) - np.mean(positive_distances)))
                if positive_distances and negative_distances
                else 0.0,
                "entity_embedding_norm_mean": float(np.linalg.norm(entity_vectors, axis=1).mean()),
                "relation_embedding_norm_mean": float(np.linalg.norm(relation_vector)),
                "embedding_variance_mean": float(entity_vectors.var(axis=0).mean()),
                "learning_rate": float(learning_rate),
                "p_norm": float(p_norm),
                "weight_decay": float(weight_decay),
            }
        )
        _finish_progress_line(
            f"transe epoch {epoch + 1}/{epochs} done | edges {len(edge_indices)}/{len(edge_indices)} | "
            f"loss {total_loss:.4f}"
        )

    metadata = {
        "algorithm": "transe",
        "implementation": "knowledge_graph_embedding_baseline",
        "created_at_utc": _timestamp_utc(),
        "graph_type": graph.graph.get("graph_type"),
        "root_qid": graph.graph.get("root_qid"),
        "relation": "parent_taxon",
        "parameters": {
            "dim": dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "margin": margin,
            "negative_samples": negative_samples,
            "p_norm": p_norm,
            "weight_decay": weight_decay,
            "seed": seed,
            "root_qid": root_qid,
        },
        "training_trace": trace,
    }
    return EmbeddingStore(qids=qids, embeddings=entity_vectors.astype(np.float32), metadata=metadata)


def build_embeddings(graph: nx.DiGraph, algorithm: str, **kwargs: Any) -> EmbeddingStore:
    """Dispatch to the requested embedding algorithm. / 指定アルゴリズムへ振り分ける。"""

    normalized = algorithm.strip().lower()
    if normalized == "node2vec":
        return build_node2vec_embeddings(graph, **kwargs)
    if normalized == "gcn":
        return build_gcn_embeddings(graph, **kwargs)
    if normalized == "graphsage":
        return build_graphsage_embeddings(graph, **kwargs)
    if normalized == "grace":
        return build_grace_embeddings(graph, **kwargs)
    if normalized == "transe":
        return build_transe_embeddings(graph, **kwargs)
    raise ValueError(f"Unsupported embedding algorithm: {algorithm}")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for graph embeddings. / graph 埋め込み用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build graph embeddings from a taxonomy NetworkX graph PKL.")
    parser.add_argument("--input", default=str(paths.taxonomy_graph_pkl))
    parser.add_argument("--output-dir", default=str(paths.graph_embeddings_dir))
    parser.add_argument(
        "--algorithm",
        choices=["node2vec", "gcn", "grace", "transe", "graphsage"],
        default="node2vec",
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-features", choices=["degree", "one_hot", "constant", "random"], default="degree")

    parser.add_argument("--walk-length", type=int, default=40)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--transe-negative-samples", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--undirected", action="store_true", default=False)

    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--residual", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--drop-edge-rate-1", type=float, default=0.2)
    parser.add_argument("--drop-edge-rate-2", type=float, default=0.4)
    parser.add_argument("--drop-feature-rate-1", type=float, default=0.0)
    parser.add_argument("--drop-feature-rate-2", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grace-encoder", choices=["gcn", "graphsage"], default="gcn")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--graphsage-num-neighbors-1", type=int, default=25)
    parser.add_argument("--graphsage-num-neighbors-2", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--p-norm", type=int, default=1)
    parser.add_argument("--root-qid", default=None)
    return parser


def _write_summary(store: EmbeddingStore, output_dir: Path) -> None:
    summary = {
        "qid_count": len(store.qids),
        "dimension": store.dim,
        "metadata": store.metadata,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the graph embedding command. / graph 埋め込みコマンドを実行する。"""

    args = build_parser().parse_args(argv)
    graph = load_graph(Path(args.input))
    output_root = Path(args.output_dir)
    run_tag = _run_timestamp_mmddhhmm()

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
        node2vec_dir = output_root / "node2vec" / run_tag
        save_embedding_store(node2vec_store, node2vec_dir)
        _write_summary(node2vec_store, node2vec_dir)

    if args.algorithm == "gcn":
        gcn_store = build_gcn_embeddings(
            graph,
            layers=args.layers,
            residual=args.residual,
            feature_mode=args.initial_features,
            weight_decay=args.weight_decay,
            undirected=args.undirected,
            root_qid=args.root_qid,
            **common_kwargs,
        )
        gcn_dir = output_root / "gcn" / run_tag
        save_embedding_store(gcn_store, gcn_dir)
        _write_summary(gcn_store, gcn_dir)

    if args.algorithm == "graphsage":
        graphsage_store = build_graphsage_embeddings(
            graph,
            layers=args.layers,
            residual=args.residual,
            num_neighbors_1=args.graphsage_num_neighbors_1,
            num_neighbors_2=args.graphsage_num_neighbors_2,
            feature_mode=args.initial_features,
            undirected=args.undirected,
            root_qid=args.root_qid,
            **common_kwargs,
        )
        graphsage_dir = output_root / "graphsage" / run_tag
        save_embedding_store(graphsage_store, graphsage_dir)
        _write_summary(graphsage_store, graphsage_dir)

    if args.algorithm == "grace":
        grace_store = build_grace_embeddings(
            graph,
            layers=args.layers,
            residual=args.residual,
            proj_dim=args.proj_dim,
            tau=args.tau,
            drop_edge_rate_1=args.drop_edge_rate_1,
            drop_edge_rate_2=args.drop_edge_rate_2,
            drop_feature_rate_1=args.drop_feature_rate_1,
            drop_feature_rate_2=args.drop_feature_rate_2,
            batch_size=args.batch_size,
            encoder_type=args.grace_encoder,
            feature_mode=args.initial_features,
            weight_decay=args.weight_decay,
            undirected=args.undirected,
            root_qid=args.root_qid,
            device=args.device,
            **common_kwargs,
        )
        grace_dir = output_root / "grace" / run_tag
        save_embedding_store(grace_store, grace_dir)
        _write_summary(grace_store, grace_dir)

    if args.algorithm == "transe":
        transe_store = build_transe_embeddings(
            graph,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            negative_samples=args.transe_negative_samples,
            p_norm=args.p_norm,
            weight_decay=args.weight_decay,
            root_qid=args.root_qid,
            dim=args.dim,
            seed=args.seed,
        )
        transe_dir = output_root / "transe" / run_tag
        save_embedding_store(transe_store, transe_dir)
        _write_summary(transe_store, transe_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
