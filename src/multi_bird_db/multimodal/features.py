from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import MultimodalFeatureMatrix, MultimodalSampleRow


@dataclass(frozen=True, slots=True)
class FeatureAssemblyResult:
    """Container returned by feature concatenation helpers."""

    matrix: MultimodalFeatureMatrix
    feature_slices: dict[str, tuple[int, int]]


def concatenate_sample_vectors(
    *,
    graph_vector: np.ndarray | None = None,
    audio_vector: np.ndarray | None = None,
    language_vector: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Concatenate available modality vectors in graph-audio-language order."""

    parts: list[np.ndarray] = []
    feature_slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, vector in (
        ("graph", graph_vector),
        ("audio", audio_vector),
        ("language", language_vector),
    ):
        if vector is None:
            continue
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        start = offset
        offset += int(array.shape[0])
        feature_slices[name] = (start, offset)
        parts.append(array)
    if not parts:
        raise ValueError("At least one modality vector is required to build a multimodal sample.")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False), feature_slices


def build_feature_matrix(
    rows_and_vectors: list[tuple[MultimodalSampleRow, np.ndarray]],
    *,
    metadata: dict[str, Any] | None = None,
) -> FeatureAssemblyResult:
    """Stack row-aligned sample vectors into one 2D feature matrix."""

    if not rows_and_vectors:
        raise ValueError("At least one multimodal sample is required.")
    rows = [row for row, _vector in rows_and_vectors]
    vectors = [np.asarray(vector, dtype=np.float32).reshape(-1) for _row, vector in rows_and_vectors]
    reference_dim = int(vectors[0].shape[0])
    for vector in vectors[1:]:
        if int(vector.shape[0]) != reference_dim:
            raise ValueError("All multimodal sample vectors must have the same dimensionality.")
    matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    payload = dict(metadata or {})
    payload.setdefault("row_count", len(rows))
    payload.setdefault("embedding_dim", int(matrix.shape[1]))
    return FeatureAssemblyResult(
        matrix=MultimodalFeatureMatrix(rows=rows, embeddings=matrix, metadata=payload),
        feature_slices={},
    )
