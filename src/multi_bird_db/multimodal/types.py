from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddingSourceConfig:
    """Exact embedding directories used by one multimodal experiment run."""

    graph_embedding_dir: Path
    audio_embedding_dir: Path
    language_embedding_dir: Path
    graph_model_name: str = "graphsage"
    audio_model_name: str = "facebook/wav2vec2-base-960h"
    language_model_name: str = "google-bert/bert-base-uncased"
    language_code: str = "en"


@dataclass(frozen=True, slots=True)
class GraphEmbeddingRun:
    """Loaded graph embeddings with 1 vector per QID."""

    run_dir: Path
    qids: list[str]
    embeddings: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Graph embeddings must be 2D, got shape {self.embeddings.shape}")
        if len(self.qids) != int(self.embeddings.shape[0]):
            raise ValueError("Graph embedding rows and qids length must match")


@dataclass(frozen=True, slots=True)
class AudioEmbeddingRow:
    """One audio embedding row aligned with the stored matrix."""

    embedding_index: int
    audio_id: str
    qid: str
    source_path: str
    relative_path: str
    window_index: int
    manifest_row: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AudioEmbeddingRun:
    """Loaded audio embeddings with row-level manifest data."""

    run_dir: Path
    audio_ids: list[str]
    qids: list[str]
    embeddings: np.ndarray
    rows: list[AudioEmbeddingRow]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Audio embeddings must be 2D, got shape {self.embeddings.shape}")
        row_count = int(self.embeddings.shape[0])
        if len(self.audio_ids) != row_count or len(self.qids) != row_count or len(self.rows) != row_count:
            raise ValueError("Audio embeddings, ids, qids, and manifest rows must have the same length")


@dataclass(frozen=True, slots=True)
class LanguageSurfaceRow:
    """One language embedding row aligned with the stored matrix."""

    embedding_index: int
    surface_id: str
    qid: str
    language: str
    surface_text: str = ""
    source: str = ""
    manifest_row: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LanguageEmbeddingRun:
    """Loaded language embeddings with row-level surface metadata."""

    run_dir: Path
    surface_ids: list[str]
    qids: list[str]
    embeddings: np.ndarray
    rows: list[LanguageSurfaceRow]
    qid_to_surfaces: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Language embeddings must be 2D, got shape {self.embeddings.shape}")
        row_count = int(self.embeddings.shape[0])
        if len(self.surface_ids) != row_count or len(self.qids) != row_count or len(self.rows) != row_count:
            raise ValueError("Language embeddings, ids, qids, and manifest rows must have the same length")


@dataclass(frozen=True, slots=True)
class MultimodalSampleRow:
    """One expanded multimodal sample with source-row lineage."""

    sample_id: str
    qid: str
    graph_embedding_index: int | None
    audio_embedding_index: int | None
    language_embedding_index: int | None
    modality_pattern: str
    target_rank: str
    target_label: str


@dataclass(frozen=True, slots=True)
class MultimodalFeatureMatrix:
    """Row-aligned multimodal feature matrix and its sample manifest."""

    rows: list[MultimodalSampleRow]
    embeddings: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Multimodal embeddings must be 2D, got shape {self.embeddings.shape}")
        if len(self.rows) != int(self.embeddings.shape[0]):
            raise ValueError("Multimodal feature rows and embeddings row count must match")
