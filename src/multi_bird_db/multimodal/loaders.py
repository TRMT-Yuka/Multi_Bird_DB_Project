from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_project_paths
from .types import (
    AudioEmbeddingRow,
    AudioEmbeddingRun,
    EmbeddingSourceConfig,
    GraphEmbeddingRun,
    LanguageEmbeddingRun,
    LanguageSurfaceRow,
)

DEFAULT_GRAPH_MODEL = "graphsage"
DEFAULT_AUDIO_MODEL = "facebook/wav2vec2-base-960h"
DEFAULT_LANGUAGE_MODEL = "google-bert/bert-base-uncased"
DEFAULT_LANGUAGE_CODE = "en"
SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str) -> str:
    normalized = SAFE_COMPONENT_RE.sub("_", value.strip())
    normalized = normalized.strip("._")
    return normalized or "item"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


def _latest_directory(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Embedding directory root does not exist: {root}")
    children = sorted((child for child in root.iterdir() if child.is_dir()), key=lambda item: item.name)
    if not children:
        raise FileNotFoundError(f"No embedding run directories found under: {root}")
    return children[-1]


def find_latest_graph_embedding_dir(root: Path | None = None) -> Path:
    paths = get_project_paths()
    return _latest_directory(root or (paths.graph_embeddings_dir / DEFAULT_GRAPH_MODEL))


def find_latest_audio_embedding_dir(root: Path | None = None) -> Path:
    paths = get_project_paths()
    model_root = paths.audio_embeddings_dir / "wav2vec2" / _safe_component(DEFAULT_AUDIO_MODEL)
    return _latest_directory(root or model_root)


def find_latest_language_embedding_dir(root: Path | None = None) -> Path:
    paths = get_project_paths()
    language_dir = root or (paths.embeddings_dir / "language" / DEFAULT_LANGUAGE_CODE)
    if not language_dir.exists():
        raise FileNotFoundError(f"Language embedding directory does not exist: {language_dir}")
    return language_dir


def resolve_default_source_config(
    *,
    graph_embedding_dir: Path | None = None,
    audio_embedding_dir: Path | None = None,
    language_embedding_dir: Path | None = None,
) -> EmbeddingSourceConfig:
    return EmbeddingSourceConfig(
        graph_embedding_dir=graph_embedding_dir or find_latest_graph_embedding_dir(),
        audio_embedding_dir=audio_embedding_dir or find_latest_audio_embedding_dir(),
        language_embedding_dir=language_embedding_dir or find_latest_language_embedding_dir(),
        graph_model_name=DEFAULT_GRAPH_MODEL,
        audio_model_name=DEFAULT_AUDIO_MODEL,
        language_model_name=DEFAULT_LANGUAGE_MODEL,
        language_code=DEFAULT_LANGUAGE_CODE,
    )


def load_graph_embedding_run(run_dir: Path) -> GraphEmbeddingRun:
    qids = list(_read_json(_require_file(run_dir / "qids.json")))
    embeddings = np.asarray(np.load(_require_file(run_dir / "embeddings.npy")), dtype=np.float32)
    metadata = dict(_read_json(_require_file(run_dir / "metadata.json")))
    return GraphEmbeddingRun(run_dir=run_dir, qids=qids, embeddings=embeddings, metadata=metadata)


def load_audio_embedding_run(run_dir: Path) -> AudioEmbeddingRun:
    audio_ids = list(_read_json(_require_file(run_dir / "audio_ids.json")))
    qids = list(_read_json(_require_file(run_dir / "qids.json")))
    embeddings = np.asarray(np.load(_require_file(run_dir / "embeddings.npy")), dtype=np.float32)
    metadata = dict(_read_json(_require_file(run_dir / "metadata.json")))
    manifest_rows = _read_tsv(_require_file(run_dir / "audio_manifest.tsv"))
    rows: list[AudioEmbeddingRow] = []
    for index, row in enumerate(manifest_rows):
        audio_id = str(row.get("audio_id") or audio_ids[index] if index < len(audio_ids) else "").strip()
        qid = str(row.get("qid") or qids[index] if index < len(qids) else "").strip()
        rows.append(
            AudioEmbeddingRow(
                embedding_index=index,
                audio_id=audio_id,
                qid=qid,
                source_path=str(row.get("source_path") or ""),
                relative_path=str(row.get("relative_path") or ""),
                window_index=int(str(row.get("window_index") or "0") or "0"),
                manifest_row={key: str(value) for key, value in row.items()},
            )
        )
    return AudioEmbeddingRun(
        run_dir=run_dir,
        audio_ids=audio_ids,
        qids=qids,
        embeddings=embeddings,
        rows=rows,
        metadata=metadata,
    )


def load_language_embedding_run(run_dir: Path) -> LanguageEmbeddingRun:
    surface_ids = list(_read_json(_require_file(run_dir / "surface_ids.json")))
    qids = list(_read_json(_require_file(run_dir / "qids.json")))
    embeddings = np.asarray(np.load(_require_file(run_dir / "embeddings.npy")), dtype=np.float32)
    qid_to_surfaces = dict(_read_json(_require_file(run_dir / "qid_to_surfaces.json")))
    metadata = dict(_read_json(_require_file(run_dir / "metadata.json")))
    manifest_path = run_dir / "surface_manifest.tsv"
    manifest_rows = _read_tsv(manifest_path) if manifest_path.exists() else []
    manifest_by_surface_id = {
        str(row.get("surface_id") or "").strip(): {key: str(value) for key, value in row.items()}
        for row in manifest_rows
    }
    rows: list[LanguageSurfaceRow] = []
    default_language = str(metadata.get("language") or DEFAULT_LANGUAGE_CODE)
    for index, surface_id in enumerate(surface_ids):
        manifest_row = manifest_by_surface_id.get(str(surface_id), {})
        rows.append(
            LanguageSurfaceRow(
                embedding_index=index,
                surface_id=str(surface_id),
                qid=str(qids[index]),
                language=str(manifest_row.get("language") or default_language),
                surface_text=str(manifest_row.get("surface_text") or ""),
                source=str(manifest_row.get("source") or ""),
                manifest_row=manifest_row,
            )
        )
    return LanguageEmbeddingRun(
        run_dir=run_dir,
        surface_ids=surface_ids,
        qids=qids,
        embeddings=embeddings,
        rows=rows,
        qid_to_surfaces={str(key): [str(item) for item in value] for key, value in qid_to_surfaces.items()},
        metadata=metadata,
    )
