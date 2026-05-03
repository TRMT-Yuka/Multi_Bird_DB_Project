from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .config import get_project_paths


LANGUAGE_SOURCE_FIELDS: dict[str, tuple[str, str]] = {
    "en": ("en_name", "enwiki_title"),
    "ja": ("ja_name", "jawiki_title"),
}


@dataclass(frozen=True, slots=True)
class SurfaceEntry:
    """One language surface candidate. / 1 件の言語 surface 候補。"""

    qid: str
    language: str
    ordinal: int
    surface_text: str
    source: str
    source_index: int

    @property
    def surface_id(self) -> str:
        return f"{self.qid}_{self.language}_{self.ordinal}"


class TextEncoder(Protocol):
    """Callable encoder interface used by language embedding backends. / 言語埋め込み用のエンコーダ接口。"""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into a 2D float array. / テキスト列を 2D 配列へ変換する。"""


@dataclass(frozen=True, slots=True)
class LanguageEmbeddingStore:
    """Store language embeddings and lookup metadata. / 言語埋め込みと参照メタデータを持つ。"""

    surface_ids: list[str]
    qids: list[str]
    embeddings: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {self.embeddings.shape}")
        if self.embeddings.shape[0] != len(self.surface_ids) or len(self.surface_ids) != len(self.qids):
            raise ValueError("surface_ids, qids, and embeddings row count must match")

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


@dataclass(frozen=True, slots=True)
class SurfaceManifestBundle:
    """In-memory grouped surface manifests for one output root. / 1 出力 root の surface 一覧。"""

    entries_by_language: dict[str, list[SurfaceEntry]]


def probe_cuda() -> dict[str, Any]:
    """Return a small CUDA availability report. / CUDA 利用可否の簡易レポートを返す。"""

    report: dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "device_count": 0,
        "torch_cuda_version": None,
        "cudnn_version": None,
        "error": None,
    }
    try:
        import torch  # type: ignore[import-not-found]

        report["torch_available"] = True
        report["torch_cuda_version"] = torch.version.cuda
        report["cudnn_version"] = torch.backends.cudnn.version()
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["device_count"] = int(torch.cuda.device_count()) if report["cuda_available"] else 0
        if report["cuda_available"] and report["device_count"]:
            report["device_names"] = [torch.cuda.get_device_name(index) for index in range(report["device_count"])]
        else:
            report["device_names"] = []
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def load_ontology_rows(input_path: Path) -> list[dict[str, Any]]:
    """Load the ontology PKL and validate its shape. / ontology PKL を読み、形を検証する。"""

    if not input_path.exists():
        raise FileNotFoundError(f"Ontology PKL does not exist: {input_path}")
    with input_path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Ontology PKL must contain a list, got: {type(rows).__name__}")
    return rows


def _parse_aliases(value: Any) -> list[str]:
    """Parse an alias cell into a normalized list of strings. / 別名セルを正規化して list 化する。"""

    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:
            items = json.loads(value)
        except json.JSONDecodeError:
            items = [value]
    else:
        items = []

    normalized: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("value", "")).strip()
        else:
            text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _candidate_values(row: dict[str, Any], language: str) -> list[tuple[str, str, int]]:
    """Return ordered surface candidates for one row. / 1 行から順序付き surface 候補を返す。"""

    if language not in LANGUAGE_SOURCE_FIELDS:
        raise ValueError(f"Unsupported language: {language}")

    name_field, title_field = LANGUAGE_SOURCE_FIELDS[language]
    candidates: list[tuple[str, str, int]] = []

    name = str(row.get(name_field, "")).strip()
    if name:
        candidates.append((name, name_field, 0))

    aliases = _parse_aliases(row.get(f"{language}_aliases", "[]"))
    for index, alias in enumerate(aliases):
        candidates.append((alias, f"{language}_aliases", index + 1))

    title = str(row.get(title_field, "")).strip()
    if title:
        candidates.append((title, title_field, len(candidates) + 1))

    return candidates


def build_surface_entries(rows: list[dict[str, Any]], language: str) -> list[SurfaceEntry]:
    """Build deterministic surface entries for one language. / 1 言語分の surface 一覧を作る。"""

    grouped: list[tuple[str, str, str, int, str]] = []
    for row in rows:
        qid = str(row.get("qid") or row.get("id") or "").strip()
        if not qid:
            continue
        for surface_text, source, source_index in _candidate_values(row, language):
            grouped.append((qid, surface_text, source, source_index, language))

    grouped.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    entries: list[SurfaceEntry] = []
    ordinal_by_qid: dict[str, int] = {}
    for qid, surface_text, source, _source_index, lang in grouped:
        ordinal = ordinal_by_qid.get(qid, 0)
        entries.append(
            SurfaceEntry(
                qid=qid,
                language=lang,
                ordinal=ordinal,
                surface_text=surface_text,
                source=source,
                source_index=_source_index,
            )
        )
        ordinal_by_qid[qid] = ordinal + 1
    return entries


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically. / テキストを原子的に書き込む。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_tsv(path: Path, entries: list[SurfaceEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "surface_id",
                    "qid",
                    "language",
                    "ordinal",
                    "surface_text",
                    "source",
                    "source_index",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(
                    {
                        "surface_id": entry.surface_id,
                        "qid": entry.qid,
                        "language": entry.language,
                        "ordinal": entry.ordinal,
                        "surface_text": entry.surface_text,
                        "source": entry.source,
                        "source_index": entry.source_index,
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _write_embedding_array(path: Path, embeddings: np.ndarray) -> None:
    """Persist an embedding matrix. / 埋め込み行列を書き出す。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            np.save(handle, embeddings)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _build_qid_to_surfaces(entries: list[SurfaceEntry]) -> dict[str, list[dict[str, Any]]]:
    """Group surface entries by qid for easy reverse lookup. / qid ごとに surface をまとめる。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(entry.qid, []).append(
            {
                "surface_id": entry.surface_id,
                "surface_text": entry.surface_text,
                "language": entry.language,
                "ordinal": entry.ordinal,
                "source": entry.source,
                "source_index": entry.source_index,
            }
        )
    return grouped


def _group_entries_by_language(rows: list[dict[str, Any]]) -> dict[str, list[SurfaceEntry]]:
    """Build ordered surface entries for every supported language. / 対応言語ごとに surface 一覧を作る。"""

    return {language: build_surface_entries(rows, language) for language in sorted(LANGUAGE_SOURCE_FIELDS)}


def _write_language_surface_files(
    language_dir: Path,
    entries: list[SurfaceEntry],
    input_path: Path,
) -> dict[str, Any]:
    """Write manifest-side files for one language. / 1 言語分の manifest 系ファイルを書き出す。"""

    _write_tsv(language_dir / "surface_manifest.tsv", entries)
    _write_json(language_dir / "surface_ids.json", [entry.surface_id for entry in entries])
    _write_json(language_dir / "qids.json", [entry.qid for entry in entries])
    qid_to_surfaces = _build_qid_to_surfaces(entries)
    _write_json(language_dir / "qid_to_surfaces.json", qid_to_surfaces)
    source_counts: dict[str, int] = {}
    for entry in entries:
        source_counts[entry.source] = source_counts.get(entry.source, 0) + 1
    summary = {
        "kind": "surface_manifest",
        "language": entries[0].language if entries else language_dir.name,
        "created_at_utc": _timestamp_utc(),
        "item_count": len(entries),
        "unique_qid_count": len({entry.qid for entry in entries}),
        "source_counts": source_counts,
        "input_file": str(input_path),
        "output_files": {
            "surface_manifest_tsv": str(language_dir / "surface_manifest.tsv"),
            "surface_ids_json": str(language_dir / "surface_ids.json"),
            "qids_json": str(language_dir / "qids.json"),
            "qid_to_surfaces_json": str(language_dir / "qid_to_surfaces.json"),
        },
    }
    metadata = {
        "kind": "surface_manifest",
        "language": summary["language"],
        "created_at_utc": summary["created_at_utc"],
        "input_file": str(input_path),
        "output_dir": str(language_dir),
        "source_counts": source_counts,
        "surface_id_pattern": "{qid}_{lang}_{ordinal}",
        "qid_to_surfaces_format": "qid -> [{surface_id, surface_text, language, ordinal, source, source_index}]",
    }
    _write_json(language_dir / "metadata.json", metadata)
    _write_json(language_dir / "summary.json", summary)
    return {
        "entries": entries,
        "summary": summary,
        "metadata": metadata,
        "qid_to_surfaces": qid_to_surfaces,
    }


def build_language_surface_manifests(input_path: Path, output_dir: Path) -> dict[str, dict[str, Any]]:
    """Build per-language manifest files for surface_id inspection. / surface_id 確認用の言語別一覧を作る。"""

    rows = load_ontology_rows(input_path)
    output_summary: dict[str, dict[str, Any]] = {}
    for language, entries in _group_entries_by_language(rows).items():
        language_dir = output_dir / language
        payload = _write_language_surface_files(language_dir, entries, input_path)
        output_summary[language] = payload["summary"]
    return output_summary


class BERTLanguageEncoder:
    """Lazy-loaded BERT text encoder. / 遅延ロードする BERT テキストエンコーダ。"""

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 32, batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._resolved_device: str | None = None
        self._hidden_size: int | None = None

    def _load(self) -> None:
        if self._tokenizer is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(
                "BERT embeddings require optional dependencies: torch and transformers."
            ) from exc

        requested_device = self.device or "auto"
        if requested_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is false. "
                "Check the NVIDIA driver, container GPU passthrough, and /dev/nvidia* devices."
            )
        else:
            resolved_device = requested_device
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.to(resolved_device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved_device
        self._hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "dim", 0) or 0)

    @property
    def hidden_size(self) -> int:
        self._load()
        return int(self._hidden_size or 0)

    @property
    def resolved_device(self) -> str:
        self._load()
        return str(self._resolved_device or "cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        if not texts:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        torch = self._torch
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            batches = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batches = {name: tensor.to(self._resolved_device) for name, tensor in batches.items()}
            with torch.no_grad():
                outputs_batch = self._model(**batches)
                hidden = outputs_batch.last_hidden_state
                attention_mask = batches["attention_mask"].unsqueeze(-1).type_as(hidden)
                summed = (hidden * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, self.hidden_size), dtype=np.float32)


def build_bert_encoder(
    model_name: str,
    device: str | None = None,
    max_length: int = 32,
    batch_size: int = 16,
) -> TextEncoder:
    """Factory for the default BERT encoder backend. / 既定の BERT エンコーダ生成器。"""

    return BERTLanguageEncoder(model_name=model_name, device=device, max_length=max_length, batch_size=batch_size)


def build_language_embeddings(
    input_path: Path,
    output_dir: Path,
    english_model: str = "google-bert/bert-base-uncased",
    japanese_model: str = "tohoku-nlp/bert-base-japanese-v3",
    batch_size: int = 16,
    max_length: int = 32,
    device: str | None = None,
    encoder_factory: Callable[[str, str | None, int, int], TextEncoder] = build_bert_encoder,
) -> dict[str, dict[str, Any]]:
    """Build BERT-based language embeddings alongside manifest files. / BERT ベースの言語埋め込みを作る。"""

    rows = load_ontology_rows(input_path)
    output_summary: dict[str, dict[str, Any]] = {}
    encoders = {
        "en": encoder_factory(english_model, device, max_length, batch_size),
        "ja": encoder_factory(japanese_model, device, max_length, batch_size),
    }

    for language, entries in _group_entries_by_language(rows).items():
        language_dir = output_dir / language
        _write_language_surface_files(language_dir, entries, input_path)
        encoder = encoders[language]
        texts = [entry.surface_text for entry in entries]
        embeddings = encoder.encode(texts)
        if embeddings.shape[0] != len(entries):
            raise ValueError(
                f"Encoder for {language} returned {embeddings.shape[0]} rows for {len(entries)} inputs"
            )
        if embeddings.ndim != 2:
            raise ValueError(f"Encoder for {language} must return a 2D array, got {embeddings.shape}")
        resolved_device = getattr(encoder, "resolved_device", device or "cpu")
        summary = _save_language_embedding_store(
            language_dir=language_dir,
            entries=entries,
            embeddings=embeddings.astype(np.float32, copy=False),
            model_name=english_model if language == "en" else japanese_model,
            input_path=input_path,
            device=str(resolved_device),
            batch_size=batch_size,
            max_length=max_length,
        )
        output_summary[language] = summary

    return output_summary


def _save_language_embedding_store(
    language_dir: Path,
    entries: list[SurfaceEntry],
    embeddings: np.ndarray,
    model_name: str,
    input_path: Path,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    """Persist embeddings and metadata for one language. / 1 言語分の埋め込みを保存する。"""

    surface_ids = [entry.surface_id for entry in entries]
    qids = [entry.qid for entry in entries]
    _write_embedding_array(language_dir / "embeddings.npy", embeddings)
    _write_json(language_dir / "surface_ids.json", surface_ids)
    _write_json(language_dir / "qids.json", qids)
    qid_to_surfaces = _build_qid_to_surfaces(entries)
    _write_json(language_dir / "qid_to_surfaces.json", qid_to_surfaces)

    source_counts: dict[str, int] = {}
    for entry in entries:
        source_counts[entry.source] = source_counts.get(entry.source, 0) + 1
    summary = {
        "kind": "language_bert_embeddings",
        "language": language_dir.name,
        "created_at_utc": _timestamp_utc(),
        "item_count": len(entries),
        "unique_qid_count": len(set(qids)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "encoder_model": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "source_counts": source_counts,
        "input_file": str(input_path),
        "output_files": {
            "embeddings_npy": str(language_dir / "embeddings.npy"),
            "surface_ids_json": str(language_dir / "surface_ids.json"),
            "qids_json": str(language_dir / "qids.json"),
            "qid_to_surfaces_json": str(language_dir / "qid_to_surfaces.json"),
            "metadata_json": str(language_dir / "metadata.json"),
            "summary_json": str(language_dir / "summary.json"),
        },
    }
    metadata = {
        "kind": "language_bert_embeddings",
        "language": language_dir.name,
        "created_at_utc": summary["created_at_utc"],
        "input_file": str(input_path),
        "output_dir": str(language_dir),
        "encoder_model": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "pooling": "mean_pooling",
        "surface_id_pattern": "{qid}_{lang}_{ordinal}",
        "qid_to_surfaces_format": "qid -> [{surface_id, surface_text, language, ordinal, source, source_index}]",
        "item_count": len(entries),
        "unique_qid_count": len(set(qids)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "source_counts": source_counts,
    }
    _write_json(language_dir / "metadata.json", metadata)
    _write_json(language_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for language surface manifests. / 言語 surface 一覧生成コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build language surface manifests from bird ontology PKL.")
    parser.add_argument("--input", default=str(paths.ontology_pkl))
    parser.add_argument("--output-dir", default=str(paths.embeddings_dir / "language"))
    return parser


def build_embeddings_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for language BERT embeddings. / 言語 BERT 埋め込み用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build BERT-based language embeddings from bird ontology PKL.")
    parser.add_argument("--input", default=str(paths.ontology_pkl))
    parser.add_argument("--output-dir", default=str(paths.embeddings_dir / "language"))
    parser.add_argument("--english-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--japanese-model", default="tohoku-nlp/bert-base-japanese-v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the surface manifest generation command. / surface 一覧生成コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    build_language_surface_manifests(Path(args.input), Path(args.output_dir))
    return 0


def main_embeddings(argv: list[str] | None = None) -> int:
    """Run the language embedding generation command. / 言語埋め込み生成コマンドを実行する。"""

    args = build_embeddings_parser().parse_args(argv)
    build_language_embeddings(
        Path(args.input),
        Path(args.output_dir),
        english_model=args.english_model,
        japanese_model=args.japanese_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
