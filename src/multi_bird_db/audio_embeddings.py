from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio_backends import get_audio_backend_spec, list_audio_backends
from .audio_windows import AudioWindow, segment_waveform
from .config import get_project_paths

DEFAULT_MODEL_NAME = "facebook/wav2vec2-base-960h"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_SECONDS = 30.0
DEFAULT_TARGET_SAMPLE_RATE = 16000
DEFAULT_BIRDNET_SAMPLE_RATE = 48000
DEFAULT_BIRDNET_MODEL_TYPE = "acoustic"
DEFAULT_BIRDNET_MODEL_VERSION = "2.4"
DEFAULT_BIRDNET_BACKEND = "auto"
DEFAULT_BIRDNET_CPU_BACKEND = "tf"
DEFAULT_BIRDNET_GPU_BACKEND = "pb"
DEFAULT_PERCH_MODEL_NAME = "perch2"
DEFAULT_PERCH_MODEL_TYPE = "Perch2"
DEFAULT_EXTENSIONS = ("mp3", "wav", "flac", "ogg", "m4a")
DEFAULT_CACHE_DIR = get_project_paths().root / "data" / "external" / "models" / "audio" / "huggingface"
SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
QID_RE = re.compile(r"^Q\d+$")

MANIFEST_COLUMNS = [
    "audio_id",
    "qid",
    "source_path",
    "relative_path",
    "window_index",
    "window_start_seconds",
    "window_end_seconds",
    "window_seconds",
    "file_type",
    "sample_rate",
    "num_samples",
    "duration_seconds",
]


def _timestamp_mmddhhmm() -> str:
    return datetime.now().astimezone().strftime("%m%d%H%M")


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_component(value: str) -> str:
    normalized = SAFE_COMPONENT_RE.sub("_", value.strip())
    normalized = normalized.strip("._")
    return normalized or "item"


def _render_progress_line(message: str) -> None:
    import sys

    sys.stderr.write(f"\r{message}")
    sys.stderr.flush()


def _finish_progress_line(message: str | None = None) -> None:
    import sys

    if message is not None:
        sys.stderr.write(f"\r{message}\n")
    else:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _write_json_atomic(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _write_tsv_atomic(output_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
            newline="",
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="	"))


def _normalize_compare_path(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def _compatible_optional_float(value: Any, expected: float | None) -> bool:
    if expected is None:
        return value in {None, ""}
    try:
        return float(value) == float(expected)
    except (TypeError, ValueError):
        return False


def _resume_metadata_matches(
    metadata: dict[str, Any],
    *,
    input_dir: Path,
    backend: str,
    model_name: str,
    max_seconds: float,
    target_sample_rate: int,
    window_seconds: float | None,
    overlap_seconds: float | None,
    extensions: tuple[str, ...],
) -> bool:
    if str(metadata.get("backend") or "") != backend:
        return False
    if str(metadata.get("model_name") or "") != model_name:
        return False
    if _normalize_compare_path(metadata.get("input_dir") or "") != _normalize_compare_path(input_dir):
        return False
    if not _compatible_optional_float(metadata.get("max_seconds"), max_seconds):
        return False
    if int(metadata.get("target_sample_rate") or -1) != int(target_sample_rate):
        return False
    if not _compatible_optional_float(metadata.get("backend_window_seconds"), window_seconds):
        return False
    if not _compatible_optional_float(metadata.get("backend_overlap_seconds"), overlap_seconds):
        return False
    recorded_extensions = sorted(str(value).lower().lstrip(".") for value in (metadata.get("file_extension_whitelist") or []))
    expected_extensions = sorted(str(value).lower().lstrip(".") for value in extensions)
    return recorded_extensions == expected_extensions


def _load_existing_audio_items(
    run_root: Path,
    *,
    input_dir: Path,
    backend: str,
    model_name: str,
    max_seconds: float,
    target_sample_rate: int,
    window_seconds: float | None,
    overlap_seconds: float | None,
    extensions: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], int]:
    if not run_root.exists():
        return {}, 0

    existing_items: dict[str, dict[str, Any]] = {}
    compatible_run_count = 0
    run_dirs = sorted((child for child in run_root.iterdir() if child.is_dir()), key=lambda item: item.name)
    for run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        manifest_path = run_dir / "audio_manifest.tsv"
        embeddings_path = run_dir / "embeddings.npy"
        audio_ids_path = run_dir / "audio_ids.json"
        qids_path = run_dir / "qids.json"
        required_paths = [metadata_path, manifest_path, embeddings_path, audio_ids_path, qids_path]
        if not all(path.exists() for path in required_paths):
            continue

        metadata = _read_json(metadata_path)
        if not _resume_metadata_matches(
            metadata,
            input_dir=input_dir,
            backend=backend,
            model_name=model_name,
            max_seconds=max_seconds,
            target_sample_rate=target_sample_rate,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            extensions=extensions,
        ):
            continue

        manifest_rows = _read_tsv(manifest_path)
        audio_ids = list(_read_json(audio_ids_path))
        qids = list(_read_json(qids_path))
        embeddings = np.asarray(np.load(embeddings_path), dtype=np.float32)
        if embeddings.ndim != 2:
            continue
        if len(audio_ids) != embeddings.shape[0] or len(qids) != embeddings.shape[0]:
            continue

        index_by_audio_id = {str(audio_id): index for index, audio_id in enumerate(audio_ids)}
        compatible_run_count += 1
        for row in manifest_rows:
            audio_id = str(row.get("audio_id") or "")
            index = index_by_audio_id.get(audio_id)
            if index is None:
                continue
            existing_items[audio_id] = {
                "audio_id": audio_id,
                "qid": str(qids[index]),
                "embedding": np.asarray(embeddings[index], dtype=np.float32).copy(),
                "manifest_row": {column: str(row.get(column) or "") for column in MANIFEST_COLUMNS},
                "source_run_dir": str(run_dir),
            }
    return existing_items, compatible_run_count


def discover_audio_files(input_dir: Path, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS) -> list[Path]:
    """Return a sorted list of audio files under one directory tree. / 1 つのツリー内の音声ファイルを列挙する。"""

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    normalized_extensions = {f".{ext.lower().lstrip('.')}" for ext in extensions}
    files = [path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in normalized_extensions]
    return sorted(files, key=lambda path: str(path))


def infer_qid(path: Path, input_dir: Path) -> str:
    """Infer a QID from the path. / パスから QID を推定する。"""

    try:
        relative_parts = path.relative_to(input_dir).parts
    except ValueError:
        relative_parts = path.parts
    for part in relative_parts:
        if QID_RE.match(part):
            return part
    return path.parent.name or "unknown"


def build_audio_id(path: Path, qid: str, input_dir: Path) -> str:
    """Create a stable audio item ID. / 安定した音声 item ID を作る。"""

    try:
        relative = path.relative_to(input_dir)
    except ValueError:
        relative = path
    parts = list(relative.parts)
    if parts and QID_RE.match(parts[0]):
        parts = parts[1:]
    if parts:
        relative_stem = Path(*parts).with_suffix("")
        suffix = _safe_component(relative_stem.as_posix())
    else:
        suffix = _safe_component(path.stem)
    return f"{qid}_{suffix}"


def _load_with_ffmpeg(path: Path, target_sample_rate: int, max_seconds: float | None) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available.")

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
    ]
    if max_seconds is not None:
        command.extend(["-t", str(max_seconds)])
    command.extend(
        [
            "-ac",
            "1",
            "-ar",
            str(target_sample_rate),
            "-vn",
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
    )
    proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}: {stderr}")

    with wave.open(io.BytesIO(proc.stdout), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        pcm_bytes = wav_file.readframes(frame_count)
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(pcm), sample_rate


def _normalize_waveform_array(waveform: Any) -> np.ndarray:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()

    waveform_array = np.asarray(waveform, dtype=np.float32)
    if waveform_array.ndim == 2:
        waveform_array = waveform_array.mean(axis=0)
    elif waveform_array.ndim != 1:
        waveform_array = waveform_array.reshape(-1)
    return np.ascontiguousarray(waveform_array.astype(np.float32, copy=False))


def _resample_waveform_array(waveform: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if source_sample_rate == target_sample_rate:
        return np.ascontiguousarray(waveform.astype(np.float32, copy=False))
    if waveform.size == 0:
        return np.zeros(0, dtype=np.float32)

    duration_seconds = waveform.shape[0] / float(source_sample_rate)
    target_length = max(int(round(duration_seconds * target_sample_rate)), 1)
    if waveform.shape[0] == 1:
        return np.full(target_length, float(waveform[0]), dtype=np.float32)

    source_positions = np.linspace(0.0, duration_seconds, num=waveform.shape[0], endpoint=False, dtype=np.float64)
    target_positions = np.linspace(0.0, duration_seconds, num=target_length, endpoint=False, dtype=np.float64)
    resampled = np.interp(target_positions, source_positions, waveform.astype(np.float64, copy=False))
    return np.ascontiguousarray(resampled.astype(np.float32, copy=False))


def load_audio_file(
    path: Path,
    target_sample_rate: int,
    max_seconds: float | None = None,
    audio_loader: Callable[[Path], tuple[Any, int]] | None = None,
) -> tuple[np.ndarray, int]:
    """Load and resample one audio file. / 1 件の音声を読み込み、必要ならリサンプルする。"""

    waveform: Any
    sample_rate: int
    if audio_loader is not None:
        waveform, sample_rate = audio_loader(path)
        waveform_array = _normalize_waveform_array(waveform)
        if max_seconds is not None and max_seconds > 0:
            max_frames = int(round(sample_rate * max_seconds))
            if waveform_array.shape[0] > max_frames:
                waveform_array = waveform_array[:max_frames]
        if sample_rate != target_sample_rate:
            waveform_array = _resample_waveform_array(waveform_array, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        return np.ascontiguousarray(waveform_array.astype(np.float32, copy=False)), sample_rate

    return _load_with_ffmpeg(path, target_sample_rate=target_sample_rate, max_seconds=max_seconds)


@dataclass(slots=True)
class AudioEmbeddingStore:
    """Store audio embeddings and row-aligned metadata. / 音声埋め込みと行対応メタデータを保持する。"""

    audio_ids: list[str]
    qids: list[str]
    embeddings: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {self.embeddings.shape}")
        if self.embeddings.shape[0] != len(self.audio_ids):
            raise ValueError("Embedding rows must match audio_ids")
        if len(self.audio_ids) != len(self.qids):
            raise ValueError("audio_ids and qids must have the same length")


def _save_audio_embedding_store(store: AudioEmbeddingStore, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", store.embeddings)
    _write_json_atomic(output_dir / "audio_ids.json", store.audio_ids)
    _write_json_atomic(output_dir / "qids.json", store.qids)
    _write_json_atomic(output_dir / "metadata.json", store.metadata)


def _waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Serialize one mono waveform to 16-bit WAV bytes. / 1 本の mono 波形を 16bit WAV にする。"""

    waveform_array = np.asarray(waveform, dtype=np.float32).reshape(-1)
    waveform_array = np.clip(waveform_array, -1.0, 1.0)
    pcm = np.asarray(np.round(waveform_array * 32767.0), dtype=np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return temp_path.read_bytes()
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _coerce_embedding_matrix(encoded: Any, expected_rows: int) -> np.ndarray:
    """Normalize model output to a 2D float32 matrix. / モデル出力を 2D float32 行列に揃える。"""

    if hasattr(encoded, "to_numpy"):
        encoded = encoded.to_numpy()
    elif hasattr(encoded, "values") and not isinstance(encoded, np.ndarray):
        encoded = encoded.values
    matrix = np.asarray(encoded, dtype=np.float32)
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    elif matrix.ndim == 1:
        if expected_rows == 1:
            matrix = matrix.reshape(1, -1)
        else:
            matrix = matrix.reshape(expected_rows, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.shape[0] != expected_rows:
        raise ValueError(f"Encoder returned {matrix.shape[0]} rows, expected {expected_rows}")
    return matrix.astype(np.float32, copy=False)


def resolve_torch_device(requested_device: str) -> str:
    """Resolve an audio runtime device string. / 音声 runtime の device 文字列を解決する。"""

    normalized = str(requested_device).strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device for audio embeddings: {requested_device}")
    if normalized == "auto":
        try:
            import torch
        except Exception:
            return "cpu"
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if normalized == "cuda":
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("CUDA was requested for audio embeddings, but torch is not installed.") from exc
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested for audio embeddings, but torch.cuda.is_available() is false.")
    return normalized


def birdnet_gpu_available() -> bool:
    """Return whether TensorFlow can see at least one GPU for BirdNET pb inference."""

    try:
        import tensorflow as tf
    except Exception:
        return False
    try:
        return bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        return False


def resolve_birdnet_runtime(requested_device: str, requested_backend: str = DEFAULT_BIRDNET_BACKEND) -> tuple[str, str]:
    """Resolve the BirdNET backend and runtime device strings."""

    normalized_device = str(requested_device).strip().lower()
    if normalized_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device for BirdNET embeddings: {requested_device}")
    normalized_backend = str(requested_backend).strip().lower()
    if normalized_backend not in {"auto", "tf", "pb"}:
        raise ValueError(f"Unsupported BirdNET backend: {requested_backend}")

    has_gpu = birdnet_gpu_available()

    if normalized_device == "cuda":
        if normalized_backend == "tf":
            raise RuntimeError("BirdNET backend 'tf' is CPU-only. Use backend 'pb' or leave the backend on 'auto'.")
        if not has_gpu:
            raise RuntimeError(
                "CUDA was requested for BirdNET embeddings, but TensorFlow could not find a GPU."
            )
        return (DEFAULT_BIRDNET_GPU_BACKEND if normalized_backend == "auto" else normalized_backend, "GPU:0")

    if normalized_device == "cpu":
        return (DEFAULT_BIRDNET_CPU_BACKEND if normalized_backend == "auto" else normalized_backend, "CPU")

    if has_gpu and normalized_backend != "tf":
        return (DEFAULT_BIRDNET_GPU_BACKEND if normalized_backend == "auto" else normalized_backend, "GPU:0")
    return (DEFAULT_BIRDNET_CPU_BACKEND if normalized_backend == "auto" else normalized_backend, "CPU")


def _ensure_spawn_start_method_for_birdnet_gpu(runtime_device: str) -> None:
    """Avoid CUDA+fork failures in BirdNET GPU workers by forcing spawn."""

    if not str(runtime_device).startswith("GPU"):
        return
    current = mp.get_start_method(allow_none=True)
    if current in {None, "spawn"}:
        mp.set_start_method("spawn", force=True)
        return
    raise RuntimeError(
        f"BirdNET GPU embeddings require multiprocessing start method 'spawn', but found '{current}'. "
        "Start a fresh Python process and retry."
    )


class Wav2Vec2AudioEncoder:
    """Encode waveforms using a Hugging Face wav2vec2 model. / Hugging Face の wav2vec2 で埋め込む。"""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "auto", cache_dir: Path | None = None):
        from transformers import AutoModel, AutoFeatureExtractor

        self.model_name = model_name
        self.device = resolve_torch_device(device)
        self.cache_dir = cache_dir
        cache_path = str(cache_dir) if cache_dir else None
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=cache_path)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_path)
        self.model.eval()
        self.model.to(self.device)
        self.sample_rate = int(getattr(self.feature_extractor, "sampling_rate", DEFAULT_TARGET_SAMPLE_RATE))

    def encode_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        import torch

        inputs = self.feature_extractor(
            [np.asarray(waveform, dtype=np.float32) for waveform in waveforms],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


class BirdNETAudioEncoder:
    """Encode 3-second windows with BirdNET. / BirdNET で 3 秒窓を埋め込む。"""

    def __init__(
        self,
        model_type: str = DEFAULT_BIRDNET_MODEL_TYPE,
        model_version: str = DEFAULT_BIRDNET_MODEL_VERSION,
        backend: str = DEFAULT_BIRDNET_BACKEND,
        device: str = "cpu",
        model: Any | None = None,
    ):
        try:
            import birdnet
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "BirdNET backend requires the `birdnet` Python package. Install the audio-birdnet extra and retry."
            ) from exc

        self.model_type = model_type
        self.model_version = model_version
        self.requested_backend = backend
        self.requested_device = device
        self.backend, self.runtime_device = resolve_birdnet_runtime(device, backend)
        _ensure_spawn_start_method_for_birdnet_gpu(self.runtime_device)
        self.device = "cuda" if self.runtime_device.startswith("GPU") else "cpu"
        self.sample_rate = DEFAULT_BIRDNET_SAMPLE_RATE
        self.model = model if model is not None else birdnet.load(model_type, model_version, self.backend)

    def _coerce_result_rows(self, encoded: Any, expected_rows: int) -> np.ndarray:
        if hasattr(encoded, "embeddings") and hasattr(encoded, "embeddings_masked"):
            embeddings = np.asarray(encoded.embeddings, dtype=np.float32)
            mask = np.asarray(encoded.embeddings_masked, dtype=bool)
            if embeddings.ndim == 3 and mask.ndim == 3:
                rows: list[np.ndarray] = []
                valid_mask = ~mask.all(axis=2)
                for input_index in range(embeddings.shape[0]):
                    valid_embeddings = embeddings[input_index][valid_mask[input_index]]
                    if valid_embeddings.size == 0:
                        raise RuntimeError(f"BirdNET returned no valid embeddings for input index {input_index}.")
                    rows.append(valid_embeddings.mean(axis=0, dtype=np.float32))
                return _coerce_embedding_matrix(np.stack(rows, axis=0), expected_rows=expected_rows)
        if hasattr(encoded, "embeddings"):
            encoded = encoded.embeddings
        encoded_array = np.asarray(encoded, dtype=np.float32)
        if encoded_array.ndim == 3:
            encoded_array = encoded_array.mean(axis=1)
        return _coerce_embedding_matrix(encoded_array, expected_rows=expected_rows)

    def encode_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        items = [
            (
                _normalize_waveform_array(waveform).reshape(-1).astype(np.float32, copy=False),
                self.sample_rate,
            )
            for waveform in waveforms
        ]
        use_gpu = self.runtime_device.startswith("GPU")
        encoded = self.model.encode_arrays(
            items,
            device=self.runtime_device,
            batch_size=1 if use_gpu else max(1, len(items)),
            n_workers=1 if use_gpu else None,
            n_producers=1,
        )
        return self._coerce_result_rows(encoded, expected_rows=len(items))


class PerchAudioEncoder:
    """Encode 5-second windows with Perch2 from Bioacoustics Model Zoo. / Bioacoustics Model Zoo の Perch2 で 5 秒窓を埋め込む。"""

    def __init__(
        self,
        model_name: str = DEFAULT_PERCH_MODEL_NAME,
        device: str = "cpu",
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.model_type = DEFAULT_PERCH_MODEL_TYPE
        self.model_version = ""
        self.backend = "bioacoustics-model-zoo"
        self.device = device
        self.sample_rate = 22050
        if model is not None:
            self.model = model
            return
        try:
            import bioacoustics_model_zoo as bmz
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Perch backend requires the `bioacoustics-model-zoo` Python package. Install the audio-perch extra and retry."
            ) from exc

        if hasattr(bmz, self.model_type):
            self.model = getattr(bmz, self.model_type)()
        else:  # pragma: no cover - future-proof fallback
            raise RuntimeError(f"bioacoustics-model-zoo does not expose a {self.model_type} model.")

    def _call_model(self, audio_paths: list[str]) -> Any:
        if hasattr(self.model, "embed"):
            return self.model.embed(audio_paths)
        if hasattr(self.model, "generate_embeddings"):
            return self.model.generate_embeddings(audio_paths)
        raise RuntimeError("Perch model does not expose embed() or generate_embeddings().")

    def encode_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="perch_embed_") as tmpdir:
            temp_paths: list[str] = []
            for index, waveform in enumerate(waveforms):
                temp_path = Path(tmpdir) / f"window_{index:04d}.wav"
                temp_path.write_bytes(_waveform_to_wav_bytes(_normalize_waveform_array(waveform), self.sample_rate))
                temp_paths.append(str(temp_path))
            encoded = self._call_model(temp_paths)
        return _coerce_embedding_matrix(encoded, expected_rows=len(waveforms))


def download_audio_models(
    backend: str = "wav2vec2",
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    device: str = "auto",
) -> dict[str, Any]:
    """Download and cache model assets needed by an audio backend. / 音声 backend に必要なモデル資産を事前取得する。"""

    backend_spec = get_audio_backend_spec(backend)
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)

    if backend == "wav2vec2":
        encoder = Wav2Vec2AudioEncoder(model_name=model_name, device=device, cache_dir=cache_path)
        return {
            "backend": backend,
            "model_name": model_name,
            "requested_device": device,
            "resolved_device": encoder.device,
            "sample_rate": encoder.sample_rate,
            "cache_dir": str(cache_path) if cache_path is not None else "",
            "required_python_packages": list(backend_spec.required_python_packages),
            "required_system_packages": list(backend_spec.required_system_packages),
        }

    raise NotImplementedError(
        f"Model predownload is currently wired only for wav2vec2. Requested backend: {backend}"
    )


def build_audio_embeddings(
    input_dir: Path,
    output_dir: Path,
    backend: str = "wav2vec2",
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    target_sample_rate: int | None = None,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    audio_loader: Callable[[Path], tuple[Any, int]] | None = None,
    encoder: Any | None = None,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Build audio embeddings for all files under one directory tree. / ディレクトリ配下の音声埋め込みを作る。"""

    backend_spec = get_audio_backend_spec(backend)

    files = discover_audio_files(input_dir, extensions=extensions)
    if not files:
        raise FileNotFoundError(f"No audio files were found under: {input_dir}")

    if backend == "wav2vec2":
        if encoder is None:
            encoder = Wav2Vec2AudioEncoder(model_name=model_name, device=device, cache_dir=cache_dir)
    elif backend == "birdnet":
        if encoder is None:
            encoder = BirdNETAudioEncoder(device=device)
        if model_name == DEFAULT_MODEL_NAME:
            model_name = f"birdnet-{encoder.model_type}-{encoder.model_version}-{encoder.backend}"
    elif backend == "perch":
        if model_name == DEFAULT_MODEL_NAME:
            model_name = DEFAULT_PERCH_MODEL_NAME
        if encoder is None:
            encoder = PerchAudioEncoder(device=device)
    else:
        raise NotImplementedError(
            f"Backend '{backend}' is registered with the contract ({backend_spec.window_seconds}s windows, "
            f"scope={backend_spec.embedding_scope}), but the encoder implementation is not wired yet."
        )

    model_sample_rate = int(getattr(encoder, "sample_rate", backend_spec.sample_rate_hz or DEFAULT_TARGET_SAMPLE_RATE))
    if backend_spec.sample_rate_hz is not None:
        if target_sample_rate == DEFAULT_TARGET_SAMPLE_RATE:
            decode_sample_rate = int(backend_spec.sample_rate_hz)
        elif target_sample_rate is None:
            decode_sample_rate = int(backend_spec.sample_rate_hz)
        else:
            decode_sample_rate = int(target_sample_rate)
        if decode_sample_rate != model_sample_rate:
            raise ValueError(
                f"target_sample_rate ({decode_sample_rate}) must match the encoder sample rate ({model_sample_rate}) for {backend}."
            )
    else:
        decode_sample_rate = int(target_sample_rate or model_sample_rate)
        if decode_sample_rate != model_sample_rate:
            raise ValueError(
                f"target_sample_rate ({decode_sample_rate}) must match the encoder sample rate ({model_sample_rate}) for {backend}."
            )

    run_root = output_dir / _safe_component(backend) / _safe_component(model_name)
    existing_items: dict[str, dict[str, Any]] = {}
    resume_source_run_count = 0
    if resume_existing:
        existing_items, resume_source_run_count = _load_existing_audio_items(
            run_root,
            input_dir=input_dir,
            backend=backend,
            model_name=model_name,
            max_seconds=max_seconds,
            target_sample_rate=decode_sample_rate,
            window_seconds=backend_spec.window_seconds,
            overlap_seconds=backend_spec.overlap_seconds,
            extensions=extensions,
        )

    run_dir = run_root / _timestamp_mmddhhmm()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    audio_ids: list[str] = []
    qids: list[str] = []
    embedding_rows: list[np.ndarray] = []

    pending_waveforms: list[np.ndarray] = []
    pending_rows: list[dict[str, str]] = []
    total_files = len(files)
    processed_files = 0
    batch_count = 0
    reused_item_count = 0
    new_item_count = 0
    last_progress_time = time.monotonic()

    def append_reused_item(item: dict[str, Any]) -> None:
        nonlocal reused_item_count
        row_with_index = dict(item["manifest_row"])
        row_with_index["embedding_index"] = str(len(audio_ids))
        manifest_rows.append(row_with_index)
        audio_ids.append(str(item["audio_id"]))
        qids.append(str(item["qid"]))
        embedding_rows.append(np.asarray(item["embedding"], dtype=np.float32).copy())
        reused_item_count += 1

    def report_progress(*, current_label: str = "", force: bool = False) -> None:
        nonlocal last_progress_time
        now = time.monotonic()
        if not force and now - last_progress_time < 0.5:
            return
        item_count = len(audio_ids) + len(pending_rows)
        label = f" | {current_label}" if current_label else ""
        _render_progress_line(
            f"audio-{backend} | files {processed_files}/{total_files} | items {item_count} | "
            f"reused {reused_item_count} | failed {len(failed_rows)} | batches {batch_count}{label}"
        )
        last_progress_time = now

    report_progress(force=True, current_label="starting")

    def flush_batch() -> None:
        nonlocal batch_count, new_item_count
        if not pending_waveforms:
            return
        embeddings = encoder.encode_batch(pending_waveforms)
        if embeddings.ndim != 2:
            raise ValueError(f"Encoder must return a 2D matrix, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(pending_rows):
            raise ValueError("Encoder row count does not match the batch size")
        batch_rows = embeddings.astype(np.float32, copy=False)
        for row_index, row in enumerate(pending_rows):
            row_with_index = dict(row)
            row_with_index["embedding_index"] = str(len(audio_ids))
            manifest_rows.append(row_with_index)
            audio_ids.append(row["audio_id"])
            qids.append(row["qid"])
            embedding_rows.append(np.asarray(batch_rows[row_index], dtype=np.float32).copy())
            new_item_count += 1
        batch_count += 1
        pending_waveforms.clear()
        pending_rows.clear()
        report_progress(force=True, current_label="batch complete")

    for path in files:
        qid = infer_qid(path, input_dir)
        base_audio_id = build_audio_id(path, qid, input_dir)
        relative_path = str(path.relative_to(input_dir)) if path.is_relative_to(input_dir) else str(path)

        if backend_spec.window_seconds is None and base_audio_id in existing_items:
            append_reused_item(existing_items[base_audio_id])
            processed_files += 1
            report_progress(force=True, current_label=f"reused {path.name}")
            continue

        try:
            waveform, sample_rate = load_audio_file(
                path,
                target_sample_rate=decode_sample_rate,
                max_seconds=max_seconds,
                audio_loader=audio_loader,
            )
        except Exception as exc:
            failed_rows.append(
                {
                    "audio_id": base_audio_id,
                    "qid": qid,
                    "source_path": str(path),
                    "relative_path": relative_path,
                    "error": str(exc),
                }
            )
            processed_files += 1
            report_progress(force=True, current_label=f"failed {path.name}")
            continue

        if backend_spec.window_seconds is None:
            windows = [
                AudioWindow(
                    index=0,
                    start_seconds=0.0,
                    end_seconds=float(waveform.shape[0]) / float(sample_rate),
                    waveform=waveform,
                )
            ]
        else:
            windows = segment_waveform(
                waveform,
                sample_rate=sample_rate,
                window_seconds=float(backend_spec.window_seconds),
                overlap_seconds=float(backend_spec.overlap_seconds or 0.0),
                pad_mode="noise" if backend == "birdnet" else "zeros",
            )
        for window in windows:
            if backend_spec.window_seconds is None:
                item_audio_id = base_audio_id
                window_index = ""
                window_start_seconds = ""
                window_end_seconds = ""
                window_seconds = ""
            else:
                item_audio_id = f"{base_audio_id}_w{window.index:04d}"
                window_index = str(window.index)
                window_start_seconds = f"{window.start_seconds:.6f}"
                window_end_seconds = f"{window.end_seconds:.6f}"
                window_seconds = f"{float(backend_spec.window_seconds):.6f}"

            if item_audio_id in existing_items:
                append_reused_item(existing_items[item_audio_id])
                continue

            duration_seconds = float(window.waveform.shape[0]) / float(sample_rate or decode_sample_rate)
            pending_waveforms.append(window.waveform)
            pending_rows.append(
                {
                    "audio_id": item_audio_id,
                    "qid": qid,
                    "source_path": str(path),
                    "relative_path": relative_path,
                    "window_index": window_index,
                    "window_start_seconds": window_start_seconds,
                    "window_end_seconds": window_end_seconds,
                    "window_seconds": window_seconds,
                    "file_type": path.suffix.lower().lstrip("."),
                    "sample_rate": str(sample_rate),
                    "num_samples": str(int(window.waveform.shape[0])),
                    "duration_seconds": f"{duration_seconds:.6f}",
                }
            )
            if len(pending_waveforms) >= batch_size:
                flush_batch()

        processed_files += 1
        report_progress(force=True, current_label=path.name)

    flush_batch()

    if not embedding_rows:
        _finish_progress_line(
            f"audio-{backend} done | files {processed_files}/{total_files} | items 0 | reused {reused_item_count} | failed {len(failed_rows)} | batches {batch_count}"
        )
        raise RuntimeError("No audio files could be embedded. Check the decoder/model setup and input files.")

    embeddings = np.stack(embedding_rows, axis=0).astype(np.float32, copy=False)
    store = AudioEmbeddingStore(
        audio_ids=audio_ids,
        qids=qids,
        embeddings=embeddings,
        metadata={
            "kind": f"audio_{backend}_embeddings",
            "dataset": "xeno-canto",
            "created_at_utc": _timestamp_utc(),
            "backend": backend,
            "backend_notes": backend_spec.notes,
            "backend_required_python_packages": list(backend_spec.required_python_packages),
            "backend_required_system_packages": list(backend_spec.required_system_packages),
            "backend_window_seconds": backend_spec.window_seconds,
            "backend_overlap_seconds": backend_spec.overlap_seconds,
            "backend_sample_rate_hz": backend_spec.sample_rate_hz,
            "backend_embedding_scope": backend_spec.embedding_scope,
            "input_dir": str(input_dir),
            "output_root": str(output_dir),
            "run_dir": str(run_dir),
            "model_name": model_name,
            "device": device,
            "resolved_device": getattr(encoder, "device", device),
            "batch_size": batch_size,
            "max_seconds": max_seconds,
            "target_sample_rate": decode_sample_rate,
            "window_seconds": backend_spec.window_seconds,
            "overlap_seconds": backend_spec.overlap_seconds,
            "encoder_model_type": getattr(encoder, "model_type", ""),
            "encoder_model_version": getattr(encoder, "model_version", ""),
            "encoder_backend_variant": getattr(encoder, "backend", ""),
            "embedding_dim": int(embeddings.shape[1]),
            "item_count": len(audio_ids),
            "unique_qid_count": len(set(qids)),
            "file_extension_whitelist": list(extensions),
            "failed_count": len(failed_rows),
            "decoder": "ffmpeg_or_custom_loader",
            "resume_existing": resume_existing,
            "reused_item_count": reused_item_count,
            "new_item_count": new_item_count,
            "resume_source_run_count": resume_source_run_count,
        },
    )
    _save_audio_embedding_store(store, run_dir)
    _write_tsv_atomic(run_dir / "audio_manifest.tsv", manifest_rows, MANIFEST_COLUMNS + ["embedding_index"])
    _write_json_atomic(run_dir / "failed_items.json", failed_rows)
    _finish_progress_line(
        f"audio-{backend} done | files {processed_files}/{total_files} | items {len(audio_ids)} | reused {reused_item_count} | failed {len(failed_rows)} | batches {batch_count}"
    )

    summary = {
        "kind": f"audio_{backend}_embeddings",
        "created_at_utc": store.metadata["created_at_utc"],
        "input_dir": str(input_dir),
        "output_root": str(output_dir),
        "run_dir": str(run_dir),
        "model_name": model_name,
        "device": device,
        "resolved_device": getattr(encoder, "device", device),
        "batch_size": batch_size,
        "max_seconds": max_seconds,
        "target_sample_rate": decode_sample_rate,
        "embedding_dim": int(embeddings.shape[1]),
        "item_count": len(audio_ids),
        "unique_qid_count": len(set(qids)),
        "failed_count": len(failed_rows),
        "successful_count": len(audio_ids),
        "resume_existing": resume_existing,
        "reused_item_count": reused_item_count,
        "new_item_count": new_item_count,
        "resume_source_run_count": resume_source_run_count,
        "output_files": {
            "embeddings_npy": str(run_dir / "embeddings.npy"),
            "audio_ids_json": str(run_dir / "audio_ids.json"),
            "qids_json": str(run_dir / "qids.json"),
            "audio_manifest_tsv": str(run_dir / "audio_manifest.tsv"),
            "metadata_json": str(run_dir / "metadata.json"),
            "failed_items_json": str(run_dir / "failed_items.json"),
        },
    }
    _write_json_atomic(run_dir / "summary.json", summary)
    return {"store": store, "summary": summary, "manifest_rows": manifest_rows, "failed_rows": failed_rows}


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for audio embeddings. / 音声埋め込み用 CLI パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build wav2vec2-based audio embeddings from a directory tree.")
    parser.add_argument("--backend", choices=[spec.name for spec in list_audio_backends()], default="wav2vec2")
    parser.add_argument("--input-dir", default=str(paths.xeno_canto_raw_dir))
    parser.add_argument("--output-dir", default=str(paths.audio_embeddings_dir))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--target-sample-rate", type=int, default=DEFAULT_TARGET_SAMPLE_RATE)
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def build_download_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for audio model downloads. / 音声モデル事前取得用 CLI パーサを作る。"""

    parser = argparse.ArgumentParser(description="Download and cache audio embedding models.")
    parser.add_argument("--backend", choices=[spec.name for spec in list_audio_backends()], default="wav2vec2")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    return parser


def main_download(argv: list[str] | None = None) -> int:
    """Run the audio model predownload command. / 音声モデル事前取得コマンドを実行する。"""

    args = build_download_parser().parse_args(argv)
    summary = download_audio_models(
        backend=args.backend,
        model_name=args.model_name,
        device=args.device,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the audio embedding command. / 音声埋め込みコマンドを実行する。"""

    args = build_parser().parse_args(argv)
    extensions = tuple(ext.strip() for ext in args.extensions.split(",") if ext.strip())
    build_audio_embeddings(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        backend=args.backend,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_seconds=args.max_seconds,
        target_sample_rate=args.target_sample_rate,
        extensions=extensions,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
