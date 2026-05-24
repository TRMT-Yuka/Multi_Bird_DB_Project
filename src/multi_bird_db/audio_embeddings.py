from __future__ import annotations

import argparse
import csv
import io
import json
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
DEFAULT_BIRDNET_BACKEND = "tf"
DEFAULT_EXTENSIONS = ("mp3", "wav", "flac", "ogg", "m4a")
DEFAULT_CACHE_DIR = Path("/tmp") / "multi_bird_db_audio_cache"
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


def _load_with_ffmpeg(path: Path, target_sample_rate: int, max_seconds: float | None) -> tuple["torch.Tensor", int]:
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

    import torch

    with wave.open(io.BytesIO(proc.stdout), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        pcm_bytes = wav_file.readframes(frame_count)
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(pcm.copy()), sample_rate


def load_audio_file(
    path: Path,
    target_sample_rate: int,
    max_seconds: float | None = None,
    audio_loader: Callable[[Path], tuple[Any, int]] | None = None,
) -> tuple["torch.Tensor", int]:
    """Load and resample one audio file. / 1 件の音声を読み込み、必要ならリサンプルする。"""

    import torch

    waveform: Any
    sample_rate: int
    if audio_loader is not None:
        waveform, sample_rate = audio_loader(path)
    else:
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(str(path))
        except Exception:
            waveform, sample_rate = _load_with_ffmpeg(path, target_sample_rate=target_sample_rate, max_seconds=max_seconds)

    if isinstance(waveform, np.ndarray):
        waveform_tensor = torch.from_numpy(waveform)
    else:
        waveform_tensor = waveform
    if not isinstance(waveform_tensor, torch.Tensor):
        waveform_tensor = torch.as_tensor(waveform_tensor)
    waveform_tensor = waveform_tensor.detach().cpu().float()
    if waveform_tensor.ndim == 2:
        waveform_tensor = waveform_tensor.mean(dim=0)
    elif waveform_tensor.ndim != 1:
        waveform_tensor = waveform_tensor.reshape(-1)

    if max_seconds is not None and max_seconds > 0:
        max_frames = int(round(sample_rate * max_seconds))
        if waveform_tensor.numel() > max_frames:
            waveform_tensor = waveform_tensor[:max_frames]

    if sample_rate != target_sample_rate:
        try:
            import torchaudio.functional as F

            waveform_tensor = F.resample(waveform_tensor.unsqueeze(0), sample_rate, target_sample_rate).squeeze(0)
            sample_rate = target_sample_rate
        except Exception as exc:
            raise RuntimeError(
                f"Audio file {path} has sample rate {sample_rate}, but resampling to {target_sample_rate} failed."
            ) from exc

    return waveform_tensor.contiguous(), sample_rate


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


def _waveform_to_wav_bytes(waveform: "torch.Tensor", sample_rate: int) -> bytes:
    """Serialize one mono waveform to 16-bit WAV bytes. / 1 本の mono 波形を 16bit WAV にする。"""

    import torch

    waveform = waveform.detach().cpu().float().flatten().clamp(-1.0, 1.0)
    pcm = (waveform * 32767.0).to(torch.int16).numpy()
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


class Wav2Vec2AudioEncoder:
    """Encode waveforms using a Hugging Face wav2vec2 model. / Hugging Face の wav2vec2 で埋め込む。"""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu", cache_dir: Path | None = None):
        from transformers import AutoModel, AutoFeatureExtractor

        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self.model.eval()
        self.model.to(device)
        self.sample_rate = int(getattr(self.feature_extractor, "sampling_rate", DEFAULT_TARGET_SAMPLE_RATE))

    def encode_batch(self, waveforms: list["torch.Tensor"]) -> np.ndarray:
        import torch

        inputs = self.feature_extractor(
            [waveform.detach().cpu().numpy() for waveform in waveforms],
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
    ):
        try:
            import birdnet
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "BirdNET backend requires the `birdnet` Python package. Install the audio-birdnet extra and retry."
            ) from exc

        self.model_type = model_type
        self.model_version = model_version
        self.backend = backend
        self.device = device
        self.sample_rate = DEFAULT_BIRDNET_SAMPLE_RATE
        self.model = birdnet.load(model_type, model_version, backend)

    def _encode_one(self, waveform: "torch.Tensor") -> np.ndarray:
        import numpy as np

        wav_bytes = _waveform_to_wav_bytes(waveform, self.sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            temp_path.write_bytes(wav_bytes)
            encoded = self.model.encode(str(temp_path))
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
        if hasattr(encoded, "to_numpy"):
            encoded = encoded.to_numpy()
        elif hasattr(encoded, "values") and not isinstance(encoded, np.ndarray):
            encoded = encoded.values
        encoded_array = np.asarray(encoded, dtype=np.float32)
        if encoded_array.ndim == 0:
            encoded_array = encoded_array.reshape(1)
        return encoded_array

    def encode_batch(self, waveforms: list["torch.Tensor"]) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for waveform in waveforms:
            embedding = self._encode_one(waveform)
            if embedding.ndim > 1:
                embedding = embedding.reshape(-1)
            embeddings.append(embedding)
        return np.stack(embeddings, axis=0).astype(np.float32, copy=False)


def build_audio_embeddings(
    input_dir: Path,
    output_dir: Path,
    backend: str = "wav2vec2",
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    target_sample_rate: int | None = None,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    audio_loader: Callable[[Path], tuple[Any, int]] | None = None,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Build audio embeddings for all files under one directory tree. / ディレクトリ配下の音声埋め込みを作る。"""

    import torch

    backend_spec = get_audio_backend_spec(backend)

    files = discover_audio_files(input_dir, extensions=extensions)
    if not files:
        raise FileNotFoundError(f"No audio files were found under: {input_dir}")

    if backend == "wav2vec2":
        if encoder is None:
            encoder = Wav2Vec2AudioEncoder(model_name=model_name, device=device, cache_dir=cache_dir)
    elif backend == "birdnet":
        if model_name == DEFAULT_MODEL_NAME:
            model_name = "birdnet-acoustic-2.4-tf"
        if encoder is None:
            encoder = BirdNETAudioEncoder(device=device)
    else:
        raise NotImplementedError(
            f"Backend '{backend}' is registered with the contract ({backend_spec.window_seconds}s windows, "
            f"scope={backend_spec.embedding_scope}), but the encoder implementation is not wired yet."
        )

    model_sample_rate = int(getattr(encoder, "sample_rate", DEFAULT_TARGET_SAMPLE_RATE))
    decode_sample_rate = int(target_sample_rate or model_sample_rate)
    if decode_sample_rate != model_sample_rate:
        raise ValueError(
            f"target_sample_rate ({decode_sample_rate}) must match the encoder sample rate ({model_sample_rate}) for {backend}."
        )

    run_dir = output_dir / _safe_component(backend) / _safe_component(model_name) / _timestamp_mmddhhmm()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    audio_ids: list[str] = []
    qids: list[str] = []
    embedding_batches: list[np.ndarray] = []

    pending_waveforms: list[torch.Tensor] = []
    pending_rows: list[dict[str, str]] = []

    def flush_batch() -> None:
        if not pending_waveforms:
            return
        embeddings = encoder.encode_batch(pending_waveforms)
        if embeddings.ndim != 2:
            raise ValueError(f"Encoder must return a 2D matrix, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(pending_rows):
            raise ValueError("Encoder row count does not match the batch size")
        embedding_batches.append(embeddings.astype(np.float32, copy=False))
        for row in pending_rows:
            row_with_index = dict(row)
            row_with_index["embedding_index"] = str(len(audio_ids))
            manifest_rows.append(row_with_index)
            audio_ids.append(row["audio_id"])
            qids.append(row["qid"])
        pending_waveforms.clear()
        pending_rows.clear()

    for path in files:
        qid = infer_qid(path, input_dir)
        base_audio_id = build_audio_id(path, qid, input_dir)
        relative_path = str(path.relative_to(input_dir)) if path.is_relative_to(input_dir) else str(path)
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
            continue

        if backend_spec.window_seconds is None:
            windows = [AudioWindow(index=0, start_seconds=0.0, end_seconds=float(waveform.numel()) / float(sample_rate), waveform=waveform)]
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

            duration_seconds = float(window.waveform.numel()) / float(sample_rate or decode_sample_rate)
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
                    "num_samples": str(int(window.waveform.numel())),
                    "duration_seconds": f"{duration_seconds:.6f}",
                }
            )
            if len(pending_waveforms) >= batch_size:
                flush_batch()

    flush_batch()

    if not embedding_batches:
        raise RuntimeError("No audio files could be embedded. Check the decoder/model setup and input files.")

    embeddings = np.concatenate(embedding_batches, axis=0)
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
            "backend_embedding_scope": backend_spec.embedding_scope,
            "input_dir": str(input_dir),
            "output_root": str(output_dir),
            "run_dir": str(run_dir),
            "model_name": model_name,
            "device": device,
            "batch_size": batch_size,
            "max_seconds": max_seconds,
            "target_sample_rate": decode_sample_rate,
            "window_seconds": backend_spec.window_seconds,
            "overlap_seconds": backend_spec.overlap_seconds,
            "embedding_dim": int(embeddings.shape[1]),
            "item_count": len(audio_ids),
            "unique_qid_count": len(set(qids)),
            "file_extension_whitelist": list(extensions),
            "failed_count": len(failed_rows),
            "decoder": "torchaudio+ffmpeg",
        },
    )
    _save_audio_embedding_store(store, run_dir)
    _write_tsv_atomic(run_dir / "audio_manifest.tsv", manifest_rows, MANIFEST_COLUMNS + ["embedding_index"])
    _write_json_atomic(run_dir / "failed_items.json", failed_rows)

    summary = {
        "kind": f"audio_{backend}_embeddings",
        "created_at_utc": store.metadata["created_at_utc"],
        "input_dir": str(input_dir),
        "output_root": str(output_dir),
        "run_dir": str(run_dir),
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_seconds": max_seconds,
        "target_sample_rate": decode_sample_rate,
        "embedding_dim": int(embeddings.shape[1]),
        "item_count": len(audio_ids),
        "unique_qid_count": len(set(qids)),
        "failed_count": len(failed_rows),
        "successful_count": len(audio_ids),
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--target-sample-rate", type=int, default=DEFAULT_TARGET_SAMPLE_RATE)
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    return parser


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
