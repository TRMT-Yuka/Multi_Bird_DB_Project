from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AudioBackendSpec:
    """Describe one audio embedding backend contract. / 音声埋め込み backend の契約を表す。"""

    name: str
    window_seconds: float | None
    overlap_seconds: float | None
    embedding_scope: str
    required_python_packages: tuple[str, ...]
    required_system_packages: tuple[str, ...]
    notes: str


class AudioBackend:
    """Common backend interface. / 共通 backend インターフェース。"""

    spec: AudioBackendSpec

    def encode_batch(self, waveforms: list[Any]) -> Any:  # pragma: no cover - interface only
        raise NotImplementedError


class Wav2Vec2Backend(AudioBackend):
    spec = AudioBackendSpec(
        name="wav2vec2",
        window_seconds=None,
        overlap_seconds=None,
        embedding_scope="file",
        required_python_packages=("torch", "torchaudio", "transformers"),
        required_system_packages=("ffmpeg",),
        notes="Whole-file encoder used as the current baseline.",
    )


class BirdNETBackend(AudioBackend):
    spec = AudioBackendSpec(
        name="birdnet",
        window_seconds=3.0,
        overlap_seconds=0.0,
        embedding_scope="window",
        required_python_packages=("birdnet", "tensorflow", "tensorflow-hub"),
        required_system_packages=("ffmpeg", "libsndfile"),
        notes="BirdNET embeddings are extracted per 3-second window.",
    )


class PerchBackend(AudioBackend):
    spec = AudioBackendSpec(
        name="perch",
        window_seconds=5.0,
        overlap_seconds=0.0,
        embedding_scope="window",
        required_python_packages=("perch-hoplite", "tensorflow", "tensorflow-hub", "jax"),
        required_system_packages=("ffmpeg", "libsndfile"),
        notes="Perch-Hoplite style embeddings are extracted per 5-second window.",
    )


BACKENDS: dict[str, type[AudioBackend]] = {
    "wav2vec2": Wav2Vec2Backend,
    "birdnet": BirdNETBackend,
    "perch": PerchBackend,
}


def get_audio_backend_spec(name: str) -> AudioBackendSpec:
    """Return one backend spec by name. / backend 名から契約を返す。"""

    backend_cls = BACKENDS.get(name)
    if backend_cls is None:
        raise KeyError(f"Unknown audio backend: {name}")
    return backend_cls.spec


def list_audio_backends() -> list[AudioBackendSpec]:
    """Return all known backend specs. / 知っている backend 一覧を返す。"""

    return [backend_cls.spec for backend_cls in BACKENDS.values()]
