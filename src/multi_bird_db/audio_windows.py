from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One fixed-length audio window. / 1 つの固定長音声窓。"""

    index: int
    start_seconds: float
    end_seconds: float
    waveform: np.ndarray


def segment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    overlap_seconds: float = 0.0,
    pad_mode: str = "zeros",
) -> list[AudioWindow]:
    """Split one waveform into fixed-length windows. / 1 波形を固定長窓に分割する。"""

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be non-negative")
    if overlap_seconds >= window_seconds:
        raise ValueError("overlap_seconds must be smaller than window_seconds")

    waveform_array = np.asarray(waveform, dtype=np.float32).reshape(-1)
    window_size = max(int(round(window_seconds * sample_rate)), 1)
    hop_size = max(int(round((window_seconds - overlap_seconds) * sample_rate)), 1)
    total_samples = int(waveform_array.shape[0])

    windows: list[AudioWindow] = []
    start = 0
    index = 0
    while start < total_samples:
        end = min(start + window_size, total_samples)
        chunk = waveform_array[start:end]
        if chunk.shape[0] < window_size:
            pad_length = window_size - chunk.shape[0]
            if pad_mode == "noise":
                padding = (np.random.standard_normal(pad_length) * 0.005).astype(chunk.dtype, copy=False)
            else:
                padding = np.zeros(pad_length, dtype=chunk.dtype)
            chunk = np.concatenate([chunk, padding], axis=0)
        windows.append(
            AudioWindow(
                index=index,
                start_seconds=start / sample_rate,
                end_seconds=end / sample_rate,
                waveform=np.ascontiguousarray(chunk, dtype=np.float32),
            )
        )
        if end >= total_samples:
            break
        start += hop_size
        index += 1
    return windows
