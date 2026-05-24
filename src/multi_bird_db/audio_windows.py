from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One fixed-length audio window. / 1 つの固定長音声窓。"""

    index: int
    start_seconds: float
    end_seconds: float
    waveform: torch.Tensor


def segment_waveform(
    waveform: torch.Tensor,
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

    window_size = max(int(round(window_seconds * sample_rate)), 1)
    hop_size = max(int(round((window_seconds - overlap_seconds) * sample_rate)), 1)
    total_samples = int(waveform.numel())

    windows: list[AudioWindow] = []
    start = 0
    index = 0
    while start < total_samples:
        end = min(start + window_size, total_samples)
        chunk = waveform[start:end]
        if chunk.numel() < window_size:
            pad_length = window_size - chunk.numel()
            if pad_mode == "noise":
                padding = torch.randn(pad_length, dtype=chunk.dtype) * 0.005
            else:
                padding = torch.zeros(pad_length, dtype=chunk.dtype)
            chunk = torch.cat([chunk, padding], dim=0)
        windows.append(
            AudioWindow(
                index=index,
                start_seconds=start / sample_rate,
                end_seconds=end / sample_rate,
                waveform=chunk,
            )
        )
        if end >= total_samples:
            break
        start += hop_size
        index += 1
    return windows
