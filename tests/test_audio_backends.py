from __future__ import annotations

import unittest

from multi_bird_db.audio_backends import get_audio_backend_spec, list_audio_backends
from multi_bird_db.audio_windows import segment_waveform

import torch


class AudioBackendSpecTests(unittest.TestCase):
    def test_registry_includes_planned_backends(self) -> None:
        names = [spec.name for spec in list_audio_backends()]
        self.assertEqual(names, ["wav2vec2", "birdnet", "perch"])

    def test_backend_contract_values(self) -> None:
        birdnet = get_audio_backend_spec("birdnet")
        perch = get_audio_backend_spec("perch")
        self.assertEqual(birdnet.window_seconds, 3.0)
        self.assertEqual(perch.window_seconds, 5.0)
        self.assertEqual(birdnet.embedding_scope, "window")
        self.assertEqual(perch.embedding_scope, "window")

    def test_segment_waveform(self) -> None:
        waveform = torch.arange(10, dtype=torch.float32)
        windows = segment_waveform(waveform, sample_rate=2, window_seconds=3.0, overlap_seconds=1.0)
        self.assertGreaterEqual(len(windows), 1)
        self.assertEqual(windows[0].index, 0)
        self.assertAlmostEqual(windows[0].start_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
