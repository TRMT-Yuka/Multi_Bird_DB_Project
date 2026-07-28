from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from multi_bird_db import audio_repair


class AudioRepairTests(unittest.TestCase):
    def test_repair_replaces_only_after_candidate_decodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            root = Path(tmpdir_str)
            audio_root = root / "audio"
            qid_dir = audio_root / "Q1"
            qid_dir.mkdir(parents=True)
            audio_path = qid_dir / "111.mp3"
            audio_path.write_bytes(b"bad")
            recording_map = root / "recording_map.json"
            recording_map.write_text(
                json.dumps(
                    [
                        {
                            "qid": "Q1",
                            "xeno_canto_species_id": "Corvus-corax",
                            "recording_ids": ["111"],
                            "download_urls": ["https://example.org/111/download"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = root / "repair.tsv"

            def fake_load_audio(path: Path, *, target_sample_rate: int, max_seconds: float):
                if path.read_bytes() == b"bad":
                    raise RuntimeError("invalid audio")
                return np.ones(16, dtype=np.float32), target_sample_rate

            def fake_clip(input_path: Path, output_path: Path, file_type: str, clip_seconds: int) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(input_path.read_bytes())

            with (
                mock.patch.object(audio_repair, "load_audio_file", side_effect=fake_load_audio),
                mock.patch.object(audio_repair, "fetch_bytes", return_value=b"good"),
                mock.patch.object(audio_repair, "probe_audio_duration_seconds", return_value=30.0),
                mock.patch.object(audio_repair, "clip_audio_file", side_effect=fake_clip),
            ):
                summary = audio_repair.repair_xeno_canto_audio(
                    input_dir=audio_root,
                    recording_map_path=recording_map,
                    report_path=report,
                )

            self.assertEqual(summary["invalid_count"], 1)
            self.assertEqual(summary["repaired_count"], 1)
            self.assertEqual(summary["failed_count"], 0)
            self.assertEqual(audio_path.read_bytes(), b"good")
            with report.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(rows[0]["status"], "repaired")
            self.assertEqual(rows[0]["reason"], "invalid audio")

    def test_check_only_reports_invalid_without_replacing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            root = Path(tmpdir_str)
            audio_root = root / "audio"
            qid_dir = audio_root / "Q1"
            qid_dir.mkdir(parents=True)
            audio_path = qid_dir / "111.mp3"
            audio_path.write_bytes(b"bad")
            recording_map = root / "recording_map.json"
            recording_map.write_text(
                json.dumps([{"qid": "Q1", "recording_ids": ["111"], "download_urls": ["https://example.org/111"]}]),
                encoding="utf-8",
            )

            with mock.patch.object(audio_repair, "load_audio_file", side_effect=RuntimeError("invalid audio")):
                summary = audio_repair.repair_xeno_canto_audio(
                    input_dir=audio_root,
                    recording_map_path=recording_map,
                    report_path=root / "repair.tsv",
                    repair=False,
                )

            self.assertEqual(summary["invalid_count"], 1)
            self.assertEqual(summary["repaired_count"], 0)
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(audio_path.read_bytes(), b"bad")


if __name__ == "__main__":
    unittest.main()
