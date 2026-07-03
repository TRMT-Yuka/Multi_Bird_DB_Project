from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from multi_bird_db import audio_finetuning


class AudioFineTuningTests(unittest.TestCase):
    def test_build_examples_uses_recording_map_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            root = Path(tmpdir_str)
            audio_root = root / "audio"
            qid_dir = audio_root / "Q1"
            qid_dir.mkdir(parents=True)
            (qid_dir / "111.mp3").write_bytes(b"stub")
            recording_map = root / "recording_map.json"
            recording_map.write_text(
                json.dumps(
                    [
                        {
                            "qid": "Q1",
                            "xeno_canto_species_id": "Corvus-corax",
                            "recording_ids": ["111"],
                            "download_urls": ["https://example.org/111"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            rows = audio_finetuning.build_finetune_examples(audio_root, recording_map)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].qid, "Q1")
            self.assertEqual(rows[0].recording_id, "111")
            self.assertEqual(rows[0].xeno_canto_species_id, "Corvus-corax")
            self.assertEqual(rows[0].download_url, "https://example.org/111")

    def test_assign_crossval_folds_keeps_singletons_train_only(self) -> None:
        rows = [
            audio_finetuning.FineTuneExample("Q1", "a", "/tmp/Q1/a.mp3", "Q1/a.mp3", "", "", None, False),
            audio_finetuning.FineTuneExample("Q1", "b", "/tmp/Q1/b.mp3", "Q1/b.mp3", "", "", None, False),
            audio_finetuning.FineTuneExample("Q2", "c", "/tmp/Q2/c.mp3", "Q2/c.mp3", "", "", None, False),
        ]
        assigned = audio_finetuning.assign_crossval_folds(rows, num_folds=5, seed=1)
        singleton = [row for row in assigned if row.qid == "Q2"][0]
        self.assertTrue(singleton.train_only_all_folds)
        self.assertIsNone(singleton.fold_index)
        regular = [row for row in assigned if row.qid == "Q1"]
        self.assertEqual({row.fold_index for row in regular}, {0, 1})

    def test_build_fold_splits_exposes_singletons_only_in_train(self) -> None:
        rows = [
            audio_finetuning.FineTuneExample("Q1", "a", "/tmp/Q1/a.mp3", "Q1/a.mp3", "", "", 0, False),
            audio_finetuning.FineTuneExample("Q1", "b", "/tmp/Q1/b.mp3", "Q1/b.mp3", "", "", 1, False),
            audio_finetuning.FineTuneExample("Q2", "c", "/tmp/Q2/c.mp3", "Q2/c.mp3", "", "", None, True),
        ]
        splits = audio_finetuning.build_fold_splits(rows, num_folds=2)
        self.assertEqual(len(splits[0].test_examples), 1)
        self.assertEqual({row.qid for row in splits[0].train_examples}, {"Q1", "Q2"})
        self.assertEqual({row.qid for row in splits[1].train_examples}, {"Q1", "Q2"})
        self.assertEqual({row.qid for row in splits[1].test_examples}, {"Q1"})

    def test_write_training_curves_creates_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            output_dir = Path(tmpdir_str)
            result = audio_finetuning._write_training_curves(
                output_dir,
                [
                    {"epoch": 1, "train_loss": 1.5, "eval_loss": 1.2, "eval_accuracy": 0.25},
                    {"epoch": 2, "train_loss": 1.0, "eval_loss": 0.8, "eval_accuracy": 0.5},
                ],
            )
            self.assertTrue((output_dir / "loss_curve.png").exists())
            self.assertTrue((output_dir / "accuracy_curve.png").exists())
            self.assertEqual(result["loss_curve_png"], str(output_dir / "loss_curve.png"))
            self.assertEqual(result["accuracy_curve_png"], str(output_dir / "accuracy_curve.png"))


if __name__ == "__main__":
    unittest.main()
