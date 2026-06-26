from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_bird_db import audio_embeddings


class AudioEmbeddingTests(unittest.TestCase):
    def test_discover_and_embed_audio_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            (qid_dir / "288057.mp3").write_bytes(b"stub-audio-one")
            (qid_dir / "119100.mp3").write_bytes(b"stub-audio-two")

            decoded_lengths: list[int] = []

            def fake_loader(path: Path) -> tuple[np.ndarray, int]:
                payload = path.read_bytes()
                decoded_lengths.append(len(payload))
                return np.arange(len(payload), dtype=np.float32), 16000

            class FakeEncoder:
                sample_rate = 16000

                def encode_batch(self, waveforms):
                    rows = []
                    for waveform in waveforms:
                        rows.append(
                            np.array(
                                [
                                    float(waveform.numel()),
                                    float(waveform[:1].item() if waveform.numel() else 0.0),
                                    float(waveform.mean().item() if waveform.numel() else 0.0),
                                ],
                                dtype=np.float32,
                            )
                        )
                    return np.vstack(rows)

            output_dir = root / "embeddings"
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                model_name="unit-test-model",
                device="cpu",
                batch_size=1,
                max_seconds=30.0,
                target_sample_rate=16000,
                extensions=("mp3",),
                cache_dir=None,
                audio_loader=fake_loader,
                encoder=FakeEncoder(),
            )

            run_dir = Path(result["summary"]["run_dir"])
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "embeddings.npy").exists())
            self.assertTrue((run_dir / "audio_ids.json").exists())
            self.assertTrue((run_dir / "qids.json").exists())
            self.assertTrue((run_dir / "audio_manifest.tsv").exists())
            self.assertTrue((run_dir / "metadata.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "failed_items.json").exists())

            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (2, 3))
            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_ids, ["Q122868_119100", "Q122868_288057"])
            qids = json.loads((run_dir / "qids.json").read_text(encoding="utf-8"))
            self.assertEqual(qids, ["Q122868", "Q122868"])
            self.assertEqual(result["summary"]["item_count"], 2)
            self.assertEqual(result["summary"]["failed_count"], 0)
            self.assertEqual(len(decoded_lengths), 2)

            manifest_rows = result["manifest_rows"]
            self.assertEqual(len(manifest_rows), 2)
            self.assertEqual(manifest_rows[0]["embedding_index"], "0")
            self.assertEqual(manifest_rows[1]["embedding_index"], "1")

    def test_birdnet_backend_uses_three_second_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            (qid_dir / "111.mp3").write_bytes(b"stub-audio-one")
            (qid_dir / "222.mp3").write_bytes(b"stub-audio-two")

            def fake_loader(path: Path) -> tuple[np.ndarray, int]:
                if path.name == "111.mp3":
                    return np.linspace(-1.0, 1.0, 5 * 48000, dtype=np.float32), 48000
                return np.linspace(-0.5, 0.5, 2 * 48000, dtype=np.float32), 48000

            class FakeBirdNETEncoder:
                sample_rate = 48000
                model_type = "acoustic"
                model_version = "2.4"
                backend = "tf"

                def encode_batch(self, waveforms):
                    rows = []
                    for index, waveform in enumerate(waveforms):
                        rows.append(
                            np.array(
                                [
                                    float(index),
                                    float(waveform.numel()),
                                    float(waveform.mean().item()),
                                ],
                                dtype=np.float32,
                            )
                        )
                    return np.vstack(rows)

            output_dir = root / "embeddings"
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                backend="birdnet",
                model_name="birdnet-unit-test",
                device="cpu",
                batch_size=2,
                max_seconds=10.0,
                target_sample_rate=48000,
                extensions=("mp3",),
                cache_dir=None,
                audio_loader=fake_loader,
                encoder=FakeBirdNETEncoder(),
            )

            run_dir = Path(result["summary"]["run_dir"])
            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (3, 3))

            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_ids, ["Q122868_111_w0000", "Q122868_111_w0001", "Q122868_222_w0000"])

            manifest_rows = result["manifest_rows"]
            self.assertEqual(manifest_rows[0]["window_index"], "0")
            self.assertEqual(manifest_rows[0]["window_seconds"], "3.000000")
            self.assertEqual(manifest_rows[1]["window_index"], "1")
            self.assertEqual(manifest_rows[2]["window_index"], "0")
            self.assertEqual(result["summary"]["item_count"], 3)
            self.assertEqual(result["summary"]["failed_count"], 0)

    def test_perch_backend_uses_five_second_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "trimmed_xeno_data"
            qid_dir = input_dir / "Q122868"
            qid_dir.mkdir(parents=True)
            (qid_dir / "333.mp3").write_bytes(b"stub-audio-three")

            def fake_loader(path: Path) -> tuple[np.ndarray, int]:
                self.assertEqual(path.name, "333.mp3")
                return np.linspace(-1.0, 1.0, 7 * 22050, dtype=np.float32), 22050

            class FakePerchEncoder:
                sample_rate = 22050
                model_type = "Perch2"
                model_version = ""
                backend = "bioacoustics-model-zoo"

                def encode_batch(self, waveforms):
                    rows = []
                    for index, waveform in enumerate(waveforms):
                        rows.append(
                            np.array(
                                [
                                    float(index),
                                    float(waveform.numel()),
                                    float(waveform.mean().item()),
                                    float(waveform[-1].item()),
                                ],
                                dtype=np.float32,
                            )
                        )
                    return np.vstack(rows)

            output_dir = root / "embeddings"
            result = audio_embeddings.build_audio_embeddings(
                input_dir=input_dir,
                output_dir=output_dir,
                backend="perch",
                model_name="perch-unit-test",
                device="cpu",
                batch_size=2,
                max_seconds=10.0,
                target_sample_rate=22050,
                extensions=("mp3",),
                cache_dir=None,
                audio_loader=fake_loader,
                encoder=FakePerchEncoder(),
            )

            run_dir = Path(result["summary"]["run_dir"])
            embeddings = np.load(run_dir / "embeddings.npy")
            self.assertEqual(embeddings.shape, (2, 4))

            audio_ids = json.loads((run_dir / "audio_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(audio_ids, ["Q122868_333_w0000", "Q122868_333_w0001"])

            manifest_rows = result["manifest_rows"]
            self.assertEqual(manifest_rows[0]["window_index"], "0")
            self.assertEqual(manifest_rows[1]["window_index"], "1")
            self.assertEqual(manifest_rows[0]["window_seconds"], "5.000000")
            self.assertEqual(manifest_rows[1]["window_seconds"], "5.000000")
            self.assertEqual(result["summary"]["item_count"], 2)
            self.assertEqual(result["summary"]["failed_count"], 0)


if __name__ == "__main__":
    unittest.main()
