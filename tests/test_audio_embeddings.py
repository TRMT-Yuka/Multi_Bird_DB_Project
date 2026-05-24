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


if __name__ == "__main__":
    unittest.main()
