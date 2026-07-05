from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_bird_db.multimodal.loaders import (
    load_audio_embedding_run,
    load_graph_embedding_run,
    load_language_embedding_run,
)


class MultimodalLoaderTests(unittest.TestCase):
    def test_load_graph_embedding_run_reads_qids_and_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / "qids.json").write_text(json.dumps(["Q1", "Q2"]), encoding="utf-8")
            np.save(run_dir / "embeddings.npy", np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            (run_dir / "metadata.json").write_text(json.dumps({"algorithm": "graphsage"}), encoding="utf-8")

            loaded = load_graph_embedding_run(run_dir)

            self.assertEqual(loaded.qids, ["Q1", "Q2"])
            self.assertEqual(loaded.metadata["algorithm"], "graphsage")
            self.assertEqual(tuple(loaded.embeddings.shape), (2, 2))

    def test_load_audio_embedding_run_reads_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / "audio_ids.json").write_text(json.dumps(["a1", "a2"]), encoding="utf-8")
            (run_dir / "qids.json").write_text(json.dumps(["Q1", "Q1"]), encoding="utf-8")
            np.save(run_dir / "embeddings.npy", np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            (run_dir / "metadata.json").write_text(json.dumps({"backend": "wav2vec2"}), encoding="utf-8")
            (run_dir / "audio_manifest.tsv").write_text(
                "\t".join(["audio_id", "qid", "source_path", "relative_path", "window_index", "embedding_index"])
                + "\n"
                + "\t".join(["a1", "Q1", "/tmp/one.mp3", "Q1/one.mp3", "0", "0"])
                + "\n"
                + "\t".join(["a2", "Q1", "/tmp/two.mp3", "Q1/two.mp3", "1", "1"])
                + "\n",
                encoding="utf-8",
            )

            loaded = load_audio_embedding_run(run_dir)

            self.assertEqual(loaded.audio_ids, ["a1", "a2"])
            self.assertEqual(len(loaded.rows), 2)
            self.assertEqual(loaded.rows[1].relative_path, "Q1/two.mp3")
            self.assertEqual(loaded.rows[1].window_index, 1)

    def test_load_language_embedding_run_uses_surface_manifest_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / "surface_ids.json").write_text(json.dumps(["Q1_en_0", "Q1_en_1"]), encoding="utf-8")
            (run_dir / "qids.json").write_text(json.dumps(["Q1", "Q1"]), encoding="utf-8")
            (run_dir / "qid_to_surfaces.json").write_text(json.dumps({"Q1": ["Q1_en_0", "Q1_en_1"]}), encoding="utf-8")
            np.save(run_dir / "embeddings.npy", np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            (run_dir / "metadata.json").write_text(json.dumps({"language": "en"}), encoding="utf-8")
            (run_dir / "surface_manifest.tsv").write_text(
                "\t".join(["surface_id", "qid", "language", "ordinal", "surface_text", "source", "source_index"])
                + "\n"
                + "\t".join(["Q1_en_0", "Q1", "en", "0", "Bird name", "en_name", "0"])
                + "\n"
                + "\t".join(["Q1_en_1", "Q1", "en", "1", "Bird alias", "en_aliases", "1"])
                + "\n",
                encoding="utf-8",
            )

            loaded = load_language_embedding_run(run_dir)

            self.assertEqual(len(loaded.rows), 2)
            self.assertEqual(loaded.rows[0].surface_text, "Bird name")
            self.assertEqual(loaded.rows[1].source, "en_aliases")


if __name__ == "__main__":
    unittest.main()
