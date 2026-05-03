from __future__ import annotations

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_bird_db.language_embeddings import build_language_embeddings, build_language_surface_manifests


class FakeEncoder:
    def __init__(self, model_name: str, device: str | None, max_length: int, batch_size: int) -> None:
        self.model_name = model_name
        self._device = device or "cpu"
        self.max_length = max_length
        self.batch_size = batch_size

    @property
    def resolved_device(self) -> str:
        return self._device

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            length = float(len(text))
            vectors.append(np.array([length, length + 1.0], dtype=np.float32))
        return np.vstack(vectors) if vectors else np.zeros((0, 2), dtype=np.float32)


class LanguageSurfaceManifestTests(unittest.TestCase):
    def test_build_language_surface_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ontology_path = tmp_path / "bird_ontology.pkl"
            output_dir = tmp_path / "embeddings" / "language"

            rows = [
                {
                    "qid": "Q1",
                    "en_name": "Bird One",
                    "en_aliases": json.dumps(["Bird One", "Birdo"], ensure_ascii=False),
                    "enwiki_title": "Bird One wiki",
                    "ja_name": "鳥一",
                    "ja_aliases": json.dumps(["鳥一", "トリイチ"], ensure_ascii=False),
                    "jawiki_title": "鳥一ウィキ",
                },
                {
                    "qid": "Q2",
                    "en_name": "",
                    "en_aliases": "[]",
                    "enwiki_title": "",
                    "ja_name": "鳥二",
                    "ja_aliases": "[]",
                    "jawiki_title": "鳥二ウィキ",
                },
            ]
            with ontology_path.open("wb") as handle:
                pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

            build_language_surface_manifests(ontology_path, output_dir)

            en_manifest_path = output_dir / "en" / "surface_manifest.tsv"
            ja_manifest_path = output_dir / "ja" / "surface_manifest.tsv"
            en_qid_to_surfaces_path = output_dir / "en" / "qid_to_surfaces.json"
            ja_qid_to_surfaces_path = output_dir / "ja" / "qid_to_surfaces.json"
            self.assertTrue(en_manifest_path.exists())
            self.assertTrue(ja_manifest_path.exists())
            self.assertTrue(en_qid_to_surfaces_path.exists())
            self.assertTrue(ja_qid_to_surfaces_path.exists())

            with en_manifest_path.open("r", encoding="utf-8", newline="") as handle:
                en_rows = list(csv.DictReader(handle, delimiter="\t"))
            with ja_manifest_path.open("r", encoding="utf-8", newline="") as handle:
                ja_rows = list(csv.DictReader(handle, delimiter="\t"))

            self.assertEqual(len(en_rows), 4)
            self.assertEqual(len(ja_rows), 6)
            self.assertEqual({row["qid"] for row in en_rows}, {"Q1"})
            self.assertEqual({row["qid"] for row in ja_rows}, {"Q1", "Q2"})
            self.assertEqual(sum(1 for row in en_rows if row["surface_text"] == "Bird One"), 2)
            self.assertEqual(sum(1 for row in ja_rows if row["surface_text"] == "鳥一"), 2)
            self.assertEqual([row["surface_id"] for row in en_rows], [f"Q1_en_{i}" for i in range(4)])
            self.assertEqual([row["surface_id"] for row in ja_rows[:4]], [f"Q1_ja_{i}" for i in range(4)])
            self.assertEqual([row["surface_id"] for row in ja_rows[4:]], ["Q2_ja_0", "Q2_ja_1"])

            en_qid_to_surfaces = json.loads(en_qid_to_surfaces_path.read_text(encoding="utf-8"))
            ja_qid_to_surfaces = json.loads(ja_qid_to_surfaces_path.read_text(encoding="utf-8"))
            self.assertEqual(list(en_qid_to_surfaces), ["Q1"])
            self.assertEqual(list(ja_qid_to_surfaces), ["Q1", "Q2"])
            self.assertEqual([item["surface_id"] for item in en_qid_to_surfaces["Q1"]], [f"Q1_en_{i}" for i in range(4)])
            self.assertEqual([item["surface_text"] for item in ja_qid_to_surfaces["Q2"]], ["鳥二", "鳥二ウィキ"])

            summary = json.loads((output_dir / "en" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["language"], "en")
            self.assertEqual(summary["item_count"], 4)
            self.assertEqual(summary["unique_qid_count"], 1)
            self.assertEqual(summary["source_counts"]["en_name"], 1)
            self.assertEqual(summary["source_counts"]["en_aliases"], 2)
            self.assertEqual(summary["source_counts"]["enwiki_title"], 1)

            ja_summary = json.loads((output_dir / "ja" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(ja_summary["language"], "ja")
            self.assertEqual(ja_summary["item_count"], 6)
            self.assertEqual(ja_summary["unique_qid_count"], 2)

    def test_build_language_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ontology_path = tmp_path / "bird_ontology.pkl"
            output_dir = tmp_path / "embeddings" / "language"

            rows = [
                {
                    "qid": "Q1",
                    "en_name": "Bird One",
                    "en_aliases": json.dumps(["Bird One", "Birdo"], ensure_ascii=False),
                    "enwiki_title": "Bird One wiki",
                    "ja_name": "鳥一",
                    "ja_aliases": json.dumps(["鳥一", "トリイチ"], ensure_ascii=False),
                    "jawiki_title": "鳥一ウィキ",
                }
            ]
            with ontology_path.open("wb") as handle:
                pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

            result = build_language_embeddings(
                ontology_path,
                output_dir,
                english_model="en-model",
                japanese_model="ja-model",
                batch_size=4,
                max_length=16,
                device="cpu",
                encoder_factory=lambda model_name, device, max_length, batch_size: FakeEncoder(
                    model_name,
                    device,
                    max_length,
                    batch_size,
                ),
            )

            en_embeddings = np.load(output_dir / "en" / "embeddings.npy")
            ja_embeddings = np.load(output_dir / "ja" / "embeddings.npy")
            self.assertEqual(en_embeddings.shape, (4, 2))
            self.assertEqual(ja_embeddings.shape, (4, 2))
            with (output_dir / "en" / "surface_manifest.tsv").open("r", encoding="utf-8", newline="") as handle:
                en_manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
            with (output_dir / "ja" / "surface_manifest.tsv").open("r", encoding="utf-8", newline="") as handle:
                ja_manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([float(row[0]) for row in en_embeddings[:, :1]], [float(len(row["surface_text"])) for row in en_manifest_rows])
            self.assertEqual([float(row[0]) for row in ja_embeddings[:, :1]], [float(len(row["surface_text"])) for row in ja_manifest_rows])

            en_summary = json.loads((output_dir / "en" / "summary.json").read_text(encoding="utf-8"))
            ja_summary = json.loads((output_dir / "ja" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(en_summary["kind"], "language_bert_embeddings")
            self.assertEqual(en_summary["encoder_model"], "en-model")
            self.assertEqual(en_summary["embedding_dim"], 2)
            self.assertEqual(ja_summary["encoder_model"], "ja-model")
            self.assertEqual(result["en"]["encoder_model"], "en-model")
            self.assertEqual(result["ja"]["encoder_model"], "ja-model")


if __name__ == "__main__":
    unittest.main()
