from __future__ import annotations

import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import networkx as nx
import numpy as np

from multi_bird_db.multimodal.cli import main


class MultimodalCliTests(unittest.TestCase):
    def test_run_baseline_writes_outputs_for_toy_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            graph_dir = root / "graph"
            audio_dir = root / "audio"
            language_dir = root / "language"
            output_dir = root / "output"
            taxonomy_graph_path = root / "taxonomy.pkl"
            for directory in (graph_dir, audio_dir, language_dir):
                directory.mkdir(parents=True, exist_ok=True)

            (graph_dir / "qids.json").write_text(json.dumps(["Q1", "Q2"]), encoding="utf-8")
            np.save(graph_dir / "embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            (graph_dir / "metadata.json").write_text(json.dumps({"algorithm": "graphsage"}), encoding="utf-8")

            (audio_dir / "audio_ids.json").write_text(json.dumps(["a1", "a2"]), encoding="utf-8")
            (audio_dir / "qids.json").write_text(json.dumps(["Q1", "Q2"]), encoding="utf-8")
            np.save(audio_dir / "embeddings.npy", np.asarray([[0.1], [0.9]], dtype=np.float32))
            (audio_dir / "metadata.json").write_text(json.dumps({"backend": "wav2vec2"}), encoding="utf-8")
            (audio_dir / "audio_manifest.tsv").write_text(
                "audio_id\tqid\tsource_path\trelative_path\twindow_index\tembedding_index\n"
                "a1\tQ1\t/tmp/a1.mp3\tQ1/a1.mp3\t0\t0\n"
                "a2\tQ2\t/tmp/a2.mp3\tQ2/a2.mp3\t0\t1\n",
                encoding="utf-8",
            )

            (language_dir / "surface_ids.json").write_text(json.dumps(["s1", "s2"]), encoding="utf-8")
            (language_dir / "qids.json").write_text(json.dumps(["Q1", "Q2"]), encoding="utf-8")
            (language_dir / "qid_to_surfaces.json").write_text(json.dumps({"Q1": ["s1"], "Q2": ["s2"]}), encoding="utf-8")
            np.save(language_dir / "embeddings.npy", np.asarray([[0.2], [0.8]], dtype=np.float32))
            (language_dir / "metadata.json").write_text(json.dumps({"language": "en"}), encoding="utf-8")
            (language_dir / "surface_manifest.tsv").write_text(
                "surface_id\tqid\tlanguage\tordinal\tsurface_text\tsource\tsource_index\n"
                "s1\tQ1\ten\t0\tBird One\ten_name\t0\n"
                "s2\tQ2\ten\t0\tBird Two\ten_name\t0\n",
                encoding="utf-8",
            )

            graph = nx.DiGraph()
            graph.add_node("QFAM1", parent_taxon="", taxon_rank_name="family", label_en="Family 1")
            graph.add_node("QFAM2", parent_taxon="", taxon_rank_name="family", label_en="Family 2")
            graph.add_node("Q1", parent_taxon="QFAM1", taxon_rank_name="species", label_en="Bird One")
            graph.add_node("Q2", parent_taxon="QFAM2", taxon_rank_name="species", label_en="Bird Two")
            with taxonomy_graph_path.open("wb") as handle:
                pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with redirect_stdout(io.StringIO()):
                exit_code = main([
                    "run-baseline",
                    "--graph-embedding-dir", str(graph_dir),
                    "--audio-embedding-dir", str(audio_dir),
                    "--language-embedding-dir", str(language_dir),
                    "--taxonomy-graph", str(taxonomy_graph_path),
                    "--target-rank", "family",
                    "--validation-fraction", "0.0",
                    "--test-fraction", "0.0",
                    "--num-epochs", "10",
                    "--output-dir", str(output_dir),
                ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "dataset_summary.json").exists())
            self.assertTrue((output_dir / "metrics.tsv").exists())
            self.assertTrue((output_dir / "predictions.tsv").exists())
            self.assertTrue((output_dir / "training_trace.json").exists())


if __name__ == "__main__":
    unittest.main()
