from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from multi_bird_db.embeddings import EmbeddingStore, save_embedding_store
from multi_bird_db.graph_evaluation import evaluate_graph_embeddings


class GraphEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = nx.DiGraph()
        graph.add_node("Q1", taxon_rank_name="species")
        graph.add_node("Q2", taxon_rank_name="species")
        graph.add_node("Q3", taxon_rank_name="genus")
        graph.add_node("Q4", taxon_rank_name="genus")
        graph.add_edge("Q3", "Q1", relation_type="parent_taxon")
        graph.add_edge("Q3", "Q2", relation_type="parent_taxon")
        graph.add_edge("Q4", "Q3", relation_type="parent_taxon")
        self.graph = graph

    def _write_store(self, root: Path, method: str, embeddings: np.ndarray) -> None:
        store = EmbeddingStore(
            qids=["Q1", "Q2", "Q3", "Q4"],
            embeddings=embeddings,
            metadata={
                "algorithm": method,
                "implementation": f"{method}_test",
            },
        )
        save_embedding_store(store, root / method)

    def test_evaluate_graph_embeddings_writes_report_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            graph_path = tmp / "graph.pkl"
            embeddings_root = tmp / "embeddings"
            output_root = tmp / "evaluation"

            with graph_path.open("wb") as handle:
                import pickle

                pickle.dump(self.graph, handle)

            embeddings_root.mkdir(parents=True, exist_ok=True)
            self._write_store(
                embeddings_root,
                "node2vec",
                np.asarray(
                    [
                        [0.0, 0.0, 0.1, 0.1],
                        [0.1, 0.0, 0.1, 0.2],
                        [5.0, 5.0, 5.1, 5.1],
                        [5.1, 5.0, 5.2, 5.1],
                    ],
                    dtype=np.float32,
                ),
            )
            self._write_store(
                embeddings_root,
                "grace",
                np.asarray(
                    [
                        [0.0, 0.1, 0.0, 0.1],
                        [0.2, 0.0, 0.1, 0.0],
                        [4.9, 5.0, 5.1, 5.0],
                        [5.0, 4.9, 5.0, 5.1],
                    ],
                    dtype=np.float32,
                ),
            )

            report = evaluate_graph_embeddings(
                graph_path=graph_path,
                embeddings_root=embeddings_root,
                output_root=output_root,
                label_field="taxon_rank_name",
                methods=["node2vec", "grace"],
                seed=7,
                silhouette_sample_size=10,
            )

            self.assertEqual(report["label_field"], "taxon_rank_name")
            self.assertEqual(set(report["methods"]), {"node2vec", "grace"})
            self.assertIn("best_by_metric", report)
            self.assertTrue((output_root / "metrics" / "clustering_metrics.csv").exists())
            self.assertTrue((output_root / "metrics" / "summary_metrics.csv").exists())
            self.assertTrue((output_root / "plots" / "clustering_metrics_barplot.png").exists())
            self.assertTrue((output_root / "logs" / "node2vec_cluster_assignments.tsv").exists())
            self.assertTrue((output_root / "logs" / "grace_cluster_assignments.tsv").exists())
            self.assertTrue((output_root / "report" / "experiment_report.md").exists())
            self.assertTrue((output_root / "report" / "experiment_report.json").exists())

            markdown = (output_root / "report" / "experiment_report.md").read_text(encoding="utf-8")
            self.assertIn("Graph Clustering Report", markdown)
            self.assertIn("node2vec", markdown)
            self.assertIn("grace", markdown)


if __name__ == "__main__":
    unittest.main()
