from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import networkx as nx

from multi_bird_db.embeddings import (
    build_gcn_embeddings,
    build_grace_embeddings,
    build_graphsage_embeddings,
    build_node2vec_embeddings,
    build_transe_embeddings,
    save_embedding_store,
)


class GraphEmbeddingTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("Q1", "Q2")
        graph.add_edge("Q2", "Q3")
        graph.graph["graph_type"] = "taxonomy_digraph"
        graph.graph["root_qid"] = "Q1"
        self.graph = graph

    def test_node2vec_records_training_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_node2vec_embeddings(
                self.graph,
                dim=8,
                walk_length=4,
                num_walks=2,
                window_size=1,
                negative_samples=1,
                epochs=3,
                learning_rate=0.01,
                seed=42,
                undirected=True,
            )
            output_dir = Path(tmpdir)
            save_embedding_store(store, output_dir)

            trace = store.metadata["training_trace"]
            self.assertEqual(len(trace), 3)
            self.assertTrue(all("average_loss" in item for item in trace))
            self.assertTrue(all(item["average_loss"] >= 0 for item in trace))
            self.assertTrue((output_dir / "loss_curve.png").exists())

    def test_transe_records_training_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_transe_embeddings(
                self.graph,
                dim=8,
                epochs=3,
                learning_rate=0.01,
                margin=1.0,
                negative_samples=1,
                seed=42,
                root_qid="Q1",
            )
            output_dir = Path(tmpdir)
            save_embedding_store(store, output_dir)

            trace = store.metadata["training_trace"]
            self.assertEqual(len(trace), 3)
            self.assertTrue(all("average_loss" in item for item in trace))
            self.assertTrue(all(item["average_loss"] >= 0 for item in trace))
            self.assertTrue((output_dir / "loss_curve.png").exists())

    def test_gcn_records_training_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_gcn_embeddings(
                self.graph,
                dim=8,
                layers=2,
                residual=0.2,
                epochs=3,
                learning_rate=0.01,
                negative_samples=1,
                feature_mode="degree",
                seed=42,
                root_qid="Q1",
                undirected=True,
            )
            output_dir = Path(tmpdir)
            save_embedding_store(store, output_dir)

            trace = store.metadata["training_trace"]
            self.assertEqual(len(trace), 3)
            self.assertTrue(all("average_loss" in item for item in trace))
            self.assertTrue(all(item["average_loss"] >= 0 for item in trace))
            self.assertTrue((output_dir / "loss_curve.png").exists())

    def test_grace_records_training_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_grace_embeddings(
                self.graph,
                dim=8,
                proj_dim=8,
                layers=2,
                residual=0.2,
                epochs=3,
                learning_rate=0.01,
                tau=0.5,
                drop_edge_rate_1=0.2,
                drop_edge_rate_2=0.3,
                drop_feature_rate_1=0.1,
                drop_feature_rate_2=0.2,
                batch_size=2,
                encoder_type="gcn",
                feature_mode="degree",
                seed=42,
                root_qid="Q1",
                undirected=True,
            )
            output_dir = Path(tmpdir)
            save_embedding_store(store, output_dir)

            trace = store.metadata["training_trace"]
            self.assertEqual(len(trace), 3)
            self.assertTrue(all("average_loss" in item for item in trace))
            self.assertTrue(all(item["average_loss"] >= 0 for item in trace))
            self.assertTrue(all("contrastive_loss" in item for item in trace))
            self.assertEqual(store.metadata["objective"], "contrastive_learning")
            self.assertEqual(store.metadata["algorithm"], "grace")
            self.assertTrue((output_dir / "loss_curve.png").exists())

    def test_graphsage_records_training_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = build_graphsage_embeddings(
                self.graph,
                dim=8,
                layers=2,
                residual=0.2,
                epochs=3,
                learning_rate=0.01,
                negative_samples=1,
                num_neighbors_1=2,
                num_neighbors_2=1,
                feature_mode="degree",
                seed=42,
                root_qid="Q1",
                undirected=True,
            )
            output_dir = Path(tmpdir)
            save_embedding_store(store, output_dir)

            trace = store.metadata["training_trace"]
            self.assertEqual(len(trace), 3)
            self.assertTrue(all("average_loss" in item for item in trace))
            self.assertTrue(all(item["average_loss"] >= 0 for item in trace))
            self.assertTrue((output_dir / "loss_curve.png").exists())


if __name__ == "__main__":
    unittest.main()
