from __future__ import annotations

import unittest

import numpy as np

from multi_bird_db.multimodal.expanders import build_multimodal_feature_matrix
from multi_bird_db.multimodal.labels import TaxonLabelAssignment
from multi_bird_db.multimodal.types import (
    AudioEmbeddingRow,
    AudioEmbeddingRun,
    GraphEmbeddingRun,
    LanguageEmbeddingRun,
    LanguageSurfaceRow,
)


class MultimodalExpanderTests(unittest.TestCase):
    def test_build_multimodal_feature_matrix_expands_qid_local_cartesian_product(self) -> None:
        graph_run = GraphEmbeddingRun(
            run_dir=None,  # type: ignore[arg-type]
            qids=['Q1'],
            embeddings=np.asarray([[10.0, 11.0]], dtype=np.float32),
            metadata={},
        )
        audio_run = AudioEmbeddingRun(
            run_dir=None,  # type: ignore[arg-type]
            audio_ids=['a1', 'a2', 'a3'],
            qids=['Q1', 'Q1', 'Q1'],
            embeddings=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            rows=[
                AudioEmbeddingRow(0, 'a1', 'Q1', '', '', 0, {}),
                AudioEmbeddingRow(1, 'a2', 'Q1', '', '', 0, {}),
                AudioEmbeddingRow(2, 'a3', 'Q1', '', '', 0, {}),
            ],
            metadata={},
        )
        language_run = LanguageEmbeddingRun(
            run_dir=None,  # type: ignore[arg-type]
            surface_ids=['s1', 's2'],
            qids=['Q1', 'Q1'],
            embeddings=np.asarray([[100.0], [200.0]], dtype=np.float32),
            rows=[
                LanguageSurfaceRow(0, 's1', 'Q1', 'en'),
                LanguageSurfaceRow(1, 's2', 'Q1', 'en'),
            ],
            qid_to_surfaces={'Q1': ['s1', 's2']},
            metadata={},
        )
        assignments = [
            TaxonLabelAssignment(qid='Q1', target_rank='family', label_qid='QFAM1', label_name='Family 1', distance_to_label=2)
        ]

        result = build_multimodal_feature_matrix(
            graph_run=graph_run,
            audio_run=audio_run,
            language_run=language_run,
            assignments=assignments,
            include_graph=True,
            include_audio=True,
            include_language=True,
        )

        self.assertEqual(tuple(result.matrix.embeddings.shape), (6, 4))
        self.assertEqual(len(result.matrix.rows), 6)
        self.assertEqual(result.feature_slices, {'graph': (0, 2), 'audio': (2, 3), 'language': (3, 4)})
        self.assertEqual({row.target_label for row in result.matrix.rows}, {'QFAM1'})
