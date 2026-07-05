from __future__ import annotations

import unittest

import numpy as np

from multi_bird_db.multimodal.features import build_feature_matrix, concatenate_sample_vectors
from multi_bird_db.multimodal.types import MultimodalSampleRow


class MultimodalFeatureTests(unittest.TestCase):
    def test_concatenate_sample_vectors_preserves_fixed_modality_order(self) -> None:
        vector, slices = concatenate_sample_vectors(
            graph_vector=np.asarray([1.0, 2.0], dtype=np.float32),
            audio_vector=np.asarray([3.0], dtype=np.float32),
            language_vector=np.asarray([4.0, 5.0], dtype=np.float32),
        )

        np.testing.assert_array_equal(vector, np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
        self.assertEqual(slices, {"graph": (0, 2), "audio": (2, 3), "language": (3, 5)})

    def test_build_feature_matrix_stacks_row_aligned_vectors(self) -> None:
        rows_and_vectors = [
            (
                MultimodalSampleRow(
                    sample_id="s1",
                    qid="Q1",
                    graph_embedding_index=0,
                    audio_embedding_index=0,
                    language_embedding_index=0,
                    modality_pattern="graph+audio+language",
                    target_rank="family",
                    target_label="QFAM1",
                ),
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            ),
            (
                MultimodalSampleRow(
                    sample_id="s2",
                    qid="Q2",
                    graph_embedding_index=1,
                    audio_embedding_index=1,
                    language_embedding_index=1,
                    modality_pattern="graph+audio+language",
                    target_rank="family",
                    target_label="QFAM2",
                ),
                np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
            ),
        ]

        result = build_feature_matrix(rows_and_vectors)

        self.assertEqual(len(result.matrix.rows), 2)
        self.assertEqual(tuple(result.matrix.embeddings.shape), (2, 3))
        self.assertEqual(result.matrix.metadata["row_count"], 2)
        self.assertEqual(result.matrix.metadata["embedding_dim"], 3)


if __name__ == "__main__":
    unittest.main()
