from __future__ import annotations

import unittest

import numpy as np

from multi_bird_db.multimodal.evaluate import evaluate_predictions, labels_from_rows, select_rows_by_qids
from multi_bird_db.multimodal.types import MultimodalFeatureMatrix, MultimodalSampleRow


def _row(sample_id: str, qid: str, label: str) -> MultimodalSampleRow:
    return MultimodalSampleRow(
        sample_id=sample_id,
        qid=qid,
        graph_embedding_index=0,
        audio_embedding_index=0,
        language_embedding_index=0,
        modality_pattern='graph+audio+language',
        target_rank='family',
        target_label=label,
    )


class MultimodalEvaluationTests(unittest.TestCase):
    def test_select_rows_by_qids_preserves_alignment(self) -> None:
        matrix = MultimodalFeatureMatrix(
            rows=[_row('s1', 'Q1', 'A'), _row('s2', 'Q2', 'B'), _row('s3', 'Q1', 'A')],
            embeddings=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            metadata={},
        )

        selected = select_rows_by_qids(matrix, {'Q1'})

        self.assertEqual([row.sample_id for row in selected.rows], ['s1', 's3'])
        np.testing.assert_array_equal(selected.embeddings, np.asarray([[1.0], [3.0]], dtype=np.float32))

    def test_evaluate_predictions_reports_accuracy_and_macro_f1(self) -> None:
        rows = [_row('s1', 'Q1', 'A'), _row('s2', 'Q2', 'B'), _row('s3', 'Q3', 'B')]

        evaluation = evaluate_predictions(
            split_name='test',
            rows=rows,
            predicted_labels=['A', 'B', 'A'],
            classes=['A', 'B'],
        )

        self.assertEqual(labels_from_rows(rows), ['A', 'B', 'B'])
        self.assertAlmostEqual(evaluation.accuracy, 2.0 / 3.0)
        self.assertAlmostEqual(evaluation.macro_f1, 2.0 / 3.0)
        self.assertEqual(evaluation.support_by_label, {'A': 1, 'B': 2})


if __name__ == '__main__':
    unittest.main()
