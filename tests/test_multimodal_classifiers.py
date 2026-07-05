from __future__ import annotations

import unittest

import numpy as np

from multi_bird_db.multimodal.classifiers import (
    SoftmaxClassifierConfig,
    fit_softmax_classifier,
    predict_labels,
    predict_probabilities,
)


class MultimodalClassifierTests(unittest.TestCase):
    def test_softmax_classifier_fits_linearly_separable_toy_problem(self) -> None:
        features = np.asarray(
            [
                [-2.0, -1.0],
                [-1.5, -1.2],
                [1.0, 1.0],
                [1.5, 1.2],
            ],
            dtype=np.float32,
        )
        labels = ['left', 'left', 'right', 'right']

        model = fit_softmax_classifier(
            features,
            labels,
            config=SoftmaxClassifierConfig(learning_rate=0.1, num_epochs=250, l2_weight=0.0, seed=7),
        )

        predicted = predict_labels(model, features)
        probabilities = predict_probabilities(model, features)

        self.assertEqual(predicted, labels)
        self.assertEqual(tuple(probabilities.shape), (4, 2))
        self.assertEqual(model.classes, ['left', 'right'])


if __name__ == '__main__':
    unittest.main()
