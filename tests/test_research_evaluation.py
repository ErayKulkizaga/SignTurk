import unittest

import numpy as np

from research.evaluate_predictions import confusion_matrix, macro_f1, top_k_accuracy


class ResearchEvaluationTests(unittest.TestCase):
    def test_confusion_matrix_and_macro_f1(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        matrix = confusion_matrix(y_true, y_pred, class_count=3)
        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [1, 1, 0],
                    [0, 2, 0],
                    [1, 0, 1],
                ]
            ),
        )
        self.assertAlmostEqual(macro_f1(matrix), (0.5 + 0.8 + 2 / 3) / 3)

    def test_top_k_accuracy(self) -> None:
        probabilities = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.5, 0.4, 0.1],
                [0.2, 0.3, 0.5],
            ]
        )
        labels = np.array([0, 1, 1])
        self.assertAlmostEqual(top_k_accuracy(probabilities, labels, 1), 1 / 3)
        self.assertAlmostEqual(top_k_accuracy(probabilities, labels, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
