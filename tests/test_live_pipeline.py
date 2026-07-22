import unittest

import numpy as np

from live_pipeline import build_legacy_feature_vector, count_detected_hands, preprocess_sequence


class LivePipelineTests(unittest.TestCase):
    def test_detected_hand_count(self) -> None:
        landmarks = np.zeros(126, dtype=np.float32)
        self.assertEqual(count_detected_hands(landmarks, minimum_norm=0.1), 0)
        landmarks[0] = 1.0
        self.assertEqual(count_detected_hands(landmarks, minimum_norm=0.1), 1)
        landmarks[63] = 1.0
        self.assertEqual(count_detected_hands(landmarks, minimum_norm=0.1), 2)

    def test_live_preprocessing_contract(self) -> None:
        rng = np.random.default_rng(42)
        sequence = rng.normal(size=(16, 126)).astype(np.float32)
        mean = np.zeros(156, dtype=np.float32)
        std = np.ones(156, dtype=np.float32)
        result = preprocess_sequence(sequence, mean, std)
        self.assertEqual(result.shape, (1, 16, 156))
        self.assertTrue(np.isfinite(result).all())

    def test_legacy_color_depth_contract(self) -> None:
        color = np.ones((16, 20), dtype=np.float32)
        depth = np.full((16, 20), 2.0, dtype=np.float32)
        mean = np.zeros(40, dtype=np.float32)
        std = np.ones(40, dtype=np.float32)
        result = build_legacy_feature_vector(color, depth, mean, std, 20)
        self.assertEqual(result.shape, (1, 16, 40))
        np.testing.assert_array_equal(result[0, :, :20], color)
        np.testing.assert_array_equal(result[0, :, 20:], depth)


if __name__ == "__main__":
    unittest.main()
