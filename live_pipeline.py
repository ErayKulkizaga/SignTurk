"""Pure preprocessing helpers for the lightweight live recognition models."""

from __future__ import annotations

import numpy as np


def hand_is_present(hand_vector: np.ndarray, minimum_norm: float) -> bool:
    return bool(np.linalg.norm(hand_vector) > minimum_norm)


def count_detected_hands(landmarks: np.ndarray | None, minimum_norm: float) -> int:
    if landmarks is None or landmarks.shape != (126,):
        return 0
    return int(hand_is_present(landmarks[:63], minimum_norm)) + int(
        hand_is_present(landmarks[63:], minimum_norm)
    )


def normalize_hands_relative(sequence: np.ndarray) -> np.ndarray:
    result = sequence.copy()
    frame_count = result.shape[0]
    for start in (0, 63):
        hand = result[:, start : start + 63].reshape(frame_count, 21, 3)
        hand_relative = hand - hand[:, 0:1, :]
        scale = np.linalg.norm(hand_relative[:, 9, :], axis=-1, keepdims=True)[
            :, :, np.newaxis
        ]
        scale = np.where(scale < 1e-6, 1.0, scale)
        result[:, start : start + 63] = (hand_relative / scale).reshape(frame_count, 63)
    return result


def compute_finger_angles(sequence: np.ndarray) -> np.ndarray:
    frame_count = sequence.shape[0]
    chains = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    angle_sets = []
    for hand_start in (0, 63):
        hand = sequence[:, hand_start : hand_start + 63].reshape(frame_count, 21, 3)
        hand_angles = np.zeros((frame_count, 15), dtype=np.float32)
        angle_index = 0
        for chain in chains:
            for index in range(len(chain) - 1):
                first = hand[:, 0 if index == 0 else chain[index - 1], :]
                centre = hand[:, chain[index], :]
                last = hand[:, chain[index + 1], :]
                first_vector, last_vector = first - centre, last - centre
                cosine = np.sum(first_vector * last_vector, axis=-1) / (
                    np.linalg.norm(first_vector, axis=-1)
                    * np.linalg.norm(last_vector, axis=-1)
                    + 1e-8
                )
                hand_angles[:, angle_index] = np.arccos(np.clip(cosine, -1, 1))
                angle_index += 1
        angle_sets.append(hand_angles)
    return np.concatenate([sequence, np.concatenate(angle_sets, axis=-1)], axis=-1)


def preprocess_sequence(
    raw_landmarks: np.ndarray,
    normalization_mean: np.ndarray | None,
    normalization_std: np.ndarray | None,
) -> np.ndarray:
    features = compute_finger_angles(normalize_hands_relative(raw_landmarks))
    if normalization_mean is not None and normalization_std is not None:
        features = (features - normalization_mean) / normalization_std
    return features[np.newaxis, :, :].astype(np.float32)


def build_legacy_feature_vector(
    color_sequence: np.ndarray,
    depth_sequence: np.ndarray,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    single_stream_dimension: int,
) -> np.ndarray:
    color = (color_sequence - normalization_mean[:single_stream_dimension]) / normalization_std[
        :single_stream_dimension
    ]
    depth = (depth_sequence - normalization_mean[single_stream_dimension:]) / normalization_std[
        single_stream_dimension:
    ]
    return np.concatenate([color, depth], axis=-1)[np.newaxis, :, :]
