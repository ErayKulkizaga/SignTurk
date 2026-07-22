from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features_core import build_skeleton_streams


POSE_COUNT = 33
HAND_COUNT = 21
LANDMARK_DIM = 4
TOTAL_LANDMARKS = 75
COCO17_TO_MEDIAPIPE33 = {
    0: 0,
    1: 2,
    2: 5,
    3: 7,
    4: 8,
    5: 11,
    6: 12,
    7: 13,
    8: 14,
    9: 15,
    10: 16,
    11: 23,
    12: 24,
    13: 25,
    14: 26,
    15: 27,
    16: 28,
}
RTMLIB_LEFT_HAND_START = 17 + 6 + 68
RTMLIB_RIGHT_HAND_START = RTMLIB_LEFT_HAND_START + HAND_COUNT
LEFT_HAND_NODES = list(range(33, 54))
RIGHT_HAND_NODES = list(range(54, 75))
HAND_NODES = LEFT_HAND_NODES + RIGHT_HAND_NODES


@dataclass
class FeatureBundle:
    skeleton_inputs: list[np.ndarray]
    hand_inputs: list[np.ndarray]
    streams: dict[str, np.ndarray]
    landmarks: np.ndarray


def sample_indices(total_frames: int, target_frames: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(target_frames, dtype=np.int64)
    if total_frames >= target_frames:
        return np.linspace(0, total_frames - 1, target_frames).round().astype(np.int64)
    indices = list(range(total_frames))
    while len(indices) < target_frames:
        indices.append(total_frames - 1)
    return np.asarray(indices[:target_frames], dtype=np.int64)


def sample_frames(frames: list[np.ndarray], target_frames: int = 32) -> list[np.ndarray]:
    indices = sample_indices(len(frames), target_frames)
    return [frames[int(idx)] for idx in indices]


def rtmlib_to_landmarks(
    keypoints,
    scores,
    image_shape: tuple[int, int] | tuple[int, int, int],
    kpt_thr: float = 0.05,
    filter_hands: bool = False,
    min_hand_points: int = 6,
    min_hand_mean_conf: float = 0.20,
    max_hand_bbox_span: float = 0.34,
    max_hand_bbox_area: float = 0.06,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    landmarks = np.zeros((TOTAL_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if keypoints.ndim == 3:
        if keypoints.shape[0] == 0:
            return landmarks
        person_scores = scores.mean(axis=1) if scores.ndim == 2 else np.ones((keypoints.shape[0],), dtype=np.float32)
        person_idx = int(np.argmax(person_scores))
        keypoints = keypoints[person_idx]
        scores = scores[person_idx] if scores.ndim == 2 else np.ones((keypoints.shape[0],), dtype=np.float32)

    if keypoints.ndim != 2 or keypoints.shape[0] < 17:
        return landmarks
    if scores.ndim != 1 or scores.shape[0] != keypoints.shape[0]:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)

    keypoints = keypoints.copy()
    keypoints[:, 0] = np.clip(keypoints[:, 0] / max(width, 1), 0.0, 1.0)
    keypoints[:, 1] = np.clip(keypoints[:, 1] / max(height, 1), 0.0, 1.0)

    def copy_keypoint(src_idx: int, dst_idx: int) -> None:
        if src_idx >= keypoints.shape[0] or src_idx >= scores.shape[0] or scores[src_idx] < kpt_thr:
            return
        landmarks[dst_idx, :2] = keypoints[src_idx, :2]
        landmarks[dst_idx, 3] = float(scores[src_idx])

    for src_idx, dst_idx in COCO17_TO_MEDIAPIPE33.items():
        copy_keypoint(src_idx, dst_idx)
    for i in range(HAND_COUNT):
        copy_keypoint(RTMLIB_LEFT_HAND_START + i, POSE_COUNT + i)
        copy_keypoint(RTMLIB_RIGHT_HAND_START + i, POSE_COUNT + HAND_COUNT + i)

    if not filter_hands:
        return landmarks

    def clear_weak_hand(nodes: list[int]) -> None:
        hand = landmarks[nodes]
        present = hand[:, 3] >= kpt_thr
        if int(present.sum()) < min_hand_points:
            landmarks[nodes] = 0.0
            return
        if float(hand[present, 3].mean()) < min_hand_mean_conf:
            landmarks[nodes] = 0.0
            return
        xy = hand[present, :2]
        span = xy.max(axis=0) - xy.min(axis=0)
        if float(span[0]) > max_hand_bbox_span or float(span[1]) > max_hand_bbox_span:
            landmarks[nodes] = 0.0
            return
        if float(span[0] * span[1]) > max_hand_bbox_area:
            landmarks[nodes] = 0.0

    clear_weak_hand(LEFT_HAND_NODES)
    clear_weak_hand(RIGHT_HAND_NODES)
    return landmarks


def take_nodes(flat_stream: np.ndarray, nodes: list[int]) -> np.ndarray:
    # flat_stream shape: (T, 75 * 4)
    arr = flat_stream.reshape(flat_stream.shape[0], TOTAL_LANDMARKS, LANDMARK_DIM)
    return arr[:, nodes, :].reshape(flat_stream.shape[0], len(nodes) * LANDMARK_DIM).astype(np.float32)


def build_feature_bundle(landmarks: np.ndarray) -> FeatureBundle:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.ndim == 3:
        flat_landmarks = landmarks.reshape(landmarks.shape[0], -1)
    else:
        flat_landmarks = landmarks
        landmarks = flat_landmarks.reshape(flat_landmarks.shape[0], TOTAL_LANDMARKS, LANDMARK_DIM)

    streams = build_skeleton_streams(flat_landmarks)
    skeleton_inputs = [
        streams["joint"][None, ...].astype(np.float32),
        streams["bone"][None, ...].astype(np.float32),
        streams["joint_motion"][None, ...].astype(np.float32),
        streams["bone_motion"][None, ...].astype(np.float32),
        streams["extra"][None, ...].astype(np.float32),
    ]
    hand_inputs = [
        take_nodes(streams["joint"], HAND_NODES)[None, ...].astype(np.float32),
        take_nodes(streams["joint_motion"], HAND_NODES)[None, ...].astype(np.float32),
        streams["extra"][None, ...].astype(np.float32),
    ]
    return FeatureBundle(skeleton_inputs=skeleton_inputs, hand_inputs=hand_inputs, streams=streams, landmarks=landmarks)


def landmarks_motion_score(landmarks: np.ndarray) -> float:
    arr = np.asarray(landmarks, dtype=np.float32).reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)
    hand = arr[:, HAND_NODES, :2]
    conf = arr[:, HAND_NODES, 3] > 0
    if hand.shape[0] < 2 or not conf.any():
        return 0.0
    diff = np.linalg.norm(hand[1:] - hand[:-1], axis=-1)
    valid = conf[1:] & conf[:-1]
    return float(diff[valid].mean()) if valid.any() else 0.0
