from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feature_builder import HAND_NODES, LANDMARK_DIM, LEFT_HAND_NODES, RIGHT_HAND_NODES, TOTAL_LANDMARKS


@dataclass
class PoseQuality:
    ok: bool
    feedback: str
    score: float
    debug: dict[str, float | bool | str]


def _mean_conf(arr: np.ndarray, nodes: list[int]) -> float:
    values = arr[nodes, 3]
    return float(values.mean()) if len(values) else 0.0


def evaluate_pose_quality(
    landmarks: np.ndarray,
    min_shoulder_width_ratio: float = 0.16,
    max_shoulder_width_ratio: float = 0.55,
    center_x_range: tuple[float, float] = (0.28, 0.72),
    upper_body_y_range: tuple[float, float] = (0.15, 0.68),
    min_pose_confidence: float = 0.25,
    min_any_hand_visible_rate: float = 0.25,
) -> PoseQuality:
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0:
        return PoseQuality(False, "Kameraya gecin", 0.0, {"reason": "empty_landmarks"})
    arr = arr.reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)
    frame = arr[-1]

    nose_conf = float(frame[0, 3])
    left_shoulder_conf = float(frame[11, 3])
    right_shoulder_conf = float(frame[12, 3])
    left_wrist_conf = float(frame[15, 3])
    right_wrist_conf = float(frame[16, 3])
    left_hand_visible_rate = _mean_conf(frame, LEFT_HAND_NODES)
    right_hand_visible_rate = _mean_conf(frame, RIGHT_HAND_NODES)
    any_hand_visible_rate = max(left_hand_visible_rate, right_hand_visible_rate)
    wrist_visible_rate = max(left_wrist_conf, right_wrist_conf)
    hand_signal_rate = max(any_hand_visible_rate, wrist_visible_rate)

    person_visible_count = sum(
        value >= min_pose_confidence
        for value in [nose_conf, left_shoulder_conf, right_shoulder_conf, left_wrist_conf, right_wrist_conf]
    )
    person_visible = person_visible_count >= 2
    shoulders_visible = left_shoulder_conf >= min_pose_confidence and right_shoulder_conf >= min_pose_confidence
    face_or_nose_visible = nose_conf >= min_pose_confidence

    if not person_visible:
        return PoseQuality(False, "Kameraya gecin", 0.0, {"person_visible": False})
    if not shoulders_visible:
        return PoseQuality(False, "Ust govdeniz gorunmeli", 0.25, {"shoulders_visible": False})

    left_shoulder = frame[11, :2]
    right_shoulder = frame[12, :2]
    shoulder_width_ratio = float(abs(left_shoulder[0] - right_shoulder[0]))
    torso_center_x = float((left_shoulder[0] + right_shoulder[0]) / 2.0)
    upper_body_y = float(frame[0, 1] if face_or_nose_visible else (left_shoulder[1] + right_shoulder[1]) / 2.0)

    hand_points = frame[HAND_NODES]
    wrist_points = frame[[15, 16]]
    visible_hand_points = hand_points[hand_points[:, 3] > 0]
    visible_wrists = wrist_points[wrist_points[:, 3] >= min_pose_confidence]
    visible_hand_or_wrist_points = visible_hand_points
    if len(visible_wrists):
        visible_hand_or_wrist_points = np.concatenate([visible_hand_or_wrist_points, visible_wrists], axis=0)
    hand_in_frame_rate = 0.0
    if len(visible_hand_or_wrist_points):
        xy = visible_hand_or_wrist_points[:, :2]
        in_frame = (xy[:, 0] >= 0.02) & (xy[:, 0] <= 0.98) & (xy[:, 1] >= 0.02) & (xy[:, 1] <= 0.98)
        hand_in_frame_rate = float(in_frame.mean())

    debug = {
        "person_visible": bool(person_visible),
        "face_or_nose_visible": bool(face_or_nose_visible),
        "left_shoulder_visible": bool(left_shoulder_conf >= min_pose_confidence),
        "right_shoulder_visible": bool(right_shoulder_conf >= min_pose_confidence),
        "left_hand_visible_rate": left_hand_visible_rate,
        "right_hand_visible_rate": right_hand_visible_rate,
        "left_wrist_confidence": left_wrist_conf,
        "right_wrist_confidence": right_wrist_conf,
        "hand_visible_rate": hand_signal_rate,
        "raw_hand_landmark_visible_rate": any_hand_visible_rate,
        "shoulder_width_ratio": shoulder_width_ratio,
        "torso_center_x": torso_center_x,
        "upper_body_center_y": upper_body_y,
        "hand_in_frame_rate": hand_in_frame_rate,
    }

    if shoulder_width_ratio < min_shoulder_width_ratio:
        return PoseQuality(False, "Biraz yaklasin", 0.45, debug)
    if shoulder_width_ratio > max_shoulder_width_ratio:
        return PoseQuality(False, "Biraz uzaklasin", 0.45, debug)
    if torso_center_x < center_x_range[0] or torso_center_x > center_x_range[1]:
        return PoseQuality(False, "Ortaya gecin", 0.55, debug)
    if upper_body_y < upper_body_y_range[0] or upper_body_y > upper_body_y_range[1]:
        return PoseQuality(False, "Ust govdeniz gorunmeli", 0.55, debug)
    if hand_signal_rate < min_any_hand_visible_rate or hand_in_frame_rate < 0.15:
        return PoseQuality(False, "Ellerinizi kadraja alin", 0.65, debug)

    score_parts = [
        min(1.0, shoulder_width_ratio / max(min_shoulder_width_ratio, 1e-6)),
        1.0 - min(abs(torso_center_x - 0.5) / 0.5, 1.0),
        min(hand_signal_rate / max(min_any_hand_visible_rate, 1e-6), 1.0),
        hand_in_frame_rate,
    ]
    score = float(np.clip(np.mean(score_parts), 0.0, 1.0))
    debug["pose_quality_score"] = score
    return PoseQuality(True, "Hazir, isareti yapabilirsiniz", score, debug)
