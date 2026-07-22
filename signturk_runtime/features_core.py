from __future__ import annotations

from pathlib import Path

import numpy as np

POSE_COUNT = 33
HAND_COUNT = 21
LANDMARK_DIM = 4
TOTAL_LANDMARKS = POSE_COUNT + HAND_COUNT + HAND_COUNT
MMPOSE_WHOLEBODY_COUNT = 133

# COCO-wholebody starts with the 17 COCO body keypoints. Map them into the
# MediaPipe-style body slots used by the existing stream builders.
COCO17_TO_MEDIAPIPE33 = {
    0: 0,   # nose
    1: 2,   # left eye
    2: 5,   # right eye
    3: 7,   # left ear
    4: 8,   # right ear
    5: 11,  # left shoulder
    6: 12,  # right shoulder
    7: 13,  # left elbow
    8: 14,  # right elbow
    9: 15,  # left wrist
    10: 16, # right wrist
    11: 23, # left hip
    12: 24, # right hip
    13: 25, # left knee
    14: 26, # right knee
    15: 27, # left ankle
    16: 28, # right ankle
}
MMPOSE_LEFT_HAND_START = 17 + 6 + 68
MMPOSE_RIGHT_HAND_START = MMPOSE_LEFT_HAND_START + HAND_COUNT

HAND_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

POSE_BONES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
]


def cache_frames(sample_id: str, video_path: str | Path, num_frames: int = 16, image_size: int = 224) -> Path:
    frames = read_rgb_frames(video_path, num_frames=num_frames, image_size=image_size)
    target = ensure_inside_project(FRAME_CACHE_DIR / f"{sample_id}.npz")
    safe_mkdir(target.parent)
    np.savez_compressed(target, frames=frames)
    return target


def compute_optical_flow(frames: np.ndarray) -> np.ndarray:
    import cv2

    flows = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow.astype(np.float32))
        prev = gray
    if not flows:
        flows.append(np.zeros((*frames.shape[1:3], 2), dtype=np.float32))
    return np.stack(flows)


def _as_numpy(values, dtype=np.float32) -> np.ndarray:
    if values is None:
        return np.empty((0,), dtype=dtype)
    try:
        return values.detach().cpu().numpy().astype(dtype)
    except AttributeError:
        return np.asarray(values, dtype=dtype)


def _mmpose_instances(result: dict) -> list[dict]:
    predictions = result.get("predictions", [])
    if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
        return predictions[0]
    if isinstance(predictions, list):
        return predictions
    return []


def _best_mmpose_instance(result: dict, kpt_thr: float = 0.05) -> dict | None:
    best, best_score = None, -1.0
    for instance in _mmpose_instances(result):
        keypoints = _as_numpy(instance.get("keypoints"))
        scores = _as_numpy(instance.get("keypoint_scores"))
        if keypoints.ndim != 2 or keypoints.shape[0] < 17:
            continue
        if scores.shape[0] != keypoints.shape[0]:
            scores = np.ones((keypoints.shape[0],), dtype=np.float32)
        valid = scores >= kpt_thr
        score = float(scores[valid].mean()) if valid.any() else float(scores.mean()) if scores.size else 0.0
        bbox_score = _as_numpy(instance.get("bbox_score"))
        if bbox_score.size:
            score += float(bbox_score.reshape(-1)[0])
        if score > best_score:
            best, best_score = instance, score
    return best


def mmpose_result_to_pose_hand_landmarks(result: dict, image_shape: tuple[int, int] | tuple[int, int, int], kpt_thr: float = 0.05) -> np.ndarray:
    """Convert one MMPose whole-body result into 33 pose + 21 left + 21 right landmarks.

    Output matches the existing MediaPipe-shaped feature contract:
    ``(75, 4)`` with normalized ``x, y``, zero ``z``, and score/visibility.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    landmarks = np.zeros((TOTAL_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    instance = _best_mmpose_instance(result, kpt_thr=kpt_thr)
    if instance is None:
        return landmarks

    keypoints = _as_numpy(instance.get("keypoints"))
    scores = _as_numpy(instance.get("keypoint_scores"))
    if keypoints.ndim != 2 or keypoints.shape[0] < 17:
        return landmarks
    if scores.shape[0] != keypoints.shape[0]:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)

    keypoints = keypoints.astype(np.float32)
    keypoints[:, 0] = np.clip(keypoints[:, 0] / max(width, 1), 0.0, 1.0)
    keypoints[:, 1] = np.clip(keypoints[:, 1] / max(height, 1), 0.0, 1.0)

    def copy_keypoint(src_idx: int, dst_idx: int) -> None:
        if src_idx >= keypoints.shape[0] or src_idx >= scores.shape[0] or scores[src_idx] < kpt_thr:
            return
        landmarks[dst_idx, :2] = keypoints[src_idx, :2]
        landmarks[dst_idx, 3] = float(scores[src_idx])

    for src_idx, dst_idx in COCO17_TO_MEDIAPIPE33.items():
        copy_keypoint(src_idx, dst_idx)

    for hand_start, dst_offset in [(MMPOSE_LEFT_HAND_START, POSE_COUNT), (MMPOSE_RIGHT_HAND_START, POSE_COUNT + HAND_COUNT)]:
        for i in range(HAND_COUNT):
            copy_keypoint(hand_start + i, dst_offset + i)

    return landmarks


def mmpose_results_to_pose_hand_landmarks(results: list[dict], image_shape: tuple[int, int] | tuple[int, int, int], kpt_thr: float = 0.05) -> np.ndarray:
    rows = [mmpose_result_to_pose_hand_landmarks(result, image_shape, kpt_thr=kpt_thr).reshape(-1) for result in results]
    if not rows:
        return np.zeros((0, TOTAL_LANDMARKS * LANDMARK_DIM), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def save_flow_feature(sample_id: str, frames: np.ndarray) -> Path:
    flow = compute_optical_flow(frames)
    target = ensure_inside_project(FEATURE_DIR / "flow" / f"{sample_id}.npz")
    safe_mkdir(target.parent)
    np.savez_compressed(target, flow=flow)
    return target


def _landmark_values(landmarks, count: int) -> list[float]:
    if landmarks is None:
        return [0.0] * count * 4
    values = []
    for lm in landmarks[:count]:
        values.extend([float(lm.x), float(lm.y), float(lm.z), float(getattr(lm, "visibility", 1.0))])
    while len(values) < count * 4:
        values.extend([0.0, 0.0, 0.0, 0.0])
    return values[: count * 4]


def _extract_with_legacy_holistic(frames: np.ndarray) -> np.ndarray:
    try:
        import mediapipe as mp
        holistic_module = getattr(getattr(mp, "solutions", None), "holistic", None)
        if holistic_module is None:
            from mediapipe.python.solutions import holistic as holistic_module
    except Exception:
        raise

    holistic = holistic_module.Holistic(static_image_mode=False, model_complexity=1)
    rows = []
    for frame in frames:
        result = holistic.process(frame)
        values = (
            _landmark_values(None if result.pose_landmarks is None else result.pose_landmarks.landmark, 33)
            + _landmark_values(None if result.left_hand_landmarks is None else result.left_hand_landmarks.landmark, 21)
            + _landmark_values(None if result.right_hand_landmarks is None else result.right_hand_landmarks.landmark, 21)
        )
        rows.append(values)
    holistic.close()
    return np.array(rows, dtype=np.float32)


def _extract_with_tasks(frames: np.ndarray) -> np.ndarray:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_dir = Path(__file__).resolve().parents[1] / "research" / "model_226" / "models" / "mediapipe"
    pose_model = model_dir / "pose_landmarker_lite.task"
    hand_model = model_dir / "hand_landmarker.task"
    if not pose_model.exists() or not hand_model.exists():
        raise FileNotFoundError(f"Missing MediaPipe task models in {model_dir}. Run the model download cell in notebook 04.")

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    rows = []
    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        for frame in frames:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            pose_result = pose_landmarker.detect(image)
            hand_result = hand_landmarker.detect(image)

            pose = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
            left, right = None, None
            for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name.lower() if handedness else ""
                if label == "left":
                    left = landmarks
                elif label == "right":
                    right = landmarks

            rows.append(_landmark_values(pose, 33) + _landmark_values(left, 21) + _landmark_values(right, 21))
    return np.array(rows, dtype=np.float32)


def extract_pose_hand_landmarks(frames: np.ndarray) -> np.ndarray:
    """Extract RGB-derived landmarks using legacy Holistic or current MediaPipe Tasks."""
    for extractor in (_extract_with_legacy_holistic, _extract_with_tasks):
        try:
            result = extractor(frames)
            if np.any(result[:, :, ...] != 0):
                return result
            return result
        except Exception:
            continue
    return np.zeros((len(frames), 33 * 4 + 21 * 4 * 2), dtype=np.float32)


def landmarks_to_array(landmarks: np.ndarray) -> np.ndarray:
    arr = np.asarray(landmarks, dtype=np.float32)
    return arr.reshape(arr.shape[0], TOTAL_LANDMARKS, LANDMARK_DIM)


def body_center_normalize(landmarks: np.ndarray) -> tuple[np.ndarray, dict]:
    arr = landmarks_to_array(landmarks)
    xyz = arr[:, :, :3].copy()
    visibility = arr[:, :, 3:4].copy()
    left_shoulder = xyz[:, 11:12, :]
    right_shoulder = xyz[:, 12:13, :]
    center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
    scale = np.clip(shoulder_width, 1e-3, None)
    normalized_xyz = (xyz - center) / scale
    normalized = np.concatenate([normalized_xyz, visibility], axis=-1)
    stats = {
        "origin": "torso_shoulders_center",
        "scale": "shoulder_width",
        "landmarks": int(TOTAL_LANDMARKS),
        "feature_dim_per_landmark": int(LANDMARK_DIM),
    }
    return normalized.astype(np.float32), stats


def build_bone_stream(normalized_landmarks: np.ndarray) -> np.ndarray:
    bones = []
    left_offset = POSE_COUNT
    right_offset = POSE_COUNT + HAND_COUNT
    for a, b in POSE_BONES:
        bones.append(normalized_landmarks[:, b, :3] - normalized_landmarks[:, a, :3])
    for offset in [left_offset, right_offset]:
        for a, b in HAND_BONES:
            bones.append(normalized_landmarks[:, offset + b, :3] - normalized_landmarks[:, offset + a, :3])
    return np.concatenate(bones, axis=-1).astype(np.float32)


def motion_stream(stream: np.ndarray) -> np.ndarray:
    motion = np.zeros_like(stream)
    motion[1:] = stream[1:] - stream[:-1]
    return motion.astype(np.float32)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    valid = denom > 1e-6
    cos = np.zeros_like(denom, dtype=np.float32)
    cos[valid] = np.sum(ba[valid] * bc[valid], axis=-1) / denom[valid]
    angle = np.zeros_like(denom, dtype=np.float32)
    angle[valid] = np.arccos(np.clip(cos[valid], -1.0, 1.0))
    return angle[..., None]


def build_extra_geometry_stream(normalized_landmarks: np.ndarray) -> np.ndarray:
    xyz = normalized_landmarks[:, :, :3]
    visibility = normalized_landmarks[:, :, 3]
    nose = xyz[:, 0]
    left_shoulder = xyz[:, 11]
    right_shoulder = xyz[:, 12]
    torso_center = (left_shoulder + right_shoulder) / 2.0
    left_wrist = xyz[:, 15]
    right_wrist = xyz[:, 16]
    left_hand_wrist = xyz[:, POSE_COUNT + 0]
    right_hand_wrist = xyz[:, POSE_COUNT + HAND_COUNT + 0]
    left_velocity = np.linalg.norm(motion_stream(left_hand_wrist), axis=-1, keepdims=True)
    right_velocity = np.linalg.norm(motion_stream(right_hand_wrist), axis=-1, keepdims=True)
    hands_distance = np.linalg.norm(left_hand_wrist - right_hand_wrist, axis=-1, keepdims=True)
    hands_distance_change = motion_stream(hands_distance)
    pose_mask = (visibility[:, :POSE_COUNT].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    left_hand_mask = (visibility[:, POSE_COUNT : POSE_COUNT + HAND_COUNT].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    right_hand_mask = (visibility[:, POSE_COUNT + HAND_COUNT :].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    both_hands_mask = left_hand_mask * right_hand_mask

    extras = [
        left_wrist - right_wrist,
        left_wrist - nose,
        right_wrist - nose,
        left_wrist - left_shoulder,
        right_wrist - right_shoulder,
        left_wrist - torso_center,
        right_wrist - torso_center,
        left_velocity,
        right_velocity,
        hands_distance,
        hands_distance_change,
        pose_mask,
        left_hand_mask,
        right_hand_mask,
        both_hands_mask,
    ]

    angle_values = []
    for offset in [POSE_COUNT, POSE_COUNT + HAND_COUNT]:
        hand = xyz[:, offset : offset + HAND_COUNT]
        for a, b, c in [(0, 2, 4), (0, 6, 8), (0, 10, 12), (0, 14, 16), (0, 18, 20)]:
            angle_values.append(_angle(hand[:, a], hand[:, b], hand[:, c]))
    extras.extend(angle_values)
    return np.concatenate(extras, axis=-1).astype(np.float32)


def build_skeleton_streams(landmarks: np.ndarray) -> dict[str, np.ndarray | dict]:
    normalized, stats = body_center_normalize(landmarks)
    joint = normalized.reshape(normalized.shape[0], -1).astype(np.float32)
    bone = build_bone_stream(normalized)
    return {
        "joint": joint,
        "bone": bone,
        "joint_motion": motion_stream(joint),
        "bone_motion": motion_stream(bone),
        "extra": build_extra_geometry_stream(normalized),
        "norm_stats": stats,
    }


def save_pose_hand_feature(sample_id: str, frames: np.ndarray) -> Path:
    landmarks = extract_pose_hand_landmarks(frames)
    target = ensure_inside_project(FEATURE_DIR / "pose_hand" / f"{sample_id}.npz")
    safe_mkdir(target.parent)
    np.savez_compressed(target, landmarks=landmarks)
    return target


def save_skeleton_stream_feature(sample_id: str, frames: np.ndarray) -> Path:
    landmarks = extract_pose_hand_landmarks(frames)
    streams = build_skeleton_streams(landmarks)
    target = ensure_inside_project(FEATURE_DIR / "skeleton_streams" / f"{sample_id}.npz")
    safe_mkdir(target.parent)
    np.savez_compressed(
        target,
        joint=streams["joint"],
        bone=streams["bone"],
        joint_motion=streams["joint_motion"],
        bone_motion=streams["bone_motion"],
        extra=streams["extra"],
        landmarks=landmarks,
    )
    return target


def save_feature_metadata(metadata: dict, name: str) -> Path:
    return save_json(metadata, FEATURE_DIR / f"{name}_metadata.json")
