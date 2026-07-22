from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .config import BackendSettings
from .feature_builder import LEFT_HAND_NODES, RIGHT_HAND_NODES, landmarks_motion_score, sample_frames, sample_indices
from .pose_quality import PoseQuality, evaluate_pose_quality


@dataclass
class RealtimeState:
    settings: BackendSettings
    frames: deque[np.ndarray] = field(default_factory=deque)
    landmarks: deque[np.ndarray] = field(default_factory=deque)
    prediction_history: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    state: str = "CAMERA_START"
    calibrated: bool = False
    calibration_started_at: float | None = None
    recording_started_at: float | None = None
    last_prediction_at: float = 0.0
    last_label: str | None = None
    last_class_id: int | None = None
    last_confidence: float = 0.0
    last_top3: list[dict] = field(default_factory=list)
    last_stable: bool = False
    last_good_left_hand: np.ndarray | None = None
    last_good_right_hand: np.ndarray | None = None
    last_good_left_hand_at: float = 0.0
    last_good_right_hand_at: float = 0.0

    def add_frame(self, frame: np.ndarray) -> None:
        max_frames = max(8, int(self.settings.max_segment_seconds * self.settings.frontend_send_fps) + 8)
        self.frames.append(frame)
        while len(self.frames) > max_frames:
            self.frames.popleft()

    def add_landmarks(self, landmarks: np.ndarray) -> None:
        max_frames = max(8, int(self.settings.max_segment_seconds * self.settings.frontend_send_fps) + 8)
        self.landmarks.append(np.asarray(landmarks, dtype=np.float32))
        while len(self.landmarks) > max_frames:
            self.landmarks.popleft()

    def stabilize_landmarks(self, landmarks: np.ndarray) -> tuple[np.ndarray, dict]:
        arr = np.asarray(landmarks, dtype=np.float32).reshape(75, 4).copy()
        now = time.time()
        left_conf = float(arr[LEFT_HAND_NODES, 3].mean())
        right_conf = float(arr[RIGHT_HAND_NODES, 3].mean())
        left_held = False
        right_held = False

        if left_conf >= self.settings.min_any_hand_visible_rate:
            self.last_good_left_hand = arr[LEFT_HAND_NODES].copy()
            self.last_good_left_hand_at = now
        elif self.last_good_left_hand is not None and now - self.last_good_left_hand_at <= self.settings.hand_hold_seconds:
            arr[LEFT_HAND_NODES] = self.last_good_left_hand
            left_held = True

        if right_conf >= self.settings.min_any_hand_visible_rate:
            self.last_good_right_hand = arr[RIGHT_HAND_NODES].copy()
            self.last_good_right_hand_at = now
        elif self.last_good_right_hand is not None and now - self.last_good_right_hand_at <= self.settings.hand_hold_seconds:
            arr[RIGHT_HAND_NODES] = self.last_good_right_hand
            right_held = True

        return arr, {
            "left_hand_held": left_held,
            "right_hand_held": right_held,
            "raw_left_hand_confidence": left_conf,
            "raw_right_hand_confidence": right_conf,
        }

    def enough_frames(self) -> bool:
        return len(self.frames) >= self.min_buffer_frames()

    def enough_landmarks(self) -> bool:
        return len(self.landmarks) >= self.min_buffer_frames()

    def min_buffer_frames(self) -> int:
        return max(4, int(self.settings.min_segment_seconds * self.settings.frontend_send_fps))

    def segment_frames(self) -> list[np.ndarray]:
        return sample_frames(list(self.frames), self.settings.target_frames)

    def segment_landmarks(self) -> np.ndarray:
        items = list(self.landmarks)
        indices = sample_indices(len(items), self.settings.target_frames)
        return np.stack([items[int(idx)] for idx in indices]).astype(np.float32)

    def cooldown_active(self) -> bool:
        return (time.time() - self.last_prediction_at) * 1000.0 < self.settings.prediction_cooldown_ms

    def update_calibration_state(self, pose: PoseQuality) -> dict:
        if self.calibrated:
            return {
                "calibrated": True,
                "just_calibrated": False,
                "calibration_progress": 1.0,
                "state": self.state,
                "pose_feedback": "Hazir, isareti yapabilirsiniz",
            }

        if not pose.ok:
            self.state = "CALIBRATION"
            self.calibration_started_at = None
            self.recording_started_at = None
            return {
                "calibrated": False,
                "just_calibrated": False,
                "calibration_progress": 0.0,
                "state": self.state,
                "pose_feedback": pose.feedback,
            }

        if self.calibration_started_at is None:
            self.calibration_started_at = time.time()

        elapsed = time.time() - self.calibration_started_at
        progress = min(1.0, elapsed / max(self.settings.calibration_seconds, 0.1))
        if progress < 1.0:
            self.state = "CALIBRATION"
            return {
                "calibrated": False,
                "just_calibrated": False,
                "calibration_progress": progress,
                "state": self.state,
                "pose_feedback": "Sabit kalin",
            }

        self.calibrated = True
        self.frames.clear()
        self.landmarks.clear()
        self.recording_started_at = None
        self.state = "READY"
        return {
            "calibrated": True,
            "just_calibrated": True,
            "calibration_progress": 1.0,
            "state": self.state,
            "pose_feedback": "Hazir, isareti yapabilirsiniz",
        }

    def update_pre_prediction_state(self, pose: PoseQuality, landmarks: np.ndarray | None = None) -> dict:
        if not pose.ok:
            self.state = "POSE_CHECK"
            self.recording_started_at = None
            return {"should_predict": False, "state": self.state}

        if self.cooldown_active() and self.last_top3:
            self.state = "SHOW_RESULT"
            self.recording_started_at = None
            remaining_ms = max(0.0, self.settings.prediction_cooldown_ms - (time.time() - self.last_prediction_at) * 1000.0)
            return {
                "should_predict": False,
                "state": self.state,
                "hold_result": True,
                "stable": self.last_stable,
                "label": self.last_label,
                "class_id": self.last_class_id,
                "confidence": self.last_confidence,
                "top3": self.last_top3,
                "cooldown_remaining_ms": remaining_ms,
            }

        if not self.enough_landmarks():
            self.state = "READY"
            return {"should_predict": False, "state": self.state}

        motion_score = landmarks_motion_score(landmarks) if landmarks is not None else 0.0
        if motion_score < self.settings.motion_threshold:
            self.state = "READY"
            return {"should_predict": False, "state": self.state, "motion_score": motion_score}

        if self.recording_started_at is None:
            self.recording_started_at = time.time()
            self.state = "RECORDING"
            return {"should_predict": False, "state": self.state, "motion_score": motion_score}

        elapsed = time.time() - self.recording_started_at
        if elapsed < self.settings.default_capture_seconds:
            self.state = "RECORDING"
            return {"should_predict": False, "state": self.state, "motion_score": motion_score, "recording_seconds": elapsed}

        self.state = "PREDICTING"
        return {"should_predict": True, "state": self.state, "motion_score": motion_score, "recording_seconds": elapsed}

    def finalize_prediction(self, top3: list[dict], top1_threshold: float, margin_threshold: float) -> dict:
        top1 = top3[0] if top3 else {"label": "", "confidence": 0.0, "class_id": -1}
        top2_conf = float(top3[1]["confidence"]) if len(top3) > 1 else 0.0
        margin = float(top1["confidence"]) - top2_conf
        stable = float(top1["confidence"]) >= top1_threshold and margin >= margin_threshold
        label = str(top1["label"])
        if stable:
            self.prediction_history.append(label)
            counts = Counter(self.prediction_history)
            stable = counts[label] >= max(1, int(self.settings.required_stable_votes))

        self.last_prediction_at = time.time()
        self.last_stable = bool(stable)
        self.last_label = label if stable else None
        self.last_class_id = int(top1["class_id"]) if stable else None
        self.last_confidence = float(top1["confidence"])
        self.last_top3 = top3
        self.recording_started_at = None
        self.frames.clear()
        self.landmarks.clear()
        self.state = "SHOW_RESULT"
        return {
            "stable": bool(stable),
            "label": label if stable else None,
            "class_id": int(top1["class_id"]) if stable else None,
            "confidence": float(top1["confidence"]),
            "top1_margin": margin,
        }
