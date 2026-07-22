from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .feature_builder import rtmlib_to_landmarks

HF_VITPOSE_DET_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
HF_VITPOSE_POSE_URL = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-s-wholebody.onnx"
HF_RTMW_DET_URL = "https://huggingface.co/memescreamer/yolox-onnx/resolve/main/yolox_m.onnx"
HF_RTMW_POSE_URL = "https://huggingface.co/Izymka/rtmw-dw-x-l_simcc-cocktail14/resolve/main/rtmw-dw-x-l_simcc-cocktail14.onnx"

_ORT_DLLS_PRELOADED = False


def _preload_onnxruntime_cuda_dlls(ort) -> str | None:
    """Load CUDA/cuDNN/MSVC DLLs bundled with ONNX Runtime before sessions open."""
    global _ORT_DLLS_PRELOADED
    if _ORT_DLLS_PRELOADED:
        return None
    _ORT_DLLS_PRELOADED = True

    if os.getenv("SIGNTURK_PRELOAD_ORT_DLLS", "true").lower() not in {"1", "true", "yes"}:
        return None

    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return None

    try:
        preload(directory="")
        return None
    except Exception as exc:
        return f"ONNX Runtime CUDA DLL preload failed: {exc}"


@dataclass
class ExtractorStatus:
    available: bool
    message: str
    providers: list[str]
    device: str | None


class RTMPoseExtractor:
    def __init__(
        self,
        kpt_thr: float = 0.05,
        preferred_extractor: str = "rtmpose",
        try_rtmpose_first: bool = True,
        allow_mediapipe_fallback: bool = False,
    ):
        self.kpt_thr = float(os.getenv("SIGNTURK_KPT_THR", str(kpt_thr)))
        self.filter_model_hands = os.getenv("SIGNTURK_FILTER_MODEL_HANDS", "false").lower() in {"1", "true", "yes"}
        self.min_hand_points = int(os.getenv("SIGNTURK_MIN_HAND_POINTS", "6"))
        self.min_hand_mean_conf = float(os.getenv("SIGNTURK_MIN_HAND_MEAN_CONF", "0.20"))
        self.max_hand_bbox_span = float(os.getenv("SIGNTURK_MAX_HAND_BBOX_SPAN", "0.34"))
        self.max_hand_bbox_area = float(os.getenv("SIGNTURK_MAX_HAND_BBOX_AREA", "0.06"))
        self._model = None
        self._fallback = None
        self.backend_name = "rtmpose"
        self.preferred_extractor = os.getenv("SIGNTURK_EXTRACTOR", preferred_extractor).strip().lower()
        self.try_rtmpose_first = os.getenv("SIGNTURK_TRY_RTMPOSE", str(try_rtmpose_first)).lower() in {"1", "true", "yes"}
        self.try_hf_fallback = os.getenv("SIGNTURK_TRY_HF_VITPOSE_FALLBACK", "true").lower() in {"1", "true", "yes"}
        self.allow_mediapipe_fallback = os.getenv("SIGNTURK_ALLOW_MEDIAPIPE_FALLBACK", str(allow_mediapipe_fallback)).lower() in {"1", "true", "yes"}
        self.status = self._init_model()

    def _init_model(self) -> ExtractorStatus:
        if self.preferred_extractor in {"hf_rtmw", "rtmw_hf", "huggingface_rtmw"}:
            return self._init_hf_rtmw()

        if self.preferred_extractor in {"hf_vitpose", "vitpose", "huggingface", "github_hf"}:
            return self._init_hf_vitpose()

        if self.preferred_extractor in {"mediapipe", "mediapipe_tasks", "fallback"} and not self.try_rtmpose_first:
            fallback = self._init_mediapipe_fallback()
            if fallback.available:
                return fallback

        try:
            import onnxruntime as ort
            from rtmlib import Wholebody
        except Exception as exc:
            return ExtractorStatus(False, f"RTMPose dependencies missing: {exc}", [], None)

        try:
            preload_warning = _preload_onnxruntime_cuda_dlls(ort)
            providers = list(ort.get_available_providers())
            device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
            det_path = os.getenv("SIGNTURK_RTMPOSE_DET") or None
            pose_path = os.getenv("SIGNTURK_RTMPOSE_POSE") or None
            self._model = Wholebody(
                det=det_path,
                pose=pose_path,
                to_openpose=False,
                mode="balanced",
                backend="onnxruntime",
                device=device,
            )
            message = "RTMPose ready" if not preload_warning else f"RTMPose ready; {preload_warning}"
            return ExtractorStatus(True, message, providers, device)
        except Exception as exc:
            if self.try_hf_fallback:
                hf_fallback = self._init_hf_vitpose()
                if hf_fallback.available:
                    hf_fallback.message = f"RTMPose init failed, using HF ViTPose wholebody fallback: {exc}"
                    return hf_fallback
            if not self.allow_mediapipe_fallback:
                return ExtractorStatus(False, f"RTMPose init failed: {exc}", providers if "providers" in locals() else [], None)
            fallback = self._init_mediapipe_fallback()
            if fallback.available:
                fallback.providers = providers if "providers" in locals() else []
                fallback.message = f"RTMPose init failed, using MediaPipe fallback: {exc}"
                return fallback
            return ExtractorStatus(False, f"RTMPose init failed: {exc}; MediaPipe fallback failed: {fallback.message}", providers if "providers" in locals() else [], None)

    def _init_hf_rtmw(self) -> ExtractorStatus:
        try:
            import onnxruntime as ort
            from rtmlib import Wholebody
        except Exception as exc:
            return ExtractorStatus(False, f"HF RTMW dependencies missing: {exc}", [], None)

        try:
            preload_warning = _preload_onnxruntime_cuda_dlls(ort)
            providers = list(ort.get_available_providers())
            device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
            root = Path(__file__).resolve().parents[1]
            local_dir = root / "model" / "signaturk" / "models" / "rtmw_hf"
            local_det = local_dir / "yolox_m.onnx"
            local_pose_balanced = local_dir / "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"
            local_pose_large = local_dir / "rtmw-dw-x-l_simcc-cocktail14.onnx"
            local_pose = local_pose_balanced if local_pose_balanced.exists() else local_pose_large
            det_path = os.getenv("SIGNTURK_HF_RTMW_DET") or os.getenv("SIGNTURK_RTMPOSE_DET") or (str(local_det) if local_det.exists() else HF_RTMW_DET_URL)
            pose_path = os.getenv("SIGNTURK_HF_RTMW_POSE") or os.getenv("SIGNTURK_RTMPOSE_POSE") or (str(local_pose) if local_pose.exists() else HF_RTMW_POSE_URL)
            pose_input_env = os.getenv("SIGNTURK_HF_RTMW_POSE_INPUT", "").strip()
            if pose_input_env:
                pose_input_size = tuple(int(item.strip()) for item in pose_input_env.split(",", 1))
            elif "256x192" in str(pose_path):
                pose_input_size = (192, 256)
            else:
                pose_input_size = (288, 384)
            self._model = Wholebody(
                det=det_path,
                pose=pose_path,
                det_input_size=(640, 640),
                pose_input_size=pose_input_size,
                to_openpose=False,
                mode="balanced",
                backend="onnxruntime",
                device=device,
            )
            self.backend_name = "hf_rtmw_wholebody"
            message = "HF RTMW wholebody ready" if not preload_warning else f"HF RTMW wholebody ready; {preload_warning}"
            return ExtractorStatus(True, message, providers, device)
        except Exception as exc:
            return ExtractorStatus(False, f"HF RTMW init failed: {exc}", providers if "providers" in locals() else [], None)

    def _init_hf_vitpose(self) -> ExtractorStatus:
        try:
            import onnxruntime as ort
            from rtmlib import Custom
        except Exception as exc:
            return ExtractorStatus(False, f"HF ViTPose dependencies missing: {exc}", [], None)

        try:
            preload_warning = _preload_onnxruntime_cuda_dlls(ort)
            providers = list(ort.get_available_providers())
            device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
            root = Path(__file__).resolve().parents[1]
            local_dir = root / "model" / "signaturk" / "models" / "hf_vitpose"
            local_det = local_dir / "yolox_s.onnx"
            local_pose = local_dir / "vitpose-s-wholebody.onnx"
            det_path = os.getenv("SIGNTURK_HF_VITPOSE_DET") or (str(local_det) if local_det.exists() else HF_VITPOSE_DET_URL)
            pose_path = os.getenv("SIGNTURK_HF_VITPOSE_POSE") or (str(local_pose) if local_pose.exists() else HF_VITPOSE_POSE_URL)
            self._model = Custom(
                det_class="YOLOX",
                det=det_path,
                det_input_size=(640, 640),
                det_mode="multiclass",
                det_categories=[0],
                pose_class="ViTPose",
                pose=pose_path,
                pose_input_size=(192, 256),
                to_openpose=False,
                backend="onnxruntime",
                device=device,
            )
            self.backend_name = "hf_vitpose_wholebody"
            message = "HF ViTPose wholebody fallback ready" if not preload_warning else f"HF ViTPose wholebody fallback ready; {preload_warning}"
            return ExtractorStatus(True, message, providers, device)
        except Exception as exc:
            return ExtractorStatus(False, f"HF ViTPose init failed: {exc}", providers if "providers" in locals() else [], None)

    def _init_mediapipe_fallback(self) -> ExtractorStatus:
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            root = __import__("pathlib").Path(__file__).resolve().parents[1]
            model_dir = root / "model" / "signaturk" / "models" / "mediapipe"
            pose_model = model_dir / "pose_landmarker_lite.task"
            hand_model = model_dir / "hand_landmarker.task"
            if not pose_model.exists() or not hand_model.exists():
                raise FileNotFoundError(f"Missing MediaPipe task files in {pose_model.parent}")

            self._mp_image_cls = __import__("mediapipe").Image
            self._mp_image_format = __import__("mediapipe").ImageFormat
            self._fallback = {
                "pose": vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=str(pose_model)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.35,
                        min_pose_presence_confidence=0.35,
                    )
                ),
                "hand": vision.HandLandmarker.create_from_options(
                    vision.HandLandmarkerOptions(
                        base_options=python.BaseOptions(model_asset_path=str(hand_model)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.35,
                        min_hand_presence_confidence=0.35,
                    )
                ),
            }
            self.backend_name = "mediapipe_tasks_fallback"
            return ExtractorStatus(True, "MediaPipe Tasks fallback ready", [], "cpu")
        except Exception as tasks_exc:
            try:
                import mediapipe as mp
                holistic_module = getattr(getattr(mp, "solutions", None), "holistic", None)
                if holistic_module is None:
                    from mediapipe.python.solutions import holistic as holistic_module

                self._fallback = holistic_module.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=0.35,
                    min_tracking_confidence=0.35,
                )
                self.backend_name = "mediapipe_fallback"
                return ExtractorStatus(True, "MediaPipe fallback ready", [], "cpu")
            except Exception as exc:
                return ExtractorStatus(False, f"tasks={tasks_exc}; solutions={exc}", [], None)

    def infer_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self._model is None and self._fallback is None:
            raise RuntimeError(self.status.message)
        if self._fallback is not None and self._model is None:
            if isinstance(self._fallback, dict):
                return self._infer_frame_mediapipe_tasks(frame_rgb)
            return self._infer_frame_mediapipe(frame_rgb)
        image_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        keypoints, scores = self._model(image_bgr)
        return rtmlib_to_landmarks(
            keypoints,
            scores,
            frame_rgb.shape[:2],
            kpt_thr=self.kpt_thr,
            filter_hands=self.filter_model_hands,
            min_hand_points=self.min_hand_points,
            min_hand_mean_conf=self.min_hand_mean_conf,
            max_hand_bbox_span=self.max_hand_bbox_span,
            max_hand_bbox_area=self.max_hand_bbox_area,
        )

    def _infer_frame_mediapipe(self, frame_rgb: np.ndarray) -> np.ndarray:
        from .feature_builder import HAND_COUNT, LANDMARK_DIM, POSE_COUNT, TOTAL_LANDMARKS

        result = self._fallback.process(np.ascontiguousarray(frame_rgb))
        landmarks = np.zeros((TOTAL_LANDMARKS, LANDMARK_DIM), dtype=np.float32)

        def copy_landmarks(source, offset: int, count: int) -> None:
            if source is None:
                return
            for idx, lm in enumerate(source.landmark[:count]):
                landmarks[offset + idx, 0] = float(np.clip(lm.x, 0.0, 1.0))
                landmarks[offset + idx, 1] = float(np.clip(lm.y, 0.0, 1.0))
                landmarks[offset + idx, 2] = 0.0
                landmarks[offset + idx, 3] = float(getattr(lm, "visibility", 1.0))

        copy_landmarks(result.pose_landmarks, 0, POSE_COUNT)
        copy_landmarks(result.left_hand_landmarks, POSE_COUNT, HAND_COUNT)
        copy_landmarks(result.right_hand_landmarks, POSE_COUNT + HAND_COUNT, HAND_COUNT)
        return landmarks

    def _infer_frame_mediapipe_tasks(self, frame_rgb: np.ndarray) -> np.ndarray:
        from .feature_builder import HAND_COUNT, LANDMARK_DIM, POSE_COUNT, TOTAL_LANDMARKS

        image = self._mp_image_cls(image_format=self._mp_image_format.SRGB, data=np.ascontiguousarray(frame_rgb))
        pose_result = self._fallback["pose"].detect(image)
        hand_result = self._fallback["hand"].detect(image)
        landmarks = np.zeros((TOTAL_LANDMARKS, LANDMARK_DIM), dtype=np.float32)

        def copy_list(source, offset: int, count: int, visibility: bool = True) -> None:
            if source is None:
                return
            for idx, lm in enumerate(source[:count]):
                landmarks[offset + idx, 0] = float(np.clip(lm.x, 0.0, 1.0))
                landmarks[offset + idx, 1] = float(np.clip(lm.y, 0.0, 1.0))
                landmarks[offset + idx, 2] = 0.0
                landmarks[offset + idx, 3] = float(getattr(lm, "visibility", 1.0) if visibility else 1.0)

        if pose_result.pose_landmarks:
            copy_list(pose_result.pose_landmarks[0], 0, POSE_COUNT, visibility=True)

        for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness[0].category_name.lower() if handedness else ""
            if label == "left":
                copy_list(hand_landmarks, POSE_COUNT, HAND_COUNT, visibility=False)
            elif label == "right":
                copy_list(hand_landmarks, POSE_COUNT + HAND_COUNT, HAND_COUNT, visibility=False)
        return landmarks

    def infer_frames(self, frames_rgb: list[np.ndarray]) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        landmarks = [self.infer_frame(frame) for frame in frames_rgb]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return np.stack(landmarks).astype(np.float32), elapsed_ms


def decode_jpeg_bytes(data: bytes, max_width: int = 640) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode JPEG frame")
    height, width = image_bgr.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        image_bgr = cv2.resize(image_bgr, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_video_frames(path: str, target_max_width: int = 640) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        height, width = frame_bgr.shape[:2]
        if width > target_max_width:
            scale = target_max_width / float(width)
            frame_bgr = cv2.resize(frame_bgr, (target_max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError("Decoded zero frames")
    return frames
