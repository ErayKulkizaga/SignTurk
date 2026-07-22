from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "research" / "model_226"
ROOT = ASSET_ROOT


@dataclass(frozen=True)
class BackendSettings:
    root: Path
    config_path: Path
    label_map_path: Path
    class_display_names_path: Path
    static_dir: Path
    target_frames: int = 32
    default_capture_seconds: float = 2.5
    min_segment_seconds: float = 1.2
    max_segment_seconds: float = 3.2
    frontend_send_fps: int = 8
    prediction_cooldown_ms: int = 3500
    calibration_seconds: float = 1.2
    hand_hold_seconds: float = 0.0
    required_stable_votes: int = 2
    top1_threshold: float = 0.70
    margin_threshold: float = 0.12
    motion_threshold: float = 0.004
    min_shoulder_width_ratio: float = 0.20
    max_shoulder_width_ratio: float = 0.42
    center_x_min: float = 0.36
    center_x_max: float = 0.64
    upper_body_y_min: float = 0.24
    upper_body_y_max: float = 0.54
    min_pose_confidence: float = 0.25
    min_any_hand_visible_rate: float = 0.12


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def default_settings() -> BackendSettings:
    return BackendSettings(
        root=ASSET_ROOT,
        config_path=ASSET_ROOT / "config" / "backend_config.json",
        label_map_path=ASSET_ROOT / "config" / "labels" / "label_map.json",
        class_display_names_path=ASSET_ROOT / "config" / "labels" / "class_display_names.json",
        static_dir=PROJECT_ROOT / "frontend",
    )


def load_backend_config(settings: BackendSettings | None = None) -> dict[str, Any]:
    settings = settings or default_settings()
    return load_json(settings.config_path)


def load_id2label(settings: BackendSettings | None = None) -> dict[int, str]:
    settings = settings or default_settings()
    label2id = load_json(settings.label_map_path)
    return {int(idx): str(label) for label, idx in label2id.items()}


def load_display_names(settings: BackendSettings | None = None) -> dict[str, str]:
    settings = settings or default_settings()
    if not settings.class_display_names_path.exists():
        return {}
    raw = load_json(settings.class_display_names_path)
    return {str(key): str(value) for key, value in raw.items()}
