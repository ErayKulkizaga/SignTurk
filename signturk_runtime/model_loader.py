from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import BackendSettings, default_settings, load_backend_config, load_display_names, load_id2label
from .feature_builder import FeatureBundle


@dataclass
class Prediction:
    probabilities: np.ndarray
    top3: list[dict[str, float | int | str]]
    latency_ms: float


def _load_tf():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    @keras.utils.register_keras_serializable(package="signaturk")
    class TemporalAttentionPool(layers.Layer):
        def call(self, inputs):
            x, weight = inputs
            return tf.reduce_sum(x * weight, axis=1)

        def compute_output_shape(self, input_shape):
            return (input_shape[0][0], input_shape[0][-1])

    return keras, TemporalAttentionPool


class SignTurkEnsemble:
    def __init__(self, settings: BackendSettings | None = None):
        self.settings = settings or default_settings()
        self.config = load_backend_config(self.settings)
        self.id2label = load_id2label(self.settings)
        self.display_names = load_display_names(self.settings)
        self.streams = list(self.config["streams"])
        self.weights = {str(k): float(v) for k, v in self.config["weights"].items()}
        self.model_files = {str(k): str(v) for k, v in self.config["model_files"].items()}
        self.models: dict[str, Any] = {}
        self.predict_fns: dict[str, Any] = {}

    def validate_config(self) -> list[str]:
        errors = []
        if "skeleton" in self.streams:
            errors.append("backend_config must not include legacy skeleton stream")
        weight_sum = sum(self.weights.get(stream, 0.0) for stream in self.streams)
        if abs(weight_sum - 1.0) > 1e-3:
            errors.append(f"ensemble weights must sum to 1.0, got {weight_sum}")
        if len(self.id2label) != int(self.config.get("num_classes", 0)):
            errors.append(f"label count mismatch: {len(self.id2label)}")
        for stream in self.streams:
            rel = self.model_files.get(stream)
            if not rel:
                errors.append(f"missing model file entry for {stream}")
                continue
            if not (self.settings.root / rel).exists():
                errors.append(f"model file missing for {stream}: {rel}")
        return errors

    def load(self) -> None:
        errors = self.validate_config()
        if errors:
            raise RuntimeError("; ".join(errors))
        keras, TemporalAttentionPool = _load_tf()
        for stream in self.streams:
            path = self.settings.root / self.model_files[stream]
            self.models[stream] = keras.models.load_model(
                path,
                custom_objects={"TemporalAttentionPool": TemporalAttentionPool},
                safe_mode=False,
                compile=False,
            )
            self._compile_predict_fn(stream)

    @property
    def loaded(self) -> bool:
        return set(self.models) == set(self.streams)

    def _inputs_for_stream(self, stream: str, features: FeatureBundle) -> list[np.ndarray]:
        if stream == "hand_stream":
            return features.hand_inputs
        return features.skeleton_inputs

    def _input_signature_for_stream(self, stream: str):
        import tensorflow as tf

        seq_len = int(self.config.get("frame_count", 32))
        dims = [168, 168, 39] if stream == "hand_stream" else [300, 144, 300, 144, 39]
        return [tf.TensorSpec(shape=(None, seq_len, dim), dtype=tf.float32) for dim in dims]

    def _compile_predict_fn(self, stream: str) -> None:
        import tensorflow as tf

        model = self.models[stream]
        signature = self._input_signature_for_stream(stream)

        @tf.function(input_signature=signature, reduce_retracing=True)
        def predict_fn(*inputs):
            return model(list(inputs), training=False)

        self.predict_fns[stream] = predict_fn

    def _predict_stream(self, stream: str, features: FeatureBundle) -> np.ndarray:
        inputs = self._inputs_for_stream(stream, features)
        predict_fn = self.predict_fns.get(stream)
        output = predict_fn(*inputs) if predict_fn is not None else self.models[stream](inputs, training=False)
        return np.asarray(output, dtype=np.float32)

    def _ensure_loaded(self, streams: list[str]) -> None:
        missing = [stream for stream in streams if stream not in self.models]
        if not missing:
            return
        errors = self.validate_config()
        if errors:
            raise RuntimeError("; ".join(errors))
        keras, TemporalAttentionPool = _load_tf()
        for stream in missing:
            path = self.settings.root / self.model_files[stream]
            self.models[stream] = keras.models.load_model(
                path,
                custom_objects={"TemporalAttentionPool": TemporalAttentionPool},
                safe_mode=False,
                compile=False,
            )
            self._compile_predict_fn(stream)

    def predict(self, features: FeatureBundle, streams: list[str] | None = None) -> Prediction:
        active_streams = list(streams or self.streams)
        self._ensure_loaded(active_streams)
        start = time.perf_counter()
        ensemble = np.zeros((1, int(self.config["num_classes"])), dtype=np.float32)
        weight_sum = sum(self.weights[stream] for stream in active_streams)
        for stream in active_streams:
            probs = self._predict_stream(stream, features)
            ensemble += probs * (self.weights[stream] / max(weight_sum, 1e-8))
        ensemble /= max(float(ensemble.sum(axis=1, keepdims=True)[0, 0]), 1e-8)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return Prediction(probabilities=ensemble[0], top3=self.topk(ensemble[0], 3), latency_ms=latency_ms)

    def warmup(self, streams: list[str] | None = None) -> float:
        active_streams = list(streams or self.streams)
        self._ensure_loaded(active_streams)
        seq_len = int(self.config.get("frame_count", 32))
        skeleton_inputs = [
            np.zeros((1, seq_len, 300), dtype=np.float32),
            np.zeros((1, seq_len, 144), dtype=np.float32),
            np.zeros((1, seq_len, 300), dtype=np.float32),
            np.zeros((1, seq_len, 144), dtype=np.float32),
            np.zeros((1, seq_len, 39), dtype=np.float32),
        ]
        hand_inputs = [
            np.zeros((1, seq_len, 168), dtype=np.float32),
            np.zeros((1, seq_len, 168), dtype=np.float32),
            np.zeros((1, seq_len, 39), dtype=np.float32),
        ]
        dummy = FeatureBundle(
            skeleton_inputs=skeleton_inputs,
            hand_inputs=hand_inputs,
            streams={},
            landmarks=np.zeros((seq_len, 75, 4), dtype=np.float32),
        )
        start = time.perf_counter()
        self.predict(dummy, streams=active_streams)
        return (time.perf_counter() - start) * 1000.0

    def topk(self, probs: np.ndarray, k: int = 3) -> list[dict[str, float | int | str]]:
        indices = np.argsort(-probs)[:k]
        rows = []
        for idx in indices:
            class_id = int(idx)
            raw_label = self.id2label.get(class_id, str(class_id))
            display = self.display_names.get(raw_label) or self.display_names.get(str(class_id))
            if not display:
                display = f"Hareket {raw_label}"
            rows.append({
                "class_id": class_id,
                "label": display,
                "raw_label": raw_label,
                "confidence": float(probs[idx]),
            })
        return rows
