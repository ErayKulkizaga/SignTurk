"""Pydantic request/response models for the /api/text/* endpoints."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CorrectRequest(BaseModel):
    words: List[str] = Field(..., description="Raw words from the LSTM model")
    use_ml: bool = False
    model_key: Optional[str] = None
    # Raw HF model id to test any model (e.g. "Qwen/Qwen3-8B"); overrides
    # model_key. model_arch picks the adapter (inferred from the name if None).
    model_name: Optional[str] = None
    model_arch: Optional[str] = None
    synthesize_audio: bool = True


class CorrectResponse(BaseModel):
    words: List[str]
    sentence: str
    source: str
    ml_latency_ms: float
    audio_url: Optional[str] = None
    tts_status: str = "disabled"
    rejected_candidate: Optional[str] = None
    rejection_reason: Optional[str] = None
    reason: str = ""
    # Action-oriented Turkish reason the ML layer failed/was unavailable
    # (+ stable category for the UI to branch on). Null when ML succeeded
    # or was never invoked (by-design rule-based).
    ml_error: Optional[str] = None
    ml_error_category: Optional[str] = None


class ModelInfo(BaseModel):
    key: str
    hf_name: str
    arch: str
    approx_size_mb: int
    instruction_tuned: bool
    turkish_native: bool
    recommended: bool
    conversational: bool
    notes: str


class ModelsResponse(BaseModel):
    default: str
    default_api: str
    default_local: str
    models: List[ModelInfo]


class PingRequest(BaseModel):
    """Which model to connectivity-test. Mirrors the model-selection fields
    of ``CorrectRequest``; all optional → defaults to the cloud API model."""

    model_key: Optional[str] = None
    model_name: Optional[str] = None
    model_arch: Optional[str] = None


class PingResponse(BaseModel):
    ok: bool
    model_key: str
    hf_name: str
    arch: str
    latency_ms: float
    detail: str  # Turkish, user-facing
    category: Optional[str] = None  # "ok" | "local" | error category


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending | done | error
    result: Optional[CorrectResponse] = None
    error: Optional[str] = None
