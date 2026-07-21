"""Backward-compatible shim for the relocated text_processing web layer.

The FastAPI router and its schemas / pipeline cache / async job store now
live in the ``text_processing.web`` package. This module re-exports that
surface so existing imports keep working unchanged:

    from text_processing_routes import router      # backend.py, devserver.py
    import text_processing_routes as routes        # tests
"""

from __future__ import annotations

from text_processing.web.cache import _PIPELINE_CACHE, PipelineCache
from text_processing.web.jobs import _EXECUTOR, _JOB_STORE, JobStore
from text_processing.web.router import (
    BASE_DIR,
    FRONTEND_DIR,
    TTS_OUTPUT_DIR,
    _get_pipeline,
    _resolve_model_key,
    _run_correction,
    _run_job,
    clear_cache,
    correct,
    correct_async,
    demo_page,
    get_audio,
    get_job,
    list_models,
    ping_model,
    router,
    text_health,
    unload_model,
)
from text_processing.web.schemas import (
    CorrectRequest,
    CorrectResponse,
    JobStatusResponse,
    JobSubmitResponse,
    ModelInfo,
    ModelsResponse,
    PingRequest,
    PingResponse,
)

__all__ = [
    "router",
    # schemas
    "CorrectRequest",
    "CorrectResponse",
    "ModelInfo",
    "ModelsResponse",
    "PingRequest",
    "PingResponse",
    "JobSubmitResponse",
    "JobStatusResponse",
    # infrastructure
    "PipelineCache",
    "JobStore",
    # endpoints / helpers
    "list_models",
    "ping_model",
    "correct",
    "correct_async",
    "get_job",
    "text_health",
    "clear_cache",
    "unload_model",
    "get_audio",
    "demo_page",
]
