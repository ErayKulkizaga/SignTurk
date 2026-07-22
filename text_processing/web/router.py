"""FastAPI endpoints for the text_processing demo (the /api/text/* router).

Composition root for the web layer: wires the request/response schemas, the
pipeline cache, and the async job store together behind one APIRouter.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from text_processing import (
    DECISION_LOG,
    DEFAULT_API_MODEL_KEY,
    DEFAULT_LOCAL_MODEL_KEY,
    DEFAULT_MODEL_KEY,
    PROMPT_VERSION,
    GrammarConfig,
    PipelineConfig,
    SignTextPipeline,
    TTSConfig,
    list_available_models,
    list_prompt_versions,
)
from text_processing.tts import tts_engine_status

from .cache import _PIPELINE_CACHE, CacheKey
from .jobs import _ASYNC_WORKERS, _EXECUTOR, _JOB_STORE
from .schemas import (
    CorrectRequest,
    CorrectResponse,
    JobStatusResponse,
    JobSubmitResponse,
    ModelInfo,
    ModelsResponse,
    PingRequest,
    PingResponse,
)

logger = logging.getLogger(__name__)

# Repo root: web/ -> text_processing/ -> <root>. Keeps frontend/ and
# uploads/tts/ at the same paths the old root-level module used.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
TTS_OUTPUT_DIR = BASE_DIR / "uploads" / "tts"
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api/text", tags=["text-processing"])

def _resolve_model_key(use_ml: bool, model_key: Optional[str]) -> str:
    """Pick the effective model key.

    When ML is requested without naming a model, default to the cloud API
    model (``DEFAULT_API_MODEL_KEY``) rather than silently downloading a
    multi-hundred-MB local checkpoint.
    """
    if model_key:
        return model_key
    return DEFAULT_API_MODEL_KEY if use_ml else DEFAULT_MODEL_KEY


def _get_pipeline(
    use_ml: bool,
    model_key: Optional[str],
    model_name: Optional[str] = None,
    model_arch: Optional[str] = None,
) -> SignTextPipeline:
    # A raw model_name implies the ML path; it overrides the registry key.
    effective_ml = use_ml or bool(model_name)
    # Resolve the effective registry key ONCE so the cache key and the
    # pipeline's GrammarConfig can't diverge (ML-without-model → API default).
    resolved_model_key = _resolve_model_key(effective_ml, model_key)
    # The same custom model id can be requested under different architectures
    # (seq2seq / causal / inference-api), which build different adapters — so
    # fold the arch into the key to avoid reusing the wrong cached pipeline.
    # Registry keys are unaffected (their arch is fixed by the ModelSpec).
    resolved = f"{model_name}#{model_arch or 'auto'}" if model_name else resolved_model_key
    cache_key: CacheKey = (effective_ml, resolved)

    def _factory() -> SignTextPipeline:
        cfg = PipelineConfig(
            grammar=GrammarConfig(
                use_ml=effective_ml,
                model_key=resolved_model_key,
                model_name_override=model_name,
                model_arch_override=model_arch,
            ),
            tts=TTSConfig(output_dir=TTS_OUTPUT_DIR),
            # synthesize_audio is decided per request in ``correct`` — the
            # cached pipeline must not bake in one choice.
            synthesize_audio=True,
        )
        return SignTextPipeline(cfg)

    return _PIPELINE_CACHE.get_or_create(cache_key, _factory)


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(
        default=DEFAULT_MODEL_KEY,
        default_api=DEFAULT_API_MODEL_KEY,
        default_local=DEFAULT_LOCAL_MODEL_KEY,
        models=[
            ModelInfo(
                key=s.key,
                hf_name=s.hf_name,
                arch=s.arch,
                approx_size_mb=s.approx_size_mb,
                instruction_tuned=s.instruction_tuned,
                turkish_native=s.turkish_native,
                recommended=s.recommended,
                conversational=s.conversational,
                notes=s.notes,
            )
            for s in list_available_models()
        ],
    )


@router.post("/ping", response_model=PingResponse)
def ping_model(req: PingRequest) -> PingResponse:
    """Connectivity-test a (cloud) grammar model with one tiny probe call.

    Lets the UI show a clear ✓/✗ with an actionable Turkish reason
    *before* a real correction — instead of users discovering a dead model
    (bad token, exhausted quota, unserved id) only mid-request. Local
    models are reported as such without downloading weights.
    """
    from text_processing import MLGrammarCorrector  # optional ML deps: import lazily

    cfg = GrammarConfig(
        use_ml=True,
        model_key=req.model_key or DEFAULT_API_MODEL_KEY,
        model_name_override=req.model_name,
        model_arch_override=req.model_arch,
    )
    try:
        spec = cfg.resolve_spec()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    resolved = req.model_name or req.model_key or DEFAULT_API_MODEL_KEY
    if spec.arch != "inference-api":
        return PingResponse(
            ok=True,
            model_key=resolved,
            hf_name=spec.hf_name,
            arch=spec.arch,
            latency_ms=0.0,
            detail="Yerel model — bağlantı testi yok; ilk kullanımda indirilir/yüklenir.",
            category="local",
        )

    corrector = MLGrammarCorrector(cfg)
    t0 = time.monotonic()
    out = corrector.correct(["merhaba", "nasılsın"])
    latency_ms = (time.monotonic() - t0) * 1000.0
    if out is not None:
        return PingResponse(
            ok=True,
            model_key=resolved,
            hf_name=spec.hf_name,
            arch=spec.arch,
            latency_ms=latency_ms,
            detail="Bağlantı başarılı — model yanıt verdi.",
            category="ok",
        )
    err = corrector.last_error
    return PingResponse(
        ok=False,
        model_key=resolved,
        hf_name=spec.hf_name,
        arch=spec.arch,
        latency_ms=latency_ms,
        detail=err.message if err else "Model yanıt vermedi (bilinmeyen sebep).",
        category=err.category if err else "unknown",
    )


def _run_correction(req: CorrectRequest) -> CorrectResponse:
    """Shared correction core for both the sync and async endpoints.

    Raises ``ValueError`` for bad input (→ 400) and lets other exceptions
    propagate (→ 500 sync / job error async).
    """
    cleaned = [w.strip() for w in req.words if w and w.strip()]
    if not cleaned:
        raise ValueError("words must contain at least one non-empty token")
    pipeline = _get_pipeline(
        req.use_ml, req.model_key, req.model_name, req.model_arch
    )  # ValueError on unknown model
    # Per-call gate: when synthesize_audio is False the pipeline never invokes
    # gTTS (previously it always ran and the URL was just hidden).
    result = pipeline.correct(cleaned, synthesize_audio=req.synthesize_audio)
    audio_url = (
        f"/api/text/audio/{result.audio_path.name}" if result.audio_path is not None else None
    )
    return CorrectResponse(
        words=result.words,
        sentence=result.sentence,
        source=result.grammar_source,
        ml_latency_ms=result.ml_latency_ms,
        audio_url=audio_url,
        tts_status=result.tts_status,
        rejected_candidate=result.rejected_candidate,
        rejection_reason=result.rejection_reason,
        reason=result.reason,
        ml_error=result.ml_error,
        ml_error_category=result.ml_error_category,
    )


@router.post("/correct", response_model=CorrectResponse)
def correct(req: CorrectRequest) -> CorrectResponse:
    try:
        return _run_correction(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("text correct failed")
        raise HTTPException(status_code=500, detail=f"correction failed: {e}") from e


@router.post("/correct/async", response_model=JobSubmitResponse)
def correct_async(req: CorrectRequest) -> JobSubmitResponse:
    """Submit a correction to run off the request thread (for slow ML calls).

    Returns a ``job_id`` to poll via ``GET /api/text/jobs/{job_id}`` so the HF
    Inference API latency / cold start never holds the HTTP request open.
    """
    try:
        job_id = _JOB_STORE.create()
    except RuntimeError as e:
        # All slots occupied by still-pending jobs — apply backpressure rather
        # than evict in-flight work.
        raise HTTPException(status_code=429, detail=str(e)) from e
    _EXECUTOR.submit(_run_job, job_id, req)
    return JobSubmitResponse(job_id=job_id, status="pending")


def _run_job(job_id: str, req: CorrectRequest) -> None:
    try:
        _JOB_STORE.set_done(job_id, _run_correction(req))
    except Exception as e:  # noqa: BLE001 — surfaced to the client as job error
        _JOB_STORE.set_error(job_id, str(e))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = _JOB_STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found (unknown or expired)")
    return JobStatusResponse(job_id=job_id, status=job.status, result=job.result, error=job.error)


@router.get("/health")
def text_health() -> Dict[str, object]:
    """Diagnostics for the text/LLM layer.

    Reports whether an HF token is configured (anonymous Inference API calls
    are heavily rate-limited), the default model policy, the live pipeline
    cache (which models are loaded / failed / how busy), recent ml-vs-rule
    decision stats, and whether the gTTS engine is installed.
    """
    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN"))
    return {
        "status": "ok",
        "hf_token_present": hf_token_present,
        "default_model": DEFAULT_MODEL_KEY,
        "default_api_model": DEFAULT_API_MODEL_KEY,
        "default_local_model": DEFAULT_LOCAL_MODEL_KEY,
        "cache": {
            "max_size": _PIPELINE_CACHE.max_size,
            "ttl_seconds": _PIPELINE_CACHE.ttl_seconds,
            "entries": _PIPELINE_CACHE.snapshot(),
        },
        "recent_decisions": DECISION_LOG.summary(),
        "prompts": {"default": PROMPT_VERSION, "available": list_prompt_versions()},
        "tts": tts_engine_status(),
        "jobs": {
            "pending": _JOB_STORE.pending_count(),
            "max_size": _JOB_STORE.max_size,
            "ttl_seconds": _JOB_STORE.ttl_seconds,
            "workers": _ASYNC_WORKERS,
        },
    }


@router.delete("/cache")
def clear_cache() -> Dict[str, object]:
    """Unload every cached pipeline (frees model RAM)."""
    removed = _PIPELINE_CACHE.clear()
    return {"cleared": removed}


@router.delete("/cache/{model_key:path}")
def unload_model(model_key: str) -> Dict[str, object]:
    """Unload a specific model's pipelines (both ML and rule-based variants)."""
    removed = sum(1 for use_ml in (True, False) if _PIPELINE_CACHE.unload((use_ml, model_key)))
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"no cached pipeline for {model_key!r}")
    return {"unloaded": model_key, "entries_removed": removed}


@router.get("/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    audio_path = TTS_OUTPUT_DIR / filename
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="audio not found")
    media_type = "audio/wav" if filename.lower().endswith(".wav") else "audio/mpeg"
    return FileResponse(audio_path, media_type=media_type, filename=filename)


@router.get("/demo", response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    demo_file = FRONTEND_DIR / "text_processing_demo.html"
    if not demo_file.is_file():
        raise HTTPException(status_code=404, detail="demo page is missing")
    return HTMLResponse(demo_file.read_text(encoding="utf-8"))
