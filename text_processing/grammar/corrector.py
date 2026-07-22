"""Public hybrid facade: rule-based + validated ML, arbitrated per request."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .adapters import (
    MLError,
    ModelAdapter,
    _build_adapter,
    _classify_ml_error,
    _scrub_exception_text,
)
from .arbiter import _arbiter_decide, _has_critical_marker, _is_simple_input
from .linguistics import normalize_words
from .prompts import _resolve_prompt
from .registry import (
    DEFAULT_MODEL_KEY,
    MODEL_REGISTRY,
    PROMPT_VERSION,
    GrammarConfig,
    ModelSpec,
)
from .rules import RuleBasedCorrector
from .telemetry import DECISION_LOG, DecisionRecord
from .validation import validate_ml_output

logger = logging.getLogger(__name__)

# ───────────────────────── ML grammar layer ─────────────────────────


class MLGrammarCorrector:
    """Lazily loads a HF model and produces a candidate sentence.

    Loading and inference failures are swallowed and surfaced as ``None``
    so callers can fall back cleanly.
    """

    def __init__(self, config: GrammarConfig) -> None:
        self.config = config
        self.spec = config.resolve_spec()
        _resolve_prompt(config.prompt_version)  # fail fast on an unknown prompt version
        self._lock = threading.Lock()
        self._adapter: Optional[ModelAdapter] = None
        self._loaded = False
        self._load_failed = False
        self._load_error: Optional[MLError] = None
        self._inference_count = 0
        self._total_latency_ms = 0.0
        # Most recent classified failure (load or inference), surfaced so
        # the facade can explain *why* the ML layer fell back.
        self.last_error: Optional[MLError] = None

    @property
    def avg_latency_ms(self) -> float:
        return self._total_latency_ms / self._inference_count if self._inference_count else 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_failed(self) -> bool:
        return self._load_failed

    @property
    def inference_count(self) -> int:
        return self._inference_count

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            try:
                logger.info(
                    "Loading grammar model %s (%s, ~%d MB) on %s",
                    self.spec.hf_name,
                    self.spec.arch,
                    self.spec.approx_size_mb,
                    self.config.device,
                )
                self._adapter = _build_adapter(self.spec, self.config)
                self._loaded = True
                return True
            except Exception as e:
                logger.warning(
                    "Grammar model load failed (%s): %s",
                    self.spec.hf_name,
                    _scrub_exception_text(e),
                )
                self._load_failed = True
                self._load_error = _classify_ml_error(e)
                return False

    def generate_candidate(self, words: List[str]) -> Tuple[Optional[str], Optional[MLError]]:
        """Run one ML inference and return ``(candidate, error)``.

        The error is *returned*, never stored on the instance, so concurrent
        requests that share this cached corrector can't clobber each other's
        failure reason. A ``None`` candidate always carries a classified
        reason (load failure, transport error, empty output, or timeout) —
        ``(None, None)`` never happens.
        """
        if not self._ensure_loaded() or self._adapter is None:
            return None, (self._load_error or MLError("load_failed", "ML modeli yüklenemedi."))
        try:
            t0 = time.monotonic()
            text = self._adapter.generate(words, self.config)
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._inference_count += 1
            self._total_latency_ms += elapsed_ms
        except Exception as e:
            # Adapters RAISE transport/load failures (local adapters always
            # did; the inference-API adapter now does too) — classify here.
            logger.warning("Grammar ML inference failed: %s", _scrub_exception_text(e))
            return None, _classify_ml_error(e)
        # Local CPU/GPU inference uses the tight local timeout; the remote
        # API path gets its own (more lenient) network budget so first-call
        # cold-starts don't trip the local guard.
        timeout_s = (
            self.config.inference_api_timeout_seconds
            if self.spec.arch == "inference-api"
            else self.config.inference_timeout_seconds
        )
        if text is None:
            # Transport errors raise (handled above); a None here means the
            # model produced empty/blank output — still actionable, not null.
            return None, MLError("empty", "Model boş ya da geçersiz bir yanıt döndürdü.")
        # A *valid* candidate that merely blew the latency budget is
        # discarded — rule-based is the safe floor — and reported as slow.
        if elapsed_ms / 1000 > timeout_s:
            logger.warning(
                "Grammar ML inference slow (%.2fs > %.2fs) for %s",
                elapsed_ms / 1000,
                timeout_s,
                self.spec.hf_name,
            )
            return None, MLError(
                "timeout",
                f"ML çıkarımı çok yavaş ({elapsed_ms / 1000:.1f}s > "
                f"{timeout_s:.0f}s eşiği) — daha küçük/başka bir model deneyin.",
            )
        return text, None

    def correct(self, words: List[str]) -> Optional[str]:
        """Backward-compatible string API. Stores ``last_error`` for callers
        that probe a *fresh* (non-shared) instance — the ``/ping`` endpoint
        and tests. The hybrid facade calls ``generate_candidate`` directly to
        stay race-free under concurrency.
        """
        text, self.last_error = self.generate_candidate(words)
        return text


# ───────────────────────── Hybrid facade ─────────────────────────


@dataclass
class GrammarResult:
    sentence: str
    source: str
    rejected_candidate: Optional[str] = None
    rejection_reason: Optional[str] = None
    ml_latency_ms: float = 0.0
    # Human-readable explanation of *why* the arbiter chose the engine
    # that produced ``sentence`` (e.g. "fast-track simple input",
    # "passed arbiter checks", "ml dropped key roots").
    reason: str = ""
    # When the ML layer was attempted but failed (or was unavailable), an
    # action-oriented Turkish message + stable category explaining why.
    # Both ``None`` when ML succeeded or was never invoked (by-design
    # rule-based: fast-track / disabled / empty input).
    ml_error: Optional[str] = None
    ml_error_category: Optional[str] = None


class GrammarCorrector:
    """Public hybrid corrector. ML candidate is validated before being used.

    `correct()` returns a plain string for backwards compatibility.
    `correct_detailed()` returns a `GrammarResult` for telemetry/debugging.
    """

    def __init__(self, config: Optional[GrammarConfig] = None) -> None:
        cfg = config or GrammarConfig(
            use_ml=os.environ.get("SIGNAI_USE_ML_GRAMMAR", "0") == "1",
            model_key=os.environ.get("SIGNAI_GRAMMAR_MODEL_KEY", DEFAULT_MODEL_KEY),
            model_name_override=os.environ.get("SIGNAI_GRAMMAR_MODEL") or None,
            model_arch_override=os.environ.get("SIGNAI_GRAMMAR_MODEL_ARCH") or None,
            prompt_version=os.environ.get("SIGNAI_PROMPT_VERSION", PROMPT_VERSION),
            hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN") or None,
        )
        self.config = cfg
        self._rule = RuleBasedCorrector()
        self._ml: Optional[MLGrammarCorrector] = None
        if cfg.use_ml:
            try:
                self._ml = MLGrammarCorrector(cfg)
            except ValueError as e:
                logger.warning("Disabling ML grammar layer: %s", e)
                self._ml = None

    @property
    def _ml_model_label(self) -> str:
        """Identity of the model that actually runs.

        A ``model_name_override`` resolves to a spec with ``key="custom"``;
        report its real ``hf_name`` so health, telemetry, and ``source``
        strings attribute the run to the model that produced it rather than
        the default registry key.
        """
        if self._ml is None:
            return self.config.model_key
        spec = self._ml.spec
        return spec.hf_name if spec.key == "custom" else spec.key

    @property
    def ml_status(self) -> Dict[str, object]:
        """Snapshot of the ML layer's load/usage state for diagnostics."""
        if self._ml is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "model_key": self._ml_model_label,
            "hf_name": self._ml.spec.hf_name,
            "arch": self._ml.spec.arch,
            "loaded": self._ml.is_loaded,
            "load_failed": self._ml.load_failed,
            "inference_count": self._ml.inference_count,
            "avg_latency_ms": round(self._ml.avg_latency_ms, 1),
        }

    def correct(self, words: List[str]) -> str:
        return self.correct_detailed(words).sentence

    def correct_detailed(self, words: List[str]) -> GrammarResult:
        """Public entry point: route the input, then log the decision.

        The routing logic lives in ``_correct_detailed_impl``; this wrapper
        records every decision (structured log line + ring buffer) so the
        many fast-track/validator/arbiter return paths share one exit.
        """
        result = self._correct_detailed_impl(words)
        self._log_decision(words, result)
        return result

    def _log_decision(self, words: List[str], result: GrammarResult) -> None:
        rec = DecisionRecord(
            input_words=list(words),
            model_key=self._ml_model_label,
            prompt_version=self.config.prompt_version,
            source=result.source,
            final_sentence=result.sentence,
            ml_latency_ms=result.ml_latency_ms,
            reason=result.reason,
            rejected_candidate=result.rejected_candidate,
            rejection_reason=result.rejection_reason,
            ml_error_category=result.ml_error_category,
        )
        DECISION_LOG.record(rec)
        logger.info(
            "grammar decision %s",
            json.dumps(
                {
                    "input_word_count": len(rec.input_words),
                    "model_key": rec.model_key,
                    "prompt_version": rec.prompt_version,
                    "source": rec.source,
                    "output_character_count": len(rec.final_sentence),
                    "candidate_rejected": rec.rejected_candidate is not None,
                    "rejection_reason": rec.rejection_reason,
                    "ml_error_category": rec.ml_error_category,
                    "ml_latency_ms": round(rec.ml_latency_ms, 1),
                    "reason": rec.reason,
                },
                ensure_ascii=False,
            ),
        )

    def _correct_detailed_impl(self, words: List[str]) -> GrammarResult:
        """Hybrid corrector with smart-arbiter routing.

        Decision tree::

            normalize ─┐
                       ├─ empty?              → ``empty``
                       ├─ no ML configured?   → ``rule-based`` (reason: ml disabled)
                       ├─ |words| ≤ 2?        → ``rule-based`` (reason: fast-track)
                       └─ otherwise           → run ML and arbitrate
                           ├─ ML None        → ``rule-based`` (reason: ml unavailable)
                           ├─ validator fail → ``rule-based`` (reason: validator rejected)
                           ├─ arbiter "rule" → ``rule-based`` (reason: arbiter findings)
                           └─ arbiter "ml"   → ``ml:<model>``  (reason: passed arbiter)
        """
        # Normalize at the entry point so the ML prompt, validator,
        # rule-based engine, and arbiter all see the same clean tokens.
        cleaned = normalize_words(words)
        if not cleaned:
            return GrammarResult(
                sentence="",
                source="rule-based",
                reason="empty input bypass",
            )

        # The rule-based output is always computed: it's the safe
        # floor for the arbiter, and the fallback for every error path.
        rule_output = self._rule.correct(cleaned)

        if self._ml is None:
            return GrammarResult(
                sentence=rule_output,
                source="rule-based",
                reason="ml disabled",
            )

        # 1a) Universal fast-track for inputs containing a semantically
        #     critical marker (question particle, negation, or number).
        #     The ML model routinely paraphrases these away — the
        #     rule-based engine preserves them deterministically.
        if _has_critical_marker(cleaned):
            return GrammarResult(
                sentence=rule_output,
                source="rule-based",
                reason="fast-track: interrogative/negation/numeric input",
            )

        # 1b) Complexity-based fast-track: trivial 1-2 token streams skip
        #     the network entirely. The rule-based parser is 100% correct
        #     for those and adds zero latency.
        if _is_simple_input(cleaned):
            return GrammarResult(
                sentence=rule_output,
                source="rule-based",
                reason=f"fast-track simple input ({len(cleaned)} tokens)",
            )

        # 2) Complex input: invoke ML and arbitrate. The classified error is
        #    returned alongside the candidate (not read off shared instance
        #    state) so concurrent requests can't cross their failure reasons.
        t0 = time.monotonic()
        candidate, ml_err = self._ml.generate_candidate(cleaned)
        ml_latency_ms = (time.monotonic() - t0) * 1000

        if candidate is None:
            # The ML layer was attempted but produced nothing. Surface the
            # *classified* reason (HF 402/404/401, timeout, network, …)
            # instead of the old opaque "ml unavailable".
            return GrammarResult(
                sentence=rule_output,
                source="rule-based",
                reason="ml unavailable" + (f": {ml_err.category}" if ml_err else ""),
                ml_latency_ms=ml_latency_ms,
                ml_error=ml_err.message if ml_err else None,
                ml_error_category=ml_err.category if ml_err else None,
            )

        # Validator catches outright garbage (empty, scaffolding leak,
        # internal tokens, no-Turkish-no-overlap). The arbiter only
        # runs on outputs that already clear this sanity bar.
        v = validate_ml_output(candidate, cleaned)
        if not v.ok:
            logger.info("ML candidate rejected (%s): %r", v.reason, candidate)
            if not self.config.accept_fallback_on_validation_failure:
                return GrammarResult(
                    sentence=candidate.strip(),
                    source=f"ml:{self._ml_model_label}:unvalidated",
                    reason=f"validator failed but override on: {v.reason}",
                    ml_latency_ms=ml_latency_ms,
                )
            return GrammarResult(
                sentence=rule_output,
                source="rule-based",
                rejected_candidate=candidate,
                rejection_reason=v.reason,
                reason=f"validator rejected: {v.reason}",
                ml_latency_ms=ml_latency_ms,
            )

        # 3) Arbiter: quality comparison between the ML output and the
        #    deterministic rule-based baseline.
        ml_clean = candidate.strip()
        decision, arbiter_reason = _arbiter_decide(cleaned, ml_clean, rule_output)

        if decision == "ml":
            return GrammarResult(
                sentence=ml_clean,
                source=f"ml:{self._ml_model_label}",
                reason=arbiter_reason,
                ml_latency_ms=ml_latency_ms,
            )

        return GrammarResult(
            sentence=rule_output,
            source="rule-based",
            rejected_candidate=candidate,
            rejection_reason=arbiter_reason,
            reason=arbiter_reason,
            ml_latency_ms=ml_latency_ms,
        )


def list_available_models() -> List[ModelSpec]:
    """Convenience helper for tooling/UI surfaces."""
    return list(MODEL_REGISTRY.values())
