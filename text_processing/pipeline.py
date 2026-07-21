"""High-level orchestrator: words in, sentence + optional audio out."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .buffer import BufferConfig, WordBuffer
from .grammar import GrammarConfig, GrammarCorrector
from .tts import TTSConfig, TTSSynthesizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    words: List[str]
    sentence: str
    audio_path: Optional[Path] = None
    grammar_source: str = "rule-based"
    ml_latency_ms: float = 0.0
    rejected_candidate: Optional[str] = None
    rejection_reason: Optional[str] = None
    # Arbiter explanation for which engine produced ``sentence``.
    reason: str = ""
    # Action-oriented Turkish reason the ML layer failed/was unavailable
    # (+ stable category). ``None`` when ML succeeded or was never invoked.
    ml_error: Optional[str] = None
    ml_error_category: Optional[str] = None
    # TTS outcome: "ok" (audio written), "disabled" (synthesis skipped),
    # "empty" (no sentence to speak), or "failed" (gTTS error / offline).
    tts_status: str = "disabled"


@dataclass
class PipelineConfig:
    buffer: BufferConfig = field(default_factory=BufferConfig)
    grammar: GrammarConfig = field(default_factory=GrammarConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    synthesize_audio: bool = True


class SignTextPipeline:
    """Glue between the LSTM word stream and the TTS layer.

    Typical usage from `backend.py`:

        pipeline = SignTextPipeline(on_result=lambda r: ws.send_json(...))
        # for each model prediction above threshold:
        pipeline.feed(prediction["label_tr"], prediction["confidence"])
        # in a periodic task (e.g. asyncio loop or per-frame):
        pipeline.tick()
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_result: Optional[Callable[[PipelineResult], None]] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.on_result = on_result
        self.grammar = GrammarCorrector(self.config.grammar)
        # The synthesizer is always constructed (cheap — just ensures the
        # output dir exists), but whether it actually RUNS is a per-call
        # decision (see ``correct(synthesize_audio=...)``). This lets one
        # cached pipeline serve both audio and no-audio requests without
        # reloading the expensive grammar model, and guarantees gTTS never
        # fires when audio is turned off.
        self.tts = TTSSynthesizer(self.config.tts)
        self.buffer = WordBuffer(self.config.buffer, on_complete=self._handle_complete)

    def feed(self, word: str, confidence: float = 1.0) -> bool:
        return self.buffer.add(word, confidence=confidence)

    def tick(self) -> Optional[PipelineResult]:
        completed = self.buffer.tick()
        if completed is None:
            return None
        return self._build_result(completed)

    def flush(self) -> Optional[PipelineResult]:
        words = self.buffer.flush()
        if not words:
            return None
        return self._build_result(words)

    def correct(self, words: List[str], synthesize_audio: Optional[bool] = None) -> PipelineResult:
        """Correct ``words`` into a sentence (+ optional audio).

        ``synthesize_audio`` overrides ``config.synthesize_audio`` for this
        call only. Pass ``False`` to guarantee gTTS is never invoked.
        """
        return self._build_result(list(words), synthesize_audio=synthesize_audio)

    def _handle_complete(self, words: List[str]) -> None:
        result = self._build_result(words)
        if self.on_result:
            try:
                self.on_result(result)
            except Exception as e:
                logger.warning("on_result callback raised: %s", e)

    def _build_result(
        self, words: List[str], synthesize_audio: Optional[bool] = None
    ) -> PipelineResult:
        result = self.grammar.correct_detailed(words)
        synthesize = self.config.synthesize_audio if synthesize_audio is None else synthesize_audio
        audio_path = None
        if not synthesize:
            tts_status = "disabled"
        elif not result.sentence:
            tts_status = "empty"
        else:
            audio_path = self.tts.synthesize_to_file(result.sentence)
            tts_status = "ok" if audio_path is not None else "failed"
        return PipelineResult(
            words=words,
            sentence=result.sentence,
            audio_path=audio_path,
            grammar_source=result.source,
            ml_latency_ms=result.ml_latency_ms,
            rejected_candidate=result.rejected_candidate,
            rejection_reason=result.rejection_reason,
            reason=result.reason,
            ml_error=result.ml_error,
            ml_error_category=result.ml_error_category,
            tts_status=tts_status,
        )
