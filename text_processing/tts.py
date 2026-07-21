"""Turkish text-to-speech with a pluggable engine.

Two backends behind one ``TTSSynthesizer``:

  * **gtts**  — Google TTS (mp3). Needs network; returns nothing offline.
  * **piper** — Piper neural TTS (wav). Fully offline, but requires the
    optional ``piper-tts`` package and a Turkish voice model
    (``$SIGNAI_PIPER_VOICE`` or ``TTSConfig.piper_voice``).

The engine is selected by ``TTSConfig.engine`` (``auto`` tries gTTS then
Piper). ``auto`` + ``$SIGNAI_TTS_OFFLINE=1`` (or ``$SIGNAI_TTS_ENGINE=piper``)
flips the order so a host with no internet still speaks. Synthesis failures
return ``None`` (callers surface this as ``tts_status="failed"``).
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import sys
import threading
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)


class _Audio(NamedTuple):
    """A synthesized clip plus the format produced by the engine."""

    data: bytes
    ext: str  # "mp3" | "wav"
    mime: str  # "audio/mpeg" | "audio/wav"


def _default_engine() -> str:
    """Resolve the default TTS engine from the environment.

    ``$SIGNAI_TTS_ENGINE`` wins; ``$SIGNAI_TTS_OFFLINE=1`` forces Piper;
    otherwise ``auto`` (gTTS first, Piper fallback).
    """
    engine = os.environ.get("SIGNAI_TTS_ENGINE")
    if engine:
        return engine.strip().lower()
    if os.environ.get("SIGNAI_TTS_OFFLINE") == "1":
        return "piper"
    return "auto"


def gtts_installed() -> bool:
    """True if the gTTS engine is importable. Cheap, no network call."""
    import importlib.util

    return importlib.util.find_spec("gtts") is not None


def piper_installed() -> bool:
    """True if the Piper offline engine is importable."""
    import importlib.util

    return importlib.util.find_spec("piper") is not None


def tts_engine_status() -> dict:
    """Engine availability snapshot for the health endpoint.

    Distinguishes "engine missing" from "engine present but can't reach the
    network" (gTTS) so operators can tell why audio is or isn't produced.
    """
    return {
        "configured": _default_engine(),
        "gtts_installed": gtts_installed(),
        "piper_installed": piper_installed(),
        "piper_voice_configured": bool(os.environ.get("SIGNAI_PIPER_VOICE")),
    }


@dataclass
class TTSConfig:
    language: str = "tr"
    slow: bool = False
    tld: str = "com.tr"
    output_dir: Path = Path("uploads") / "tts"
    # "auto" | "gtts" | "piper" — defaults from the environment.
    engine: str = field(default_factory=_default_engine)
    # Path to a Piper ``.onnx`` voice. Falls back to $SIGNAI_PIPER_VOICE.
    piper_voice: Optional[str] = None


class TTSSynthesizer:
    """Synthesizes speech via the configured engine(s).

    Keeps the rest of the pipeline from importing any TTS library directly.
    ``synthesize_to_file`` / ``synthesize_to_bytes`` are the stable public API;
    engine selection and audio format are internal details.
    """

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self.config = config or TTSConfig()
        # The output dir is created lazily (in ``synthesize_to_file``), so a
        # no-audio pipeline never needs write access to it.
        self._lock = threading.Lock()
        self._piper_voice = None  # lazily loaded PiperVoice
        self._piper_voice_path: Optional[str] = None

    # ── Engine order ──────────────────────────────────────────
    def _engine_order(self) -> list:
        engine = (self.config.engine or "auto").lower()
        if engine == "gtts":
            return [self._synth_gtts]
        if engine == "piper":
            return [self._synth_piper]
        if engine == "auto":
            return [self._synth_gtts, self._synth_piper]
        # Unknown value → behave like auto but warn once.
        logger.warning("Unknown TTS engine %r; falling back to auto.", engine)
        return [self._synth_gtts, self._synth_piper]

    def _synthesize(self, text: str) -> Optional[_Audio]:
        if not text or not text.strip():
            return None
        for backend in self._engine_order():
            audio = backend(text)
            if audio is not None:
                return audio
        return None

    # ── Backends ──────────────────────────────────────────────
    def _synth_gtts(self, text: str) -> Optional[_Audio]:
        try:
            from gtts import gTTS
        except ImportError:
            logger.info("gTTS not installed; skipping gtts backend.")
            return None
        try:
            tts = gTTS(
                text=text, lang=self.config.language, slow=self.config.slow, tld=self.config.tld
            )
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            return _Audio(buf.getvalue(), "mp3", "audio/mpeg")
        except Exception as e:
            logger.warning("gTTS synthesis failed (offline?): %s", e)
            return None

    def _load_piper_voice(self):
        voice_path = self.config.piper_voice or os.environ.get("SIGNAI_PIPER_VOICE")
        if not voice_path:
            logger.info("No Piper voice configured (SIGNAI_PIPER_VOICE); skipping piper backend.")
            return None
        if self._piper_voice is not None and self._piper_voice_path == voice_path:
            return self._piper_voice
        try:
            from piper import PiperVoice
        except ImportError:
            logger.info("piper-tts not installed; skipping piper backend.")
            return None
        try:
            self._piper_voice = PiperVoice.load(voice_path)
            self._piper_voice_path = voice_path
            return self._piper_voice
        except Exception as e:
            logger.warning("Piper voice load failed (%s): %s", voice_path, e)
            return None

    def _synth_piper(self, text: str) -> Optional[_Audio]:
        voice = self._load_piper_voice()
        if voice is None:
            return None
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                # piper-tts writes a complete WAV (sets channels/rate/width).
                voice.synthesize(text, wav_file)
            data = buf.getvalue()
            return _Audio(data, "wav", "audio/wav") if data else None
        except Exception as e:
            logger.warning("Piper synthesis failed: %s", e)
            return None

    # ── Public API ────────────────────────────────────────────
    def synthesize_to_bytes(self, text: str) -> Optional[bytes]:
        audio = self._synthesize(text)
        return audio.data if audio is not None else None

    def synthesize_to_file(self, text: str, filename: Optional[str] = None) -> Optional[Path]:
        audio = self._synthesize(text)
        if audio is None:
            return None
        # Reduce a caller-supplied name to a bare basename so a value like
        # "../../etc/passwd" can't escape output_dir (arbitrary file write).
        raw_name = filename or f"tts_{uuid.uuid4().hex[:12]}.{audio.ext}"
        name = Path(raw_name).name
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / name
        with self._lock:
            path.write_bytes(audio.data)
        return path


def play_audio(path: Path) -> bool:
    """Best-effort local playback for dev/test only. Returns True if a player was invoked."""
    p = Path(path)
    if not p.exists():
        logger.warning("Audio file not found: %s", p)
        return False
    candidates = []
    if sys.platform == "darwin":
        candidates = [["afplay", str(p)]]
    elif sys.platform.startswith("win"):
        candidates = [["cmd", "/c", "start", "", str(p)]]
    else:
        candidates = [
            ["mpg123", "-q", str(p)],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(p)],
            ["mpv", "--really-quiet", str(p)],
            ["xdg-open", str(p)],
        ]
    for cmd in candidates:
        if shutil.which(cmd[0]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                logger.warning("Player %s failed: %s", cmd[0], e)
                continue
    logger.info("No audio player available; serve %s via your UI instead.", p)
    return False
