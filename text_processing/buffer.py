"""Word buffering with timing-based sequence completion."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class BufferConfig:
    debounce_seconds: float = 1.0
    silence_seconds: float = 2.5
    min_confidence: float = 0.0
    max_sequence_length: int = 32


@dataclass
class _BufferState:
    words: List[str] = field(default_factory=list)
    last_word: Optional[str] = None
    last_added_at: float = 0.0
    last_seen_at: float = 0.0


class WordBuffer:
    """Collects model word predictions and emits a completed sequence after silence.

    Thread-safe. Caller pushes words via `add(word, confidence=...)` and either
    polls `pop_if_complete()` periodically or registers `on_complete` to be
    invoked when a silence gap is detected (call `tick()` from a loop / timer).
    """

    def __init__(
        self,
        config: Optional[BufferConfig] = None,
        on_complete: Optional[Callable[[List[str]], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or BufferConfig()
        self.on_complete = on_complete
        self._clock = clock
        self._lock = threading.Lock()
        self._state = _BufferState()

    def add(self, word: str, confidence: float = 1.0, now: Optional[float] = None) -> bool:
        """Add a word. Returns True if it was accepted (not debounced or filtered)."""
        if not word or confidence < self.config.min_confidence:
            return False
        now = now if now is not None else self._clock()
        with self._lock:
            s = self._state
            s.last_seen_at = now
            if s.last_word == word and (now - s.last_added_at) < self.config.debounce_seconds:
                return False
            s.words.append(word)
            s.last_word = word
            s.last_added_at = now
            if len(s.words) >= self.config.max_sequence_length:
                completed = self._drain_locked()
            else:
                completed = None
        if completed is not None and self.on_complete:
            self.on_complete(completed)
        return True

    def tick(self, now: Optional[float] = None) -> Optional[List[str]]:
        """Check if the buffer has been silent long enough to complete a sequence.

        Returns the completed word list (also fires `on_complete`), or None.
        """
        now = now if now is not None else self._clock()
        with self._lock:
            if not self._state.words:
                return None
            if (now - self._state.last_added_at) < self.config.silence_seconds:
                return None
            completed = self._drain_locked()
        if self.on_complete:
            self.on_complete(completed)
        return completed

    def pop_if_complete(self, now: Optional[float] = None) -> Optional[List[str]]:
        return self.tick(now)

    def flush(self) -> List[str]:
        """Force-drain whatever is in the buffer right now."""
        with self._lock:
            return self._drain_locked()

    def peek(self) -> List[str]:
        with self._lock:
            return list(self._state.words)

    def __len__(self) -> int:
        with self._lock:
            return len(self._state.words)

    def _drain_locked(self) -> List[str]:
        words = self._state.words
        self._state = _BufferState()
        return words
