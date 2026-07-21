"""Bounded, TTL-aware LRU cache of text pipelines, keyed by (use_ml, model_key)."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from text_processing import SignTextPipeline

logger = logging.getLogger(__name__)

# Cache bounds. Each cached pipeline may hold a multi-hundred-MB local model
# in RAM, so the cache is capped and entries expire after inactivity instead
# of living for the whole process lifetime. Overridable via env.
_CACHE_MAX_SIZE = int(os.environ.get("SIGNAI_PIPELINE_CACHE_MAX", "3"))
_CACHE_TTL_SECONDS = float(os.environ.get("SIGNAI_PIPELINE_CACHE_TTL", "1800"))  # 0 = no TTL

CacheKey = Tuple[bool, str]


@dataclass
class _CacheEntry:
    pipeline: SignTextPipeline
    created_at: float
    last_used_at: float


class PipelineCache:
    """Bounded, TTL-aware LRU cache of pipelines keyed by (use_ml, model_key).

    Replaces the previous unbounded dict: every distinct model combination
    used to load a model and keep it resident forever. This enforces a max
    entry count (LRU eviction) and an idle TTL, and exposes manual unload so
    operators can free a stuck/large model without restarting the server.
    """

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.max_size = max(1, max_size)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._entries: "OrderedDict[CacheKey, _CacheEntry]" = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(
        self, key: CacheKey, factory: Callable[[], SignTextPipeline]
    ) -> SignTextPipeline:
        now = time.monotonic()
        with self._lock:
            self._evict_expired_locked(now)
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_used_at = now
                self._entries.move_to_end(key)
                return entry.pipeline
            # Build on miss (inside the lock, mirroring the old double-checked
            # behavior) so two concurrent requests can't load the same model
            # twice. factory() may raise (e.g. unknown model_key) — propagate
            # without caching.
            pipeline = factory()
            self._entries[key] = _CacheEntry(pipeline, now, now)
            self._evict_overflow_locked()
            return pipeline

    def _evict_expired_locked(self, now: float) -> None:
        if not self.ttl_seconds:
            return
        stale = [k for k, e in self._entries.items() if now - e.last_used_at > self.ttl_seconds]
        for k in stale:
            self._entries.pop(k, None)
            logger.info("pipeline cache: evicted %s (idle TTL)", k)

    def _evict_overflow_locked(self) -> None:
        while len(self._entries) > self.max_size:
            k, _ = self._entries.popitem(last=False)  # LRU = oldest used
            logger.info("pipeline cache: evicted %s (max_size)", k)

    def unload(self, key: CacheKey) -> bool:
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            return n

    def snapshot(self) -> List[Dict[str, object]]:
        now = time.monotonic()
        with self._lock:
            # Drop idle-expired entries first so /health reports live state,
            # not pipelines that have actually timed out.
            self._evict_expired_locked(now)
            items = list(self._entries.items())
        out: List[Dict[str, object]] = []
        for (use_ml, model_key), entry in items:
            info: Dict[str, object] = {
                "use_ml": use_ml,
                "model_key": model_key,
                "idle_seconds": round(now - entry.last_used_at, 1),
                "age_seconds": round(now - entry.created_at, 1),
            }
            info.update(entry.pipeline.grammar.ml_status)
            out.append(info)
        return out


_PIPELINE_CACHE = PipelineCache(_CACHE_MAX_SIZE, _CACHE_TTL_SECONDS)
