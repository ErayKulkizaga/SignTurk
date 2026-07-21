"""Bounded, TTL-aware store of async correction jobs + the worker pool."""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .schemas import CorrectResponse

_JOB_MAX_SIZE = int(os.environ.get("SIGNAI_JOB_MAX", "64"))
_JOB_TTL_SECONDS = float(os.environ.get("SIGNAI_JOB_TTL", "300"))
_ASYNC_WORKERS = int(os.environ.get("SIGNAI_ASYNC_WORKERS", "2"))


@dataclass
class _Job:
    status: str = "pending"  # pending | done | error
    result: Optional[CorrectResponse] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    # Set when the job leaves "pending". TTL is measured from here so a slow
    # worker's result can't be evicted out from under a still-running job.
    finished_at: Optional[float] = None


class JobStore:
    """Bounded, TTL-aware store of async correction jobs (thread-safe).

    Bounded so a flood of submissions can't grow memory unboundedly; results
    expire after a TTL since clients are expected to poll promptly.
    """

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.max_size = max(1, max_size)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._jobs: "OrderedDict[str, _Job]" = OrderedDict()
        self._lock = threading.Lock()

    def create(self) -> str:
        """Register a new pending job, evicting only FINISHED jobs to make room.

        Raises ``RuntimeError`` when the store is full of still-pending jobs —
        the caller maps that to HTTP 429 rather than silently evicting work in
        flight (whose ``set_done``/``set_error`` would otherwise be dropped).
        """
        job_id = uuid.uuid4().hex
        with self._lock:
            self._evict_expired_locked(time.monotonic())
            while len(self._jobs) >= self.max_size:
                if not self._evict_oldest_finished_locked():
                    raise RuntimeError("job store saturated")
            self._jobs[job_id] = _Job()
        return job_id

    def _evict_oldest_finished_locked(self) -> bool:
        for k, j in self._jobs.items():  # OrderedDict → oldest first
            if j.status != "pending":
                self._jobs.pop(k, None)
                return True
        return False

    def set_done(self, job_id: str, result: CorrectResponse) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status, job.result = "done", result
                job.finished_at = time.monotonic()

    def set_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status, job.error = "error", error
                job.finished_at = time.monotonic()

    def get(self, job_id: str) -> Optional[_Job]:
        with self._lock:
            self._evict_expired_locked(time.monotonic())
            return self._jobs.get(job_id)

    def _evict_expired_locked(self, now: float) -> None:
        # TTL applies only to FINISHED jobs (timed from completion). Pending
        # jobs are never TTL-evicted, so a slow ML request can't 404 mid-flight.
        if not self.ttl_seconds:
            return
        stale = [
            k
            for k, j in self._jobs.items()
            if j.finished_at is not None and now - j.finished_at > self.ttl_seconds
        ]
        for k in stale:
            self._jobs.pop(k, None)

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "pending")


_JOB_STORE = JobStore(_JOB_MAX_SIZE, _JOB_TTL_SECONDS)
_EXECUTOR = ThreadPoolExecutor(max_workers=_ASYNC_WORKERS, thread_name_prefix="text-async")
