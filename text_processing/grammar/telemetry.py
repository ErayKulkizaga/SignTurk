"""In-process decision log — telemetry behind the health endpoint."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

# ───────────────────────── Decision logging ─────────────────────────


@dataclass
class DecisionRecord:
    """One grammar decision, captured for telemetry and the health endpoint."""

    input_words: List[str]
    model_key: str
    prompt_version: str
    source: str
    final_sentence: str
    ml_latency_ms: float
    reason: str
    rejected_candidate: Optional[str] = None
    rejection_reason: Optional[str] = None
    ml_error_category: Optional[str] = None


class _DecisionLog:
    """Thread-safe ring buffer of the most recent grammar decisions.

    Lets ``/api/text/health`` report what the LLM layer has been doing
    (ml-vs-rule split, recent rejection reasons, average latency) without
    wiring a database in. Bounded — old records fall off the end.
    """

    def __init__(self, maxlen: int = 200) -> None:
        self._dq: Deque[DecisionRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, rec: DecisionRecord) -> None:
        with self._lock:
            self._dq.append(rec)

    def recent(self, n: Optional[int] = None) -> List[DecisionRecord]:
        with self._lock:
            items = list(self._dq)
        return items if n is None else items[-n:]

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()

    def summary(self) -> Dict[str, object]:
        with self._lock:
            items = list(self._dq)
        ml = sum(1 for r in items if r.source.startswith("ml:"))
        rule = sum(1 for r in items if not r.source.startswith("ml:"))
        rejections = [r.rejection_reason for r in items if r.rejection_reason]
        latencies = [r.ml_latency_ms for r in items if r.ml_latency_ms]
        ml_errors = [r.ml_error_category for r in items if r.ml_error_category]
        return {
            "window": len(items),
            "ml_chosen": ml,
            "rule_chosen": rule,
            "rejections": len(rejections),
            "last_rejection_reasons": rejections[-5:],
            "avg_ml_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "ml_errors": len(ml_errors),
            "last_ml_error_categories": ml_errors[-5:],
        }


# Process-wide log of recent decisions, surfaced via the health endpoint.
DECISION_LOG = _DecisionLog()
