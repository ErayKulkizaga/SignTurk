"""Sanity validation of ML candidate sentences before they are trusted."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .linguistics import _TR_CHARS, normalize_words
from .prompts import _FEW_SHOT_EXAMPLES

# ───────────────────────── Output validation ─────────────────────────


_INTERNAL_TOKEN_PATTERN = re.compile(r"<extra_id_\d+>|</?s>|<pad>|<unk>")

# Prompt-template scaffolding the model should never echo back. When a
# small model gets confused it tends to parrot the few-shot block
# verbatim, including these labels — a strong "this is a leak" signal.
_SCAFFOLDING_PATTERN = re.compile(
    r"(?i)\b(?:girdi|çıktı|cikti|kelimeler|cümle|cumle|görev|gorev)\s*:"
)


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""


def _has_few_shot_leak(candidate_lower: str, user_input_lower: str) -> bool:
    """True if the candidate parrots a few-shot example the user didn't ask for.

    Compares both the input and the output side of every few-shot pair
    against the candidate. A match counts as a leak only when the user's
    actual input differs from that example — otherwise the candidate
    legitimately matches the demonstration.
    """
    for shot_inp, shot_out in _FEW_SHOT_EXAMPLES:
        shot_inp_norm = " ".join(normalize_words(shot_inp.split()))
        shot_out_norm = " ".join(normalize_words(shot_out.split()))
        if shot_inp_norm and shot_inp_norm in candidate_lower and shot_inp_norm != user_input_lower:
            return True
        # The model leaking a *different* example's output is also a
        # parrot — but allow it if the user's input matches that shot.
        if shot_out_norm and shot_out_norm in candidate_lower and shot_inp_norm != user_input_lower:
            return True
    return False


def validate_ml_output(candidate: str, input_words: List[str]) -> ValidationResult:
    if not candidate or not candidate.strip():
        return ValidationResult(False, "empty")
    text = candidate.strip()
    if _INTERNAL_TOKEN_PATTERN.search(text):
        return ValidationResult(False, "contains model-internal tokens")
    if _SCAFFOLDING_PATTERN.search(text):
        return ValidationResult(False, "contains prompt scaffolding tokens")
    if len(text) < 3:
        return ValidationResult(False, "too short")
    normalized_input = normalize_words(input_words)
    input_chars = sum(len(w) for w in input_words) + max(0, len(input_words) - 1)
    if len(text) > max(120, input_chars * 6):
        return ValidationResult(False, "too long vs input")
    lower = text.lower()
    user_input_lower = " ".join(normalized_input)
    if _has_few_shot_leak(lower, user_input_lower):
        return ValidationResult(False, "leaks verbatim few-shot example")
    has_tr_char = any(ch in _TR_CHARS for ch in text)
    has_input_overlap = any(w in lower for w in normalized_input if len(w) > 2)
    if not (has_tr_char or has_input_overlap):
        return ValidationResult(False, "no Turkish chars and no input overlap")
    return ValidationResult(True)
