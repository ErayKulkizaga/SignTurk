"""Turkish grammar correction: rule-based engine + multi-model HF hybrid layer.

Hybrid strategy:
    1. Optional ML model (HF transformers locally, or HF Inference API)
       produces a candidate sentence.
    2. The candidate is run through a sanity validator (empty, too long,
       too short, non-Turkish, echoing the input, model-internal tokens).
    3. On rejection — or any load/inference error / timeout — the
       deterministic rule-based corrector is used.

This package was split out of a single ~2000-line ``grammar.py`` god-module
into cohesive layers so each concern can be read, tested, and changed in
isolation:

    linguistics  pure Turkish phonology/morphology (no I/O)
    registry     model catalog + GrammarConfig
    prompts      versioned zero-shot prompt templates
    validation   ML-candidate sanity checks
    rules        the deterministic rule-based corrector (primary engine)
    adapters     model execution backends + ML error classification
    arbiter      per-request ML-vs-rule quality comparator
    telemetry    in-process decision log
    corrector    the public hybrid facade

The public API is unchanged: every name previously importable from
``text_processing.grammar`` is re-exported here, so external imports keep
working.
"""

from __future__ import annotations

from .adapters import MLError, _classify_ml_error, _InferenceAPIAdapter
from .arbiter import (
    _CRITICAL_NEGATION_TOKENS,
    _CRITICAL_QUESTION_TOKENS,
    _ends_with_unconjugated_verb,
    _has_critical_marker,
    _input_signals_intent,
    _is_simple_input,
    _lemma_forms,
    _ml_contains_hallucinated_intent,
    _root_preservation_score,
)
from .corrector import (
    GrammarCorrector,
    GrammarResult,
    MLGrammarCorrector,
    list_available_models,
)
from .linguistics import (
    apply_ablative,
    apply_accusative,
    apply_consonant_assimilation,
    apply_consonant_softening,
    apply_dative,
    apply_dative_suffix,
    apply_locative,
    apply_vowel_harmony,
    apply_vowel_narrowing,
    build_question_particle,
    conjugate_present_continuous,
    negate_present_continuous,
    normalize_words,
    turkish_lower,
)
from .prompts import (
    PROMPT_TEMPLATES,
    PromptSpec,
    build_causal_prompt,
    build_seq2seq_prompt,
    build_zero_shot_messages,
    build_zero_shot_prompt,
    list_prompt_versions,
)
from .registry import (
    DEFAULT_API_MODEL_KEY,
    DEFAULT_LOCAL_MODEL_KEY,
    DEFAULT_MODEL_KEY,
    MODEL_REGISTRY,
    PROMPT_VERSION,
    GrammarConfig,
    ModelSpec,
)
from .rules import RuleBasedCorrector
from .telemetry import DECISION_LOG, DecisionRecord
from .validation import ValidationResult, validate_ml_output

__all__ = [
    # ── Linguistics (domain) ──
    "apply_ablative",
    "apply_accusative",
    "apply_consonant_assimilation",
    "apply_consonant_softening",
    "apply_dative",
    "apply_dative_suffix",
    "apply_locative",
    "apply_vowel_harmony",
    "apply_vowel_narrowing",
    "build_question_particle",
    "conjugate_present_continuous",
    "negate_present_continuous",
    "normalize_words",
    "turkish_lower",
    # ── Registry / config ──
    "DEFAULT_API_MODEL_KEY",
    "DEFAULT_LOCAL_MODEL_KEY",
    "DEFAULT_MODEL_KEY",
    "MODEL_REGISTRY",
    "PROMPT_VERSION",
    "GrammarConfig",
    "ModelSpec",
    # ── Prompts ──
    "PROMPT_TEMPLATES",
    "PromptSpec",
    "build_causal_prompt",
    "build_seq2seq_prompt",
    "build_zero_shot_messages",
    "build_zero_shot_prompt",
    "list_prompt_versions",
    # ── Validation ──
    "ValidationResult",
    "validate_ml_output",
    # ── Rule-based engine (primary) ──
    "RuleBasedCorrector",
    # ── ML adapters / error model ──
    "MLError",
    # ── Telemetry ──
    "DECISION_LOG",
    "DecisionRecord",
    # ── Hybrid facade (public) ──
    "GrammarCorrector",
    "GrammarResult",
    "MLGrammarCorrector",
    "list_available_models",
]
