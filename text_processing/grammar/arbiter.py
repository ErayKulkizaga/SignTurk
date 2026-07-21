"""Smart arbiter: per-request quality comparator (ML vs rule-based)."""

from __future__ import annotations

from typing import List, Optional

from .linguistics import (
    _IRREGULAR_VERB_STEMS,
    _QUESTION_PARTICLE_VARIANTS,
    apply_consonant_softening,
    normalize_words,
)

# ───────────────────────── Smart arbiter ─────────────────────────
#
# The arbiter is a quality comparator that decides — per request —
# whether to surface the ML candidate or the deterministic rule-based
# output. It runs *only* for complex inputs; trivial 1-2 token streams
# skip the ML round-trip entirely (the rule-based parser is 100%
# correct and instantaneous for those).

# Inputs at or below this size are fast-tracked: covers verb-only
# ("gitmek"), pronoun+verb ("sen gelmek"), and noun+verb ("su istemek")
# SOV pairs.
_ARBITER_FAST_TRACK_MAX_TOKENS = 2

# Inputs at or below this size enforce the strict (1.0) root-preservation
# floor: short streams carry no redundancy, so every input lemma must
# survive into the ML output.
_ARBITER_STRICT_PRESERVATION_MAX_TOKENS = 3

# Minimum fraction of input lemmas that must appear in the ML output for
# *longer* inputs. The relaxed floor lets advanced LLMs drop genuinely
# redundant tokens (e.g. ``kafe kapanmak zaman güzel hava`` →
# ``Kafe kapanıyor güzel havada.`` may legitimately drop ``zaman``).
# Short inputs use the strict 1.0 floor instead — see ``_arbiter_decide``.
_ARBITER_MIN_ROOT_PRESERVATION = 0.50

# Critical markers — tokens whose semantics the ML layer routinely
# paraphrases away (question particle, negation, numeric quantity).
# Their presence forces the universal fast-track to the rule-based
# engine regardless of input length.
_CRITICAL_NUMERIC_TOKENS = frozenset(
    {
        "sıfır",
        "bir",
        "iki",
        "üç",
        "dört",
        "beş",
        "altı",
        "yedi",
        "sekiz",
        "dokuz",
        "on",
        "yirmi",
        "otuz",
        "kırk",
        "elli",
        "altmış",
        "yetmiş",
        "seksen",
        "doksan",
        "yüz",
        "bin",
        "milyon",
    }
)
# Keep the ML fast-track in sync with the full interrogative registry so
# 1st-person forms (mıyım, müyüz, …) also bypass ML — otherwise their
# question semantics can be paraphrased away.
_CRITICAL_QUESTION_TOKENS = _QUESTION_PARTICLE_VARIANTS
_CRITICAL_NEGATION_TOKENS = frozenset({"değil", "yok", "hayır"})

# Anti-hallucination — intent/necessity vocabulary the ML model
# fabricates when the input has no supporting lemma. ``istiyorsun``
# from a bare ``sen gelmek mi`` is the canonical failure mode.
_HALLUCINATED_INTENT_PREFIXES = ("isti", "gerek")
_HALLUCINATED_INTENT_EXACT = frozenset({"lazım"})
_INPUT_INTENT_LEMMAS = ("iste", "gerek")


def _is_simple_input(words: List[str]) -> bool:
    """True for trivially simple token streams the arbiter fast-tracks
    to the rule-based engine without ever calling the ML."""
    return len(words) <= _ARBITER_FAST_TRACK_MAX_TOKENS


def _has_critical_marker(words: List[str]) -> bool:
    """True if any input token is a question particle, negation, or
    numeric quantity. Inputs with critical markers bypass the ML
    entirely — the rule-based engine preserves them deterministically.
    """
    for raw in words:
        if not raw:
            continue
        token = raw.strip().lower()
        if not token:
            continue
        if token.isdigit():
            return True
        if token in _CRITICAL_NUMERIC_TOKENS:
            return True
        if token in _CRITICAL_QUESTION_TOKENS:
            return True
        if token in _CRITICAL_NEGATION_TOKENS:
            return True
    return False


def _input_signals_intent(input_words: List[str]) -> bool:
    """True if the user's tokens already contain an intent/need lemma
    (``iste`` / ``gerek``). When present, intent vocabulary in the ML
    output is legitimate and the anti-hallucination guard stands down.
    """
    for w in input_words:
        if not w:
            continue
        low = w.lower()
        if any(lemma in low for lemma in _INPUT_INTENT_LEMMAS):
            return True
    return False


def _ml_contains_hallucinated_intent(ml_output: str) -> bool:
    """True if the ML output contains an intent/necessity word — checked
    against forbidden prefixes (``isti*``, ``gerek*``) and the exact
    word ``lazım``. Strips trailing punctuation so ``lazım.`` still
    matches.
    """
    for raw in ml_output.split():
        token = raw.strip(".,!?;:\"'()[]").lower()
        if not token:
            continue
        if token in _HALLUCINATED_INTENT_EXACT:
            return True
        if any(token.startswith(prefix) for prefix in _HALLUCINATED_INTENT_PREFIXES):
            return True
    return False


def _lemma_forms(word: str) -> List[str]:
    """Substring patterns any inflected form of ``word`` should contain.

    For verbs: the bare stem (``git`` from ``gitmek``), the irregular
    pre-softened stem (``gid``), and a shorter prefix for stems whose
    final a/e is consumed by narrowing before ``-yor`` (``söyle`` →
    ``söyl``, ``anla`` → ``anl``).

    For nouns: the lemma itself, plus the softened root for stems whose
    final p/ç/t/k voices before a vowel suffix (``kitap`` → ``kitab``).
    """
    word = word.strip().lower()
    if not word:
        return []
    if word.endswith(("mek", "mak")):
        stem = word[:-3]
        forms = [stem] if stem else [word]
        if stem and stem[-1] in "ae":
            forms.append(stem[:-1])
        irreg = _IRREGULAR_VERB_STEMS.get(word)
        if irreg:
            forms.append(irreg)
        return forms
    forms = [word]
    softened = apply_consonant_softening(word)
    if softened != word:
        forms.append(softened)
    return forms


def _root_preservation_score(input_words: List[str], output: str) -> float:
    """Fraction of input lemmas whose stem (or a known inflected form)
    appears in ``output``. Returns 1.0 for empty input.
    """
    normalized = normalize_words(input_words)
    if not normalized:
        return 1.0
    output_lower = output.lower()
    matches = 0
    for w in normalized:
        if any(form and form in output_lower for form in _lemma_forms(w)):
            matches += 1
    return matches / len(normalized)


def _input_verbs(input_words: List[str]) -> List[str]:
    """Input glosses that are verbs — infinitive forms ending in
    ``-mek``/``-mak``. These are the sentence's main verbs; sign-language
    translation must preserve the action even when redundant nouns drop.
    """
    return [w for w in normalize_words(input_words) if w.endswith(("mek", "mak"))]


def _ml_dropped_main_verb(input_words: List[str], ml_output: str) -> Optional[str]:
    """First input verb whose stem (or any inflected form) is absent from
    ``ml_output``, or ``None`` when every input verb survives.

    Reuses :func:`_lemma_forms` so the same stem derivation that drives
    root preservation governs the verb guard: irregular stems (``gitmek``
    → ``gid``) and ``-yor`` narrowing (``söylemek`` → ``söyl``) are
    honoured, so a valid conjugation (``gördüm``, ``gidiyorum``,
    ``söylüyorum``) is never flagged — only an actually dropped or
    substituted verb (``görmek`` → ``bakmışım``) is.
    """
    output_lower = ml_output.lower()
    for verb in _input_verbs(input_words):
        forms = _lemma_forms(verb)
        if not any(form and form in output_lower for form in forms):
            return verb
    return None


def _ends_with_unconjugated_verb(text: str) -> bool:
    """True if the last orthographic word looks like an infinitive
    (ends in ``-mek``/``-mak``). Detects the "model failed to conjugate
    the verb" failure mode that produced ``Sen beni sevmek.``.
    """
    cleaned = text.strip().rstrip(".,!?;:").strip()
    if not cleaned:
        return False
    last_word = cleaned.split()[-1].lower()
    return last_word.endswith("mek") or last_word.endswith("mak")

def _arbiter_decide(
    input_words: List[str],
    ml_output: str,
    rule_output: str,
) -> tuple:
    """Pick the better candidate between the ML output and the
    deterministic rule-based output.

    Returns ``("ml"|"rule", reason)``. The rule-based output is
    treated as the safe floor: ML wins only when it passes every
    sanity heuristic (length, no-mastar, root preservation).
    """
    rule_len = len(rule_output)
    ml_len = len(ml_output)

    # Length sanity — compared against the rule-based baseline, which
    # is always a valid Turkish sentence for the same input.
    if rule_len and ml_len < rule_len * 0.5:
        return "rule", (f"ml output too short ({ml_len} chars vs rule-based {rule_len})")
    if rule_len and ml_len > rule_len * 4:
        return "rule", (f"ml output too long ({ml_len} chars vs rule-based {rule_len})")

    # No-mastar: the system prompt explicitly forbids leaving verbs in
    # infinitive form. If the last word is still ``-mek``/``-mak``,
    # the model didn't follow instructions — defer to rule-based.
    if _ends_with_unconjugated_verb(ml_output):
        return "rule", "ml output ends with unconjugated infinitive (-mek/-mak)"

    # Anti-hallucination — reject intent/necessity verbs the user never
    # supplied. Catches the ``Sen gelmek istiyorsun.`` failure mode
    # from a bare ``sen gelmek mi`` input.
    if _ml_contains_hallucinated_intent(ml_output) and not _input_signals_intent(input_words):
        return "rule", "arbiter rejected: hallucinated intent verb"

    # Main-verb preservation — independent of the root-preservation
    # percentage below. In sign-language translation the main verb carries
    # the core action, so it must survive even when enough *other* roots
    # remain to clear the relaxed percentage floor. If the ML output drops
    # or substitutes any input verb (``görmek`` → ``bakmışım``), defer to
    # rule-based; valid conjugations (``gördüm``) pass via ``_lemma_forms``.
    dropped_verb = _ml_dropped_main_verb(input_words, ml_output)
    if dropped_verb is not None:
        return "rule", f"ml dropped/substituted main verb ({dropped_verb})"

    # Root preservation — every input lemma should still be detectable
    # in the ML output (in some inflected form). Missing roots = the
    # model paraphrased away the user's content. Short inputs enforce
    # the strict 1.0 floor; longer inputs tolerate the relaxed default.
    required = (
        1.0
        if len(input_words) <= _ARBITER_STRICT_PRESERVATION_MAX_TOKENS
        else _ARBITER_MIN_ROOT_PRESERVATION
    )
    preservation = _root_preservation_score(input_words, ml_output)
    if preservation < required:
        return "rule", (f"ml dropped key roots (preservation={preservation:.2f} < {required:.2f})")

    return "ml", f"passed arbiter checks (root preservation={preservation:.2f})"
