"""Deterministic rule-based Turkish corrector — the reliable primary engine.

The safe floor for the hybrid corrector and the fallback for every ML error
path. Kept intact; only relocated out of the former ``grammar.py`` module.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

from .linguistics import (
    _CASE_GOVERNORS,
    _EXISTENTIAL_NEGATION_AGREEMENT,
    _EXISTENTIAL_NEGATION_TOKENS,
    _GREETINGS,
    _NEGATION_COPULA_AGREEMENT,
    _NEGATION_COPULA_TOKENS,
    _PRONOUN_CASE_FORMS,
    _PRONOUN_PERSON,
    _PRONOUNS,
    _QUESTION_PARTICLE_VARIANTS,
    _STANDALONE_NEGATIVE_TOKENS,
    apply_ablative,
    apply_accusative,
    apply_dative,
    apply_locative,
    build_question_particle,
    conjugate_present_continuous,
    negate_present_continuous,
    normalize_words,
)

# ───────────────────────── Rule-based engine ─────────────────────────


class RuleBasedCorrector:
    """Generalized SOV parser for Turkish sign-language word streams.

    Instead of a token-by-token state machine, ``correct()`` splits the
    input into three syntactic roles and applies the morphology engine
    role-by-role:

        [Greeting]* [Subject] [Object/Adverb]* [Verb]

    * **Verb (Yüklem)** — the *rightmost* token shaped like an
      infinitive (``-mek``/``-mak``). Scanning instead of trusting the
      last position lets us handle inverted/devrik input streams
      where the verb arrives first or in the middle (sign streams
      don't always reorder to canonical SOV). The verb is removed
      from the token list and reattached at the end, conjugated.
    * **Subject (Özne)** — the first *remaining* (non-verb,
      non-greeting) token. If it's a pronoun the verb's person
      agreement is read off it directly; otherwise the subject is
      treated as a 3rd-person singular noun. With no remaining
      tokens (verb-only input) the conjugation falls back to
      1st-person ("ben"), which reads naturally for sign streams.
    * **Objects / adverbs** — the remaining tokens, in their
      original input order. All of them receive the case the verb
      governs (dative for motion, locative for stative, ablative for
      ablation, accusative for transitive). Verbs outside
      ``_CASE_GOVERNORS`` leave their middle tokens bare (yalın
      hâl / belirtisiz nesne).

    The phonology engine (harmony, softening, assimilation, narrowing)
    powers every inflection — nothing in this method is hardcoded to a
    specific word.
    """

    class _Classification(NamedTuple):
        """Strict two-vector decomposition of the normalized input.

        ``semantic_payload`` holds *only* the nouns, pronouns, and
        verbs that drive SOV parsing and case government. Functional
        markers (question particle, ``değil``, ``yok``, ``hayır``) are
        consumed into the boolean flags below — they are guaranteed
        never to reach ``_apply_case``. ``leading_interjections``
        carries greetings + ``hayır`` because both ride the same
        comma-after-interjection punctuation rule.

        Marker flags are idempotent: duplicate markers (``["değil",
        "değil"]``) or co-occurring markers (``["hayır", ..., "değil"]``)
        resolve cleanly to a single negation state.
        """

        semantic_payload: Tuple[str, ...]
        leading_interjections: Tuple[str, ...]
        is_question: bool
        is_copular_negation: bool
        is_existential_negation: bool
        has_hayir: bool

        @property
        def force_negative_verb(self) -> bool:
            # Both ``değil`` (§2.7) and ``hayır`` (§3.7) imply the verb
            # should be conjugated negatively per §2.6.1; combining
            # them is idempotent.
            return self.is_copular_negation or self.has_hayir

    def correct(self, words: List[str]) -> str:
        # Single normalization pass at the entry point — every
        # downstream stage reads from the same cleaned token stream,
        # so marker membership tests are O(1) hash lookups against
        # the registry sets. Total work is O(N) over the input.
        cleaned = normalize_words(words)
        if not cleaned:
            return ""

        # Lone non-verb (number, bare noun, interjection) shortcut:
        # capitalised + period, no SOV pipeline. Critical markers
        # fall through to the classifier so they shape the predicate.
        if len(cleaned) == 1 and not self._is_critical_marker_token(cleaned[0]):
            only = cleaned[0]
            if not (only.endswith("mek") or only.endswith("mak")):
                return self._punctuate([only])

        classification = self._classify(cleaned)

        verb_idx = self._find_verb_index(classification.semantic_payload)
        if verb_idx is None:
            return self._build_nominal(classification)
        return self._build_verbal(classification, verb_idx)

    @staticmethod
    def _classify(cleaned: List[str]) -> "RuleBasedCorrector._Classification":
        """Linear-pass token classification. Returns a strict two-vector
        decomposition that *guarantees* functional markers are absent
        from ``semantic_payload``.
        """
        interjections: List[str] = []
        semantic: List[str] = []
        is_question = False
        is_copular_negation = False
        is_existential_negation = False
        has_hayir = False

        # Greetings cluster at the head; their position matters for
        # the comma-after-greeting punctuation rule. Everything after
        # the greeting prefix is body.
        i = 0
        while i < len(cleaned) and cleaned[i] in _GREETINGS:
            interjections.append(cleaned[i])
            i += 1

        for tok in cleaned[i:]:
            if tok in _QUESTION_PARTICLE_VARIANTS:
                is_question = True
            elif tok in _NEGATION_COPULA_TOKENS:
                is_copular_negation = True
            elif tok in _EXISTENTIAL_NEGATION_TOKENS:
                is_existential_negation = True
            elif tok in _STANDALONE_NEGATIVE_TOKENS:
                interjections.append(tok)
                has_hayir = True
            else:
                semantic.append(tok)

        return RuleBasedCorrector._Classification(
            semantic_payload=tuple(semantic),
            leading_interjections=tuple(interjections),
            is_question=is_question,
            is_copular_negation=is_copular_negation,
            is_existential_negation=is_existential_negation,
            has_hayir=has_hayir,
        )

    @staticmethod
    def _find_verb_index(payload: Tuple[str, ...]) -> Optional[int]:
        """Rightmost ``-mek/-mak`` infinitive — handles inverted/devrik
        streams (``["gitmek", "sen", "okul"]``) and complement-verb
        compounds where the main verb isn't last.
        """
        for idx in range(len(payload) - 1, -1, -1):
            if payload[idx].endswith("mek") or payload[idx].endswith("mak"):
                return idx
        return None

    @staticmethod
    def _split_subject_middle(
        remaining: Tuple[str, ...],
    ) -> Tuple[Optional[str], List[str], str]:
        """Pick subject = first remaining token; middle = the rest.
        Pronouns drive person agreement directly; non-pronoun subjects
        default to 3rd-person singular. Verb-only input falls back to
        1st-person (``ben``) which reads naturally for sign streams.
        """
        if not remaining:
            return None, [], "ben"
        subject = remaining[0]
        pronoun = subject if subject in _PRONOUNS else "o"
        return subject, list(remaining[1:]), pronoun

    def _build_verbal(
        self,
        c: "RuleBasedCorrector._Classification",
        verb_idx: int,
    ) -> str:
        """Synthesise a sentence around a verbal predicate.

        Precondition: every token in ``c.semantic_payload`` is a noun,
        pronoun, or verb. Functional markers are *not* present here —
        they are encoded in the boolean flags on ``c``.
        """
        payload = c.semantic_payload
        verb = payload[verb_idx]
        remaining = payload[:verb_idx] + payload[verb_idx + 1 :]
        subject, middle, pronoun_for_conjugation = self._split_subject_middle(remaining)
        pronoun_key = _PRONOUN_PERSON.get(pronoun_for_conjugation, "1s")

        case = _CASE_GOVERNORS.get(verb)
        inflected_middle = [self._apply_case(w, case) for w in middle]

        tokens: List[str] = list(c.leading_interjections)
        if subject is not None:
            tokens.append(subject)
        tokens.extend(inflected_middle)

        # Predicate shape (mutually exclusive after this point):
        #   interrogative  →  bare-yor + harmonized particle + "?"
        #   existential    →  yok-with-agreement
        #   forced-neg     →  -me/-ma- infix per §2.6.1
        #   default        →  positive present continuous
        if c.is_question:
            stem_form = (
                negate_present_continuous(verb, "o")
                if c.force_negative_verb
                else conjugate_present_continuous(verb, "o")
            )
            tokens.append(stem_form)
            tokens.append(build_question_particle(stem_form, pronoun_key))
            return self._punctuate(tokens, terminator="?")

        if c.is_existential_negation:
            tokens.append(_EXISTENTIAL_NEGATION_AGREEMENT[pronoun_key])
            return self._punctuate(tokens)

        if c.force_negative_verb:
            tokens.append(negate_present_continuous(verb, pronoun_for_conjugation))
            return self._punctuate(tokens)

        tokens.append(conjugate_present_continuous(verb, pronoun_for_conjugation))
        return self._punctuate(tokens)

    @staticmethod
    def _is_critical_marker_token(token: str) -> bool:
        """Truth predicate used by the single-token shortcut. A lone
        marker token (``mi``, ``değil``, ``yok``) must NOT be treated
        as a regular noun — it falls through to the marker-aware
        pipeline instead.
        """
        return (
            token in _QUESTION_PARTICLE_VARIANTS
            or token in _NEGATION_COPULA_TOKENS
            or token in _EXISTENTIAL_NEGATION_TOKENS
            or token in _STANDALONE_NEGATIVE_TOKENS
        )

    def _build_nominal(
        self,
        c: "RuleBasedCorrector._Classification",
    ) -> str:
        """Compose the sentence when the semantic payload has no
        ``-mek/-mak`` verb. Three marker-driven shapes are produced:

        * ``Sen öğrenci misin?`` — question particle attaches with 4-way
          harmony off the predicate noun and carries person agreement.
        * ``Ben öğrenci değilim.`` — copular negation appends ``değil``
          with person agreement.
        * ``Ev yok.`` — existential negation replaces the predicate.

        With no markers at all the body is emitted verbatim — preserves
        the ``Merhaba, ben eray.`` shape for greeting+noun-only input.
        """
        leading = list(c.leading_interjections)
        body = list(c.semantic_payload)

        # Empty body but markers present: surface the bare marker so
        # the output is at least grammatically anchored.
        if not body:
            if c.is_question:
                return self._punctuate(leading + ["mi"], terminator="?")
            if c.is_existential_negation:
                return self._punctuate(leading + ["yok"])
            if c.is_copular_negation:
                return self._punctuate(leading + ["değil"])
            return self._punctuate(leading)

        if not (c.is_question or c.is_copular_negation or c.is_existential_negation):
            return self._punctuate(leading + body)

        # Body shape: optional subject (first) + middle + predicate (last).
        if len(body) == 1:
            subject: Optional[str] = None
            predicate = body[0]
            middle: List[str] = []
            # A lone pronoun is the subject, so the copula/question must
            # agree with it ("ben mi" → "Ben miyim?", not "Ben mi?"). A
            # non-pronoun predicate falls back to 3rd person.
            pronoun_key = _PRONOUN_PERSON.get(predicate, "3s")
        else:
            subject = body[0]
            predicate = body[-1]
            middle = body[1:-1]
            pronoun_key = _PRONOUN_PERSON.get(subject, "3s")

        tokens: List[str] = list(leading)
        if subject is not None:
            tokens.append(subject)
        tokens.extend(middle)
        tokens.append(predicate)

        if c.is_question:
            tokens.append(build_question_particle(predicate, pronoun_key))
            return self._punctuate(tokens, terminator="?")
        if c.is_copular_negation:
            tokens.append(_NEGATION_COPULA_AGREEMENT[pronoun_key])
            return self._punctuate(tokens)
        # Existential negation appends the harmonised ``yok`` copular
        # form after the predicate noun (``["ev", "yok"]`` → ``Ev yok.``).
        tokens.append(_EXISTENTIAL_NEGATION_AGREEMENT[pronoun_key])
        return self._punctuate(tokens)
    @staticmethod
    def _apply_case(noun: str, case: Optional[str]) -> str:
        # Belt-and-suspenders: even if a critical marker reaches this
        # point (it should not — ``_classify`` strips them), never let
        # it acquire a noun case suffix. Markers are not nouns.
        if RuleBasedCorrector._is_critical_marker_token(noun):
            return noun
        # Personal pronouns decline irregularly (bana/ona/onu…); consult the
        # explicit table before the generic noun appliers would mangle them.
        if case is not None:
            irregular = _PRONOUN_CASE_FORMS.get(noun)
            if irregular is not None and case in irregular:
                return irregular[case]
        if case == "dative":
            return apply_dative(noun)
        if case == "locative":
            return apply_locative(noun)
        if case == "ablative":
            return apply_ablative(noun)
        if case == "accusative":
            return apply_accusative(noun)
        return noun

    @staticmethod
    def _punctuate(tokens: List[str], terminator: str = ".") -> str:
        """Join tokens with single spaces, capitalize, terminate.

        ``terminator`` is appended only if the sentence does not
        already end in one of ``.``/``!``/``?`` — callers pass ``"?"``
        for interrogative shapes.
        """
        if not tokens:
            return ""
        # Drop empties defensively so a stray "" or whitespace-only
        # token never produces ``"Ben  geliyorum."`` with a double space.
        cleaned_tokens = [t for t in tokens if t and t.strip()]
        if not cleaned_tokens:
            return ""
        parts: List[str] = []
        for i, tok in enumerate(cleaned_tokens):
            prev_interjection = i > 0 and (
                cleaned_tokens[i - 1] in _GREETINGS
                or cleaned_tokens[i - 1] in _STANDALONE_NEGATIVE_TOKENS
            )
            if prev_interjection and tok not in {",", ".", "!", "?"}:
                parts[-1] = parts[-1] + ","
            parts.append(tok)
        sentence = " ".join(parts)
        # Turkish-aware first-letter capitalization: locale-naive str.upper()
        # turns "i" into "I" (should be "İ") and would mishandle "ı" → "I".
        first = sentence[0]
        first_upper = "İ" if first == "i" else "I" if first == "ı" else first.upper()
        sentence = first_upper + sentence[1:]
        if not sentence.endswith((".", "!", "?")):
            sentence += terminator
        return sentence
