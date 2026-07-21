"""Turkish phonology & morphology primitives — pure domain, no I/O.

Vowel/consonant classes, person-agreement and case-government tables, the
phonology engine (harmony, softening, assimilation, narrowing), noun
case-suffix appliers, and verb conjugation. Depends on nothing else in the
package; imported by every other grammar submodule.
"""

from __future__ import annotations

import string
from typing import Dict, List

# ── Vowel classes (TDK §1.2: dilin durumu, dudak, ağız açıklığı) ──
_VOWELS = "aeıioöuü"
_BACK_VOWELS = "aıou"  # kalın ünlüler
_FRONT_VOWELS = "eiöü"  # ince ünlüler
_ROUNDED_VOWELS = "oöuü"  # yuvarlak ünlüler
_UNROUNDED_VOWELS = "aeıi"  # düz ünlüler

_TR_CHARS = "çğıöşüÇĞİÖŞÜ"

# ── Consonant classes (TDK §1.3) ──
# "Fıstıkçı Şahap" — sert ünsüzler that trigger suffix-initial
# hardening (ünsüz benzeşmesi, §1.6).
_VOICELESS_CONSONANTS = "fstkçşhp"
# Stem-final consonants that soften before a vowel suffix (§1.7).
_SOFTENING_MAP = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}
# Suffix-initial hardening after a voiceless stem (§1.6).
_HARDENING_MAP = {"c": "ç", "d": "t", "g": "k"}

# Monosyllabic stems that exceptionally soften (§1.7.3 lists these as
# dictionary items — the algorithm cannot derive them).
_MONO_SOFTEN_EXCEPTIONS = {
    "uç": "uc",
    "dip": "dib",
    "renk": "reng",
    "denk": "deng",
    "kap": "kab",
    "cep": "ceb",
}

# Irregular monosyllabic verb stems whose final consonant softens
# despite the multi-syllable rule (``git`` → ``gidiyor``).
_IRREGULAR_VERB_STEMS = {
    "gitmek": "gid",
    "etmek": "ed",
    "tatmak": "tad",
}

_GREETINGS = {"merhaba", "selam", "günaydın", "iyi günler", "iyi akşamlar"}

# ── Person agreement (kişi ekleri, §2.6.4) ──
_PRONOUN_PERSON = {
    "ben": "1s",
    "sen": "2s",
    "o": "3s",
    "biz": "1p",
    "siz": "2p",
    "onlar": "3p",
}
_PRONOUNS = set(_PRONOUN_PERSON)

# Suffixes attaching after the -yor tense morpheme. The "o" of -yor
# fixes u-harmony for 1st/2nd persons, so these strings are invariant
# across all verbs; only -lar (3p) follows last-vowel harmony but here
# it always follows -yor's "o" → -lar.
_YOR_PERSON_SUFFIX = {
    "1s": "um",
    "2s": "sun",
    "3s": "",
    "1p": "uz",
    "2p": "sunuz",
    "3p": "lar",
}

# ── Predicate-shape registries (§2.6.5 soru eki, §2.7.4 değil) ──
#
# Tokens that radically reshape the predicate rather than acting as
# noun objects. The rule-based engine extracts these out of the SOV
# body *before* case-government runs, so they never receive spurious
# dative/locative/accusative suffixes.

# Question particle (mi/mı/mu/mü) carries 4-way harmony of the
# preceding word's last vowel; person agreement attaches to the
# particle (``geliyor musun``, ``geliyor muyum``) not the verb.
_QUESTION_PARTICLE_TOKENS = frozenset({"mi", "mı", "mu", "mü"})
# Conjugated forms that may already show up in the LSTM dictionary —
# any of these also flips the predicate into interrogative shape.
_QUESTION_PARTICLE_VARIANTS = frozenset(
    {
        "mi",
        "mı",
        "mu",
        "mü",
        "misin",
        "mısın",
        "musun",
        "müsün",
        "miyim",
        "mıyım",
        "muyum",
        "müyüm",
        "miyiz",
        "mıyız",
        "muyuz",
        "müyüz",
    }
)

# Negative copula ``değil`` carries straightforward person agreement
# and pairs with a bare ``-yor`` verb form (``geliyor değilim``).
_NEGATION_COPULA_TOKENS = frozenset({"değil"})
_NEGATION_COPULA_AGREEMENT = {
    "1s": "değilim",
    "2s": "değilsin",
    "3s": "değil",
    "1p": "değiliz",
    "2p": "değilsiniz",
    "3p": "değiller",
}

# Existential negation ``yok`` is its own predicate head — it replaces
# the verb rather than attaching to it (``Ev yok.`` not ``Ev yokuyor.``).
_EXISTENTIAL_NEGATION_TOKENS = frozenset({"yok"})
_EXISTENTIAL_NEGATION_AGREEMENT = {
    "1s": "yokum",
    "2s": "yoksun",
    "3s": "yok",
    "1p": "yokuz",
    "2p": "yoksunuz",
    "3p": "yoklar",
}

# Personal pronouns have IRREGULAR case forms that the generic noun
# case-appliers get wrong (``ben`` + dative is ``bana`` not ``bene``;
# ``o`` declines on an ``on-`` stem → ``ona/onu/onda/ondan`` not
# ``oya/oyu/oda/odan``). The rule engine consults this table before
# falling back to the regular appliers.
_PRONOUN_CASE_FORMS = {
    "ben": {"dative": "bana", "accusative": "beni", "locative": "bende", "ablative": "benden"},
    "sen": {"dative": "sana", "accusative": "seni", "locative": "sende", "ablative": "senden"},
    "o": {"dative": "ona", "accusative": "onu", "locative": "onda", "ablative": "ondan"},
    "biz": {"dative": "bize", "accusative": "bizi", "locative": "bizde", "ablative": "bizden"},
    "siz": {"dative": "size", "accusative": "sizi", "locative": "sizde", "ablative": "sizden"},
    "onlar": {
        "dative": "onlara",
        "accusative": "onları",
        "locative": "onlarda",
        "ablative": "onlardan",
    },
}

# Standalone interjection ``hayır`` — emitted bare with a trailing
# comma when followed by further content (``Hayır, ben geliyorum.``).
_STANDALONE_NEGATIVE_TOKENS = frozenset({"hayır"})

# ── Case-government tables (TDK §4.2.3-4.2.4) ──
# Each verb maps to the case it imposes on the preceding noun.
_MOTION_VERBS = {
    "gitmek",
    "gelmek",
    "varmak",
    "ulaşmak",
    "dönmek",
    "girmek",
    "binmek",
    "koşmak",
    "uğramak",
    "yetişmek",
    "yönelmek",
    "yaklaşmak",
}
_STATIVE_VERBS = {
    "beklemek",
    "durmak",
    "oturmak",
    "kalmak",
    "yaşamak",
    "uyumak",
    "yatmak",
    "bulunmak",
}
_ABLATION_VERBS = {
    "çıkmak",
    "ayrılmak",
    "kaçmak",
    "uzaklaşmak",
    "inmek",
    "vazgeçmek",
    "korkmak",
}
# Transitive verbs that govern an accusative-marked complement.
# NOTE: high-frequency "indefinite-object" verbs (istemek, okumak,
# görmek, almak, yemek) are deliberately *excluded* here. Sign-language
# input is article-less, and applying the accusative to those verbs
# yields stilted output like "Ben kitabı okuyorum." where the natural
# reading is the indefinite "Ben kitap okuyorum." (belirtisiz nesne,
# §4.2.3-B). Add a verb back if you want the definite reading.
_TRANSITIVE_VERBS = {
    "yazmak",
    "içmek",
    "vermek",
    "sevmek",
    "anlamak",
    "bilmek",
    "tanımak",
    "yapmak",
    "etmek",
    "açmak",
    "kapatmak",
    "izlemek",
    "dinlemek",
    "söylemek",
    "satmak",
    "kullanmak",
    "öğrenmek",
    "öğretmek",
    "düşünmek",
    "kırmak",
    "yıkamak",
    "bulmak",
    "kazanmak",
    "atmak",
    "çekmek",
}
_CASE_GOVERNORS: Dict[str, str] = {
    **{v: "dative" for v in _MOTION_VERBS},
    **{v: "locative" for v in _STATIVE_VERBS},
    **{v: "ablative" for v in _ABLATION_VERBS},
    **{v: "accusative" for v in _TRANSITIVE_VERBS},
}
# Back-compat alias used by older imports.
_DATIVE_VERBS = _MOTION_VERBS

# Characters stripped from token edges before any downstream processing.
# Covers ASCII punctuation plus the Unicode quotation marks / ellipses
# that real user input often contains.
_TOKEN_STRIP_CHARS = string.punctuation + "…“”‘’«»·"


# Turkish has a dotted/dotless ``i`` pair that Python's locale-naive
# ``str.lower()`` mishandles:
#   "İ" (U+0130) → "i̇"  (i + COMBINING DOT ABOVE, *not* "i")
#   "I" (U+0049) → "i"        (English rule; the Turkish lower of I is ``ı``)
# Translating these two code points before ``.lower()`` produces the
# canonical Turkish casefolding. Without this fix, an uppercased marker
# like ``"DEĞİL"`` fails membership in ``_NEGATION_COPULA_TOKENS`` and
# leaks into the SOV pipeline, where it picks up a dative suffix
# (``"deği̇le"``). See the regression test of the same name.
_TURKISH_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})


def turkish_lower(text: str) -> str:
    """Locale-correct lowercase for Turkish text."""
    return text.translate(_TURKISH_LOWER_MAP).lower()


def normalize_words(words: List[str]) -> List[str]:
    """Lower-case (Turkish-aware), trim, and strip edge punctuation.

    Applied uniformly to every entry-point so the rule-based and ML
    branches see the same shape of input. Tokens that collapse to empty
    after stripping (``"."``, ``"  "``) are dropped.
    """
    out: List[str] = []
    for w in words:
        if not isinstance(w, str):
            continue
        cleaned = turkish_lower(w.strip().strip(_TOKEN_STRIP_CHARS).strip())
        if cleaned:
            out.append(cleaned)
    return out


# ───────────────────────── Phonology engine ─────────────────────────


def _last_vowel(word: str, fallback: str = "a") -> str:
    for ch in reversed(word):
        if ch in _VOWELS:
            return ch
    return fallback


def _previous_vowel(word: str, fallback: str = "a") -> str:
    """Return the vowel before the final vowel of ``word`` (or ``fallback``).

    Used by the narrowing rule: when the trailing a/e of a verb stem
    is replaced before -yor, the roundness of the new vowel comes from
    the *preceding* vowel (``oyna`` references ``o`` → rounded → ``u``,
    ``başla`` references ``a`` → unrounded → ``ı``).
    """
    seen_last = False
    for ch in reversed(word):
        if ch in _VOWELS:
            if seen_last:
                return ch
            seen_last = True
    return fallback


def _syllable_count(word: str) -> int:
    """Naive syllable count = number of vowels. Adequate for Turkish."""
    return sum(1 for c in word if c in _VOWELS)

def apply_vowel_harmony(word: str, kind: str = "2way") -> str:
    """Return the harmony-correct suffix vowel for ``word``.

    ``kind="2way"`` returns ``a`` or ``e`` (büyük ünlü uyumu — dative
    ``-a/-e``, locative ``-da/-de``, ablative ``-dan/-den``, plural
    ``-lar/-ler``).

    ``kind="4way"`` returns ``ı/i/u/ü`` (küçük ünlü uyumu — accusative
    ``-(y)ı/-(y)i/-(y)u/-(y)ü``, possessive, the narrowing buffer
    inserted before ``-yor`` on consonant-final verb stems).
    """
    v = _last_vowel(word, "a")
    if kind == "2way":
        return "a" if v in _BACK_VOWELS else "e"
    if kind == "4way":
        if v in _BACK_VOWELS:
            return "u" if v in _ROUNDED_VOWELS else "ı"
        return "ü" if v in _ROUNDED_VOWELS else "i"
    raise ValueError(f"unknown harmony kind: {kind!r}")


def build_question_particle(preceding: str, pronoun_key: str = "3s") -> str:
    """Build the harmonic question particle for the predicate.

    The particle's vowel follows 4-way harmony off ``preceding``'s last
    vowel, and person agreement attaches to the particle itself rather
    than to the predicate verb (``Sen geliyor musun?``, not
    ``Sen geliyorsun mu?``). The 3rd-person plural is intentionally
    realised on the verb (``geliyorlar``) with the particle staying
    bare (``mı``), so this builder returns just the harmonic ``mV``
    for that key.
    """
    v = apply_vowel_harmony(preceding or "i", "4way")
    if pronoun_key == "1s":
        return f"m{v}y{v}m"
    if pronoun_key == "2s":
        return f"m{v}s{v}n"
    if pronoun_key == "1p":
        return f"m{v}y{v}z"
    if pronoun_key == "2p":
        return f"m{v}s{v}n{v}z"
    return f"m{v}"


def apply_consonant_softening(stem: str) -> str:
    """Soften a stem-final p/ç/t/k before a vowel suffix (ünsüz yumuşaması).

    Implements §1.7.1:

        kitap → kitab    ağaç → ağac
        kanat → kanad    çocuk → çocuį

    Monosyllabic stems are normally exempt (§1.7.3), except for the
    lexical exceptions listed in ``_MONO_SOFTEN_EXCEPTIONS``
    (``renk → reng``, ``uç → uc``, ...).
    """
    if not stem:
        return stem
    if stem in _MONO_SOFTEN_EXCEPTIONS:
        return _MONO_SOFTEN_EXCEPTIONS[stem]
    if stem[-1] in _SOFTENING_MAP and _syllable_count(stem) >= 2:
        return stem[:-1] + _SOFTENING_MAP[stem[-1]]
    return stem


def apply_consonant_assimilation(stem: str, suffix: str) -> str:
    """Harden a suffix-initial c/d/g to ç/t/k after a voiceless stem.

    Implements ünsüz benzeşmesi (§1.6 — "Fıstıkçı Şahap"). The stem is
    not modified; the suffix is returned with its first consonant
    flipped when needed:

        kitap + da  → kitap + ta
        ağaç  + dan → ağaç  + tan
        çiçek + ci  → çiçek + çi
    """
    if not (stem and suffix):
        return suffix
    if stem[-1] in _VOICELESS_CONSONANTS and suffix[0] in _HARDENING_MAP:
        return _HARDENING_MAP[suffix[0]] + suffix[1:]
    return suffix


def apply_vowel_narrowing(stem: str) -> str:
    """Narrow a verb-stem-final a/e to ı/i/u/ü before ``-yor`` (§1.8.2).

    The replacement vowel follows 4-way harmony of the *previous* vowel
    (or the stem-final vowel itself for monosyllabic stems):

        başla → başlı   (a, a) → ı       bekle → bekli   (e, e) → i
        oyna  → oynu    (o, a) → u       söyle → söylü   (ö, e) → ü
        ye    → yi      (e)    → i       de    → di      (e)    → i
    """
    if not stem or stem[-1] not in "ae":
        return stem
    ref = _previous_vowel(stem, fallback=stem[-1])
    if ref in _BACK_VOWELS:
        new_vowel = "u" if ref in _ROUNDED_VOWELS else "ı"
    else:
        new_vowel = "ü" if ref in _ROUNDED_VOWELS else "i"
    return stem[:-1] + new_vowel


# ─────────────────── Noun case-suffix appliers ────────────────────


def apply_dative(noun: str) -> str:
    """Attach yönelme hâli ``-a/-e`` with full phonology.

    Vowel harmony (2-way), ``y`` buffer for vowel-final stems, and
    final-consonant softening for multi-syllabic stems:

        okul  → okula     araba → arabaya
        ev    → eve       kitap → kitaba
        ayak  → ayağa     park  → parka
    """
    word = turkish_lower((noun or "").strip())
    if not word:
        return noun
    vowel = apply_vowel_harmony(word, "2way")
    if word[-1] in _VOWELS:
        return f"{word}y{vowel}"
    return f"{apply_consonant_softening(word)}{vowel}"


def apply_locative(noun: str) -> str:
    """Attach bulunma hâli ``-da/-de/-ta/-te``.

    Consonant-initial suffix → no softening; the suffix-initial ``d``
    hardens to ``t`` after a voiceless stem (§1.6):

        ev    → evde       okul → okulda
        kitap → kitapta    iş   → işte
    """
    word = turkish_lower((noun or "").strip())
    if not word:
        return noun
    suffix = "d" + apply_vowel_harmony(word, "2way")
    suffix = apply_consonant_assimilation(word, suffix)
    return f"{word}{suffix}"


def apply_ablative(noun: str) -> str:
    """Attach ayrılma hâli ``-dan/-den/-tan/-ten``.

    ev    → evden      okul → okuldan
    kitap → kitaptan   iş  → işten
    """
    word = turkish_lower((noun or "").strip())
    if not word:
        return noun
    suffix = "d" + apply_vowel_harmony(word, "2way") + "n"
    suffix = apply_consonant_assimilation(word, suffix)
    return f"{word}{suffix}"


def apply_accusative(noun: str) -> str:
    """Attach belirtme hâli ``-(y)ı/-(y)i/-(y)u/-(y)ü``.

    Vowel harmony (4-way), ``y`` buffer for vowel-final stems, final-
    consonant softening for multi-syllabic stems:

        kitap → kitabı    araba → arabayı
        ağaç  → ağacı     su    → suyu
        renk  → rengi     ev    → evi
    """
    word = turkish_lower((noun or "").strip())
    if not word:
        return noun
    vowel = apply_vowel_harmony(word, "4way")
    if word[-1] in _VOWELS:
        return f"{word}y{vowel}"
    return f"{apply_consonant_softening(word)}{vowel}"


def apply_dative_suffix(noun: str) -> str:
    """Back-compat alias for :func:`apply_dative`."""
    return apply_dative(noun)

# ───────────────────── Verb conjugation ────────────────────


def conjugate_present_continuous(infinitive: str, pronoun: str = "ben") -> str:
    """Conjugate a ``-mek/-mak`` infinitive into present continuous (§2.6.2C).

    Pipeline:
        1. Look up irregular pre-softened stem (git/et/tat) or strip
           the ``-mek/-mak`` suffix.
        2. Stems ending in ``a``/``e``: apply vowel narrowing (§1.8.2);
           narrowed vowel substitutes for the buffer, ``-yor`` attaches
           directly.
        3. Stems ending in any other vowel (``ı/i/u/ü/o/ö``): no buffer,
           ``-yor`` attaches directly (``oku`` → ``okuyor``).
        4. Consonant-final stems: optional softening for multi-syllabic
           p/ç/t/k (already pre-softened for irregular monosyllables),
           then insert a 4-way harmonized buffer vowel before ``-yor``.
        5. Append the person suffix selected by ``pronoun``. Defaults to
           1st person singular ("ben") because sign-language streams are
           overwhelmingly first-person and an implicit "ben" reads more
           naturally than the impersonal third-person form.

    Examples:
        gel   + ben → geliyorum    git   + sen → gidiyorsun
        anla  + biz → anlıyoruz    söyle + o   → söylüyor
        ye    + sen → yiyorsun     tut   + ben → tutuyorum
        oku   + ben → okuyorum     iste  + ben → istiyorum
    """
    pronoun_key = _PRONOUN_PERSON.get(turkish_lower((pronoun or "ben").strip()), "1s")
    person_suffix = _YOR_PERSON_SUFFIX[pronoun_key]

    verb = turkish_lower((infinitive or "").strip())
    if verb in _IRREGULAR_VERB_STEMS:
        stem = _IRREGULAR_VERB_STEMS[verb]
    elif verb.endswith(("mek", "mak")):
        stem = verb[:-3]
    else:
        return verb

    if not stem:
        return verb

    if stem[-1] in "ae":
        stem = apply_vowel_narrowing(stem)
        connector = ""
    elif stem[-1] in _VOWELS:
        connector = ""
    else:
        stem = apply_consonant_softening(stem)
        connector = apply_vowel_harmony(stem, "4way")

    return f"{stem}{connector}yor{person_suffix}"


def negate_present_continuous(infinitive: str, pronoun: str = "ben") -> str:
    """Conjugate a ``-mek/-mak`` infinitive into NEGATIVE present continuous.

    The negative morpheme ``-ma/-me`` (§2.6.1) sits between the stem
    and the tense suffix; the resulting stem ends in ``a``/``e`` so
    ünlü daralması (§1.8.2) narrows it before ``-yor``:

        gel- + -me + -yor + -um   → gelmiyorum
        git- + -me + -yor + -sun  → gitmiyorsun
        oku- + -ma + -yor + -uz   → okumuyoruz
        başla- + -ma + -yor + -lar → başlamıyorlar
        ye- + -me + -yor + -um    → yemiyorum

    Because the suffix that touches the stem (``m``) is a consonant,
    the irregular pre-softened roots (``gid``/``ed``/``tad``) are *not*
    used here — softening only fires before a vowel-initial suffix.
    """
    pronoun_key = _PRONOUN_PERSON.get(turkish_lower((pronoun or "ben").strip()), "1s")
    person_suffix = _YOR_PERSON_SUFFIX[pronoun_key]

    verb = turkish_lower((infinitive or "").strip())
    if verb.endswith(("mek", "mak")):
        stem = verb[:-3]
    else:
        return verb
    if not stem:
        return verb

    neg_vowel = apply_vowel_harmony(stem, "2way")  # -ma / -me
    with_neg = f"{stem}m{neg_vowel}"
    narrowed = apply_vowel_narrowing(with_neg)
    return f"{narrowed}yor{person_suffix}"
