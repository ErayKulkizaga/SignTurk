"""Versioned zero-shot prompt templates for the ML grammar layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .linguistics import normalize_words
from .registry import PROMPT_VERSION

# ───────────────────────── Prompt templates ─────────────────────────


# Few-shot demonstrations are intentionally empty: the zero-shot prompt
# below eliminates the parrot-the-example failure mode small models hit
# when they encounter a tricky input. The constant is retained (empty)
# so the leak validator continues to work as a defensive no-op.
_FEW_SHOT_EXAMPLES: List[tuple] = []


_ZERO_SHOT_TEMPLATE = (
    "Verilen işaret dili kelimelerini dilbilgisi kurallarına uygun, "
    "düzgün bir Türkçe cümleye çevir.\nKelimeler: {words}\nCümle:"
)


def build_zero_shot_prompt(words: List[str], version: Optional[str] = None) -> str:
    """Strict zero-shot instruction. No demonstrations — small models
    parrot them and even larger models phrase more cleanly without the
    example bias. Used by every adapter (local seq2seq, local causal,
    HF Inference API text_generation path). ``version`` selects a prompt
    from ``PROMPT_TEMPLATES`` (defaults to ``PROMPT_VERSION``)."""
    return _resolve_prompt(version).zero_shot_template.format(
        words=" ".join(normalize_words(words))
    )


_CHAT_SYSTEM_INSTRUCTION = (
    "Sen uzman bir Türk İşaret Dili (TİD) çevirmenisin. Görevin, "
    "verilen kök kelimeleri doğal bir Türkçe cümleye VEYA tamlamaya "
    "çevirmektir. Kurallar:\n"
    "- Kelimeler anlamsal olarak bir isim tamlaması veya fiilimsi "
    "grubu oluşturuyorsa (Örn: 'kafe kapanmak zaman' -> 'kafenin "
    "kapanma zamanı', 'ben su içmek istemek' -> 'Ben su içmek "
    "istiyorum'), bu yapıyı koru. Her eylemi zorla ana yüklem yapma.\n"
    "- Girdide bulunmayan 'istemek', 'gerekmek', 'başlamak' gibi "
    "yeni fiiller veya fazladan anlamlar KESİNLİKLE uydurma. "
    "Sadece sana verilen kelimelerdeki eylemi ve niyeti yansıt.\n"
    "- Sadece cümlenin asıl yargısını bildiren ana fiili özneye ve "
    "zamana göre çekimle.\n"
    "- İsim tamlamalarında (tamlayan/tamlanan) ve nesnelerde "
    "(yönelme, bulunma, belirtme) ismin hâl eklerini doğru kullan.\n"
    "- Yalnızca nihai metni yaz; açıklama veya etiket ekleme."
)


# Refined prompt variant — more explicit about the no-new-meaning constraint
# and a single-sentence output. Opt in via ``prompt_version`` /
# ``$SIGNAI_PROMPT_VERSION``; compare against zs-1 with the eval harness.
_ZERO_SHOT_TEMPLATE_V2 = (
    "Aşağıdaki Türk İşaret Dili kök kelimelerini, anlamı değiştirmeden ve "
    "girdide olmayan kelime eklemeden tek ve akıcı bir Türkçe cümleye çevir.\n"
    "Kelimeler: {words}\nCümle:"
)
_CHAT_SYSTEM_INSTRUCTION_V2 = (
    _CHAT_SYSTEM_INSTRUCTION + "\n- Çıktı yalnızca tek bir cümle olmalı; birden fazla cümle üretme."
)


@dataclass(frozen=True)
class PromptSpec:
    """A versioned prompt: a zero-shot completion template (``{words}``) and a
    chat system instruction. Lets us A/B prompt wording without code edits and
    trace which version produced a given output (logged with each decision)."""

    version: str
    zero_shot_template: str
    chat_system: str


PROMPT_TEMPLATES: Dict[str, PromptSpec] = {
    "zs-1": PromptSpec("zs-1", _ZERO_SHOT_TEMPLATE, _CHAT_SYSTEM_INSTRUCTION),
    "zs-2": PromptSpec("zs-2", _ZERO_SHOT_TEMPLATE_V2, _CHAT_SYSTEM_INSTRUCTION_V2),
}


def _resolve_prompt(version: Optional[str]) -> PromptSpec:
    spec = PROMPT_TEMPLATES.get(version or PROMPT_VERSION)
    if spec is None:
        raise ValueError(
            f"Unknown prompt_version {version!r}. Choose from: {sorted(PROMPT_TEMPLATES)}"
        )
    return spec


def list_prompt_versions() -> List[str]:
    return sorted(PROMPT_TEMPLATES)


def build_zero_shot_messages(
    words: List[str], version: Optional[str] = None
) -> List[Dict[str, str]]:
    """Chat-style version of the zero-shot prompt for conversational
    Inference API models (Qwen, Llama-Instruct, ...). The system role
    carries the task instruction; the user role carries the inputs.
    ``version`` selects a prompt from ``PROMPT_TEMPLATES``.
    """
    spec = _resolve_prompt(version)
    joined = " ".join(normalize_words(words))
    return [
        {"role": "system", "content": spec.chat_system},
        {"role": "user", "content": f"Kelimeler: {joined}"},
    ]


# Back-compat aliases: the historical seq2seq/causal prompt split is no
# longer needed — both adapter families now run the same zero-shot text.
def build_seq2seq_prompt(words: List[str], version: Optional[str] = None) -> str:
    return build_zero_shot_prompt(words, version)


def build_causal_prompt(words: List[str], version: Optional[str] = None) -> str:
    return build_zero_shot_prompt(words, version)
