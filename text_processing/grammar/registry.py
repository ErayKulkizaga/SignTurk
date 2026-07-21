"""Model registry & grammar configuration.

The catalog of supported models (``ModelSpec`` + ``MODEL_REGISTRY``), the
default-model policy, the prompt-version constant, and ``GrammarConfig``
(which resolves a request to a concrete ``ModelSpec``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ───────────────────────── Model registry ─────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_name: str
    arch: str
    approx_size_mb: int
    instruction_tuned: bool = False
    turkish_native: bool = False
    recommended: bool = False
    # When True, the Inference API adapter dispatches to
    # ``client.chat_completion`` with a messages array instead of
    # ``client.text_generation`` with a raw prompt. Required for HF
    # serverless models routed to "conversational"-only providers
    # (Qwen-Instruct, Llama-Instruct, ...).
    conversational: bool = False
    notes: str = ""


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ── Cloud / Inference API ──
    # Qwen2.5-7B-Instruct is a strong open-weights instruction model
    # under 10B for Turkish (competitive with Llama-3.1-8B-Instruct and
    # Gemma-2-9B-it on public leaderboards). We do NOT ship our own
    # benchmark proving it is "SOTA" for TİD-style translation; for
    # TİD-specific numbers run ``python -m text_processing.eval`` against
    # ``text_processing/eval/dataset.jsonl``. Add another cloud entry
    # only when the eval harness shows it is measurably better.
    "qwen2.5-7b-api": ModelSpec(
        key="qwen2.5-7b-api",
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        arch="inference-api",
        approx_size_mb=0,
        instruction_tuned=True,
        recommended=True,
        conversational=True,
        notes="HF Inference API · Qwen2.5-7B-Instruct (chat_completion). "
        "Strong instruction LLM with good Turkish. Cloud, no local "
        "download. ★ Recommended.",
    ),
    # Newer Qwen generation — experimental, NOT the default (we couldn't verify
    # the exact serverless id from CI). Set HF_TOKEN and select it to A/B
    # against qwen2.5; if this id isn't served, use a custom model id.
    "qwen3-8b-api": ModelSpec(
        key="qwen3-8b-api",
        hf_name="Qwen/Qwen3-8B",
        arch="inference-api",
        approx_size_mb=0,
        instruction_tuned=True,
        conversational=True,
        recommended=False,
        notes="EXPERIMENTAL · Qwen3-8B (cloud, chat). Newer than Qwen2.5. "
        "HF_TOKEN gerekir; serverless'ta yoksa Özel model kutusundan "
        "güncel id dene.",
    ),
    # ── Local — downloaded into HF cache on first use ──
    "mt0-small": ModelSpec(
        key="mt0-small",
        hf_name="bigscience/mt0-small",
        arch="seq2seq",
        approx_size_mb=600,
        instruction_tuned=True,
        notes="Multilingual instruction-tuned T5. Strong zero-shot Turkish.",
    ),
    "mt0-base": ModelSpec(
        key="mt0-base",
        hf_name="bigscience/mt0-base",
        arch="seq2seq",
        approx_size_mb=2300,
        instruction_tuned=True,
        notes="Bigger mT0, better fluency but heavier.",
    ),
    "mt5-small": ModelSpec(
        key="mt5-small",
        hf_name="google/mt5-small",
        arch="seq2seq",
        approx_size_mb=1200,
        notes="Vanilla multilingual T5. Not instruction-tuned; weak zero-shot.",
    ),
    "turkish-gpt2": ModelSpec(
        key="turkish-gpt2",
        hf_name="ytu-ce-cosmos/turkish-gpt2",
        arch="causal",
        approx_size_mb=500,
        turkish_native=True,
        notes="Native Turkish GPT-2. Causal LM, completion-style prompting.",
    ),
    "turkish-gpt2-medium": ModelSpec(
        key="turkish-gpt2-medium",
        hf_name="ytu-ce-cosmos/turkish-gpt2-medium",
        arch="causal",
        approx_size_mb=1400,
        turkish_native=True,
        notes="Larger Turkish GPT-2. Better fluency, slower inference.",
    ),
}

# Two explicit defaults so callers never silently download a local model
# when they meant to hit the cloud API (and vice-versa):
#   * DEFAULT_API_MODEL_KEY  — cloud HF Inference API, zero local download.
#   * DEFAULT_LOCAL_MODEL_KEY — smallest local model, for offline/no-token use.
# The overall default is the cloud API: enabling ML without naming a model
# must NOT pull a multi-hundred-MB checkpoint onto the host. Callers that
# want a local model pick one explicitly (e.g. DEFAULT_LOCAL_MODEL_KEY).
DEFAULT_API_MODEL_KEY = "qwen2.5-7b-api"
DEFAULT_LOCAL_MODEL_KEY = "mt0-small"
DEFAULT_MODEL_KEY = DEFAULT_API_MODEL_KEY

# Prompt template version, logged with every decision so a given output can
# be traced back to the exact prompt that produced it.
PROMPT_VERSION = "zs-1"

# Architectures the adapter factory knows how to build.
_SUPPORTED_ARCHS = frozenset({"seq2seq", "causal", "inference-api"})

# Substring hints used to guess an architecture for a raw ``model_name_override``
# when the caller did not state one explicitly. Causal hints win over seq2seq.
_CAUSAL_NAME_HINTS = (
    "gpt",
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "falcon",
    "phi",
    "bloom",
    "opt",
)
_SEQ2SEQ_NAME_HINTS = ("t5", "mt0", "mt5", "bart", "mbart", "pegasus", "ul2")

def _infer_arch_from_name(hf_name: str) -> str:
    """Best-effort architecture guess for a raw HF model name.

    Used only on the ``model_name_override`` path, where the historical
    behavior was to *always* assume ``seq2seq`` — which silently picks the
    wrong adapter for a causal/chat model. Defaults to ``seq2seq`` when no
    hint matches (the safest local loader); callers can force the arch via
    ``GrammarConfig.model_arch_override`` / ``$SIGNAI_GRAMMAR_MODEL_ARCH``.
    """
    lname = hf_name.lower()
    if any(h in lname for h in _CAUSAL_NAME_HINTS):
        return "causal"
    if any(h in lname for h in _SEQ2SEQ_NAME_HINTS):
        return "seq2seq"
    return "seq2seq"


# ───────────────────────── Config ─────────────────────────


@dataclass
class GrammarConfig:
    use_ml: bool = False
    model_key: str = DEFAULT_MODEL_KEY
    model_name_override: Optional[str] = None
    # Architecture for ``model_name_override`` ("seq2seq" | "causal" |
    # "inference-api"). When None, it is inferred from the name (see
    # ``_infer_arch_from_name``) instead of blindly assuming seq2seq.
    model_arch_override: Optional[str] = None
    # Prompt template version (see ``PROMPT_TEMPLATES``). Logged with every
    # decision so an output can be traced back to the prompt that produced it.
    prompt_version: str = PROMPT_VERSION
    # Decoding parameters — defaults tuned for strict, deterministic
    # zero-shot generation (no creative hallucinations from sampling).
    max_new_tokens: int = 50
    inference_timeout_seconds: float = 6.0
    device: str = "cpu"
    num_beams: int = 1  # greedy (was 4-beam search)
    temperature: float = 0.1  # near-deterministic (was 0.7)
    accept_fallback_on_validation_failure: bool = True
    # Apply torch.quantization.quantize_dynamic to local CPU models so
    # inference fits in laptop RAM and finishes inside the timeout.
    quantize_cpu_models: bool = True
    # HF Inference API auth. Picked up from $HF_TOKEN /
    # $HUGGINGFACE_API_TOKEN when None. Anonymous calls work for public
    # models but are heavily rate-limited.
    hf_token: Optional[str] = None
    # Per-request timeout passed to the InferenceClient.
    inference_api_timeout_seconds: float = 15.0

    def resolve_spec(self) -> ModelSpec:
        if self.model_name_override:
            arch = self.model_arch_override or _infer_arch_from_name(self.model_name_override)
            if arch not in _SUPPORTED_ARCHS:
                raise ValueError(
                    f"Unknown model_arch_override {arch!r}. Choose from: {sorted(_SUPPORTED_ARCHS)}"
                )
            if self.model_arch_override is None:
                logger.info(
                    "model_name_override %r: inferred arch=%s (set "
                    "model_arch_override / $SIGNAI_GRAMMAR_MODEL_ARCH to force).",
                    self.model_name_override,
                    arch,
                )
            return ModelSpec(
                key="custom",
                hf_name=self.model_name_override,
                arch=arch,
                approx_size_mb=0,
                conversational=(arch == "inference-api"),
                notes=f"User-provided model name (arch={arch}).",
            )
        spec = MODEL_REGISTRY.get(self.model_key)
        if spec is None:
            raise ValueError(
                f"Unknown model_key {self.model_key!r}. Choose from: {sorted(MODEL_REGISTRY)}"
            )
        return spec
