"""Model execution backends + Inference API error classification.

Local seq2seq/causal adapters, the Hugging Face serverless Inference API
adapter, and :func:`_classify_ml_error` — which turns the failures the ML
layer used to swallow into actionable, user-facing reasons.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, NamedTuple, Optional, Protocol

from .prompts import build_zero_shot_messages, build_zero_shot_prompt
from .registry import GrammarConfig, ModelSpec

logger = logging.getLogger(__name__)

# ───────────────────────── Model adapters ─────────────────────────


class ModelAdapter(Protocol):
    def generate(self, words: List[str], config: GrammarConfig) -> Optional[str]: ...


def _maybe_quantize_for_cpu(model, device: str, enabled: bool):
    """Apply dynamic int8 quantization to Linear layers on CPU.

    No-op on GPU (quantize_dynamic does not support CUDA kernels) and
    when explicitly disabled. Failures are logged and the original model
    is returned so the rest of the pipeline still works.
    """
    if not enabled or device != "cpu":
        return model
    try:
        import torch

        quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        logger.info("Applied dynamic int8 quantization for CPU inference.")
        return quantized
    except Exception as e:
        logger.warning("Dynamic quantization failed (%s); using fp model.", e)
        return model


class _Seq2SeqAdapter:
    def __init__(self, spec: ModelSpec, config: GrammarConfig) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(spec.hf_name)
        self.model.to(config.device)
        self.model.eval()
        self.model = _maybe_quantize_for_cpu(self.model, config.device, config.quantize_cpu_models)
        self.device = config.device

    def generate(self, words: List[str], config: GrammarConfig) -> Optional[str]:
        import torch

        prompt = build_zero_shot_prompt(words, config.prompt_version)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
            self.device
        )
        with torch.no_grad():
            # Strict greedy decoding (do_sample=False, num_beams=1) — kills
            # the creative hallucinations sampling produces on small models.
            out = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                do_sample=False,
                no_repeat_ngram_size=3,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return text or None

class _CausalAdapter:
    def __init__(self, spec: ModelSpec, config: GrammarConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_name)
        self.model.to(config.device)
        self.model.eval()
        self.model = _maybe_quantize_for_cpu(self.model, config.device, config.quantize_cpu_models)
        self.device = config.device

    def generate(self, words: List[str], config: GrammarConfig) -> Optional[str]:
        import torch

        prompt = build_zero_shot_prompt(words, config.prompt_version)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
            self.device
        )
        with torch.no_grad():
            # Greedy decoding — matches the seq2seq path.
            out = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                num_beams=config.num_beams,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        completion = full[len(prompt) :].strip() if full.startswith(prompt) else full.strip()
        completion = completion.split("\n", 1)[0].strip()
        return completion or None


def _extract_chat_content(result) -> Optional[str]:
    """Pull ``choices[0].message.content`` out of a chat_completion
    response, supporting both attribute-style objects (newer
    huggingface_hub) and plain dicts (older releases / test stubs).
    """
    try:
        return result.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        pass
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


class MLError(NamedTuple):
    """A classified ML failure.

    ``category`` is a stable identifier the UI/telemetry branch on
    (``quota``, ``auth``, ``not_served``, ``rate_limit``, ``timeout`` …);
    ``message`` is an action-oriented Turkish string shown to the user.
    """

    category: str
    message: str


# HF tokens look like ``hf_<alphanumerics>``. Never echo one back to a
# client, even if a provider error string happens to embed it.
_HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{4,}")


def _scrub_exception_text(exc: BaseException) -> str:
    """``str(exc)`` with any embedded HF token masked — safe for both
    user-facing messages and application logs."""
    return _HF_TOKEN_PATTERN.sub("hf_***", str(exc)).strip()

def _status_code_of(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status extraction from an Inference API exception.

    Works by duck-typing so this module never has to import
    ``huggingface_hub`` (an optional dependency): reads
    ``exc.response.status_code`` when present, otherwise scans the message.

    The message scan is deliberately conservative — it only accepts a
    4xx/5xx code that is *followed by a capitalized reason phrase* (``402
    Payment Required``, ``404 Client Error``) or *preceded by* ``HTTP`` /
    ``status``. That way unrelated integers in the text (``... <= 512``,
    ``line 503``, ``temperature 0.500``) are not mistaken for a status and
    misclassified.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    text = str(exc)
    match = re.search(r"\b([45]\d{2})\b(?=\s+[A-Z])", text) or re.search(
        r"(?:HTTP|status)\D{0,6}\b([45]\d{2})\b", text, re.IGNORECASE
    )
    return int(match.group(1)) if match else None


def _classify_ml_error(exc: BaseException) -> MLError:
    """Map an Inference API / model-load exception to an :class:`MLError`.

    Turns the broad failures the ML layer used to swallow into ``None``
    into an actionable, Turkish, user-facing reason. Status-bearing HF
    errors are mapped by HTTP code; network/timeouts and missing optional
    dependencies are matched on the message. Any embedded HF token is
    scrubbed from the free-text fallback.
    """
    if isinstance(exc, ModuleNotFoundError) or "no module named" in str(exc).lower():
        return MLError(
            "deps",
            "Gerekli paket kurulu değil — `pip install huggingface_hub` "
            "(yerel modeller için ayrıca `transformers`).",
        )
    status = _status_code_of(exc)
    if status == 402:
        return MLError(
            "quota",
            "HF ücretsiz kotası doldu — PRO hesap kullanın ya da başka bir model seçin.",
        )
    if status in (401, 403):
        return MLError("auth", "HF token geçersiz veya yetkisiz — HF_TOKEN değerini kontrol edin.")
    if status == 404:
        return MLError(
            "not_served",
            "Model sağlayıcıda sunulmuyor — başka bir model ya da güncel bir özel id deneyin.",
        )
    if status == 429:
        return MLError(
            "rate_limit",
            "İstek sınırına ulaşıldı (rate limit) — birazdan tekrar deneyin.",
        )
    if status in (408, 504):
        return MLError(
            "timeout",
            "Ağ geçidi zaman aşımına uğradı (gateway timeout) — birazdan tekrar deneyin.",
        )
    if status == 503:
        return MLError(
            "loading",
            "Model sağlayıcıda yükleniyor (cold start) — birkaç saniye sonra tekrar deneyin.",
        )
    if status is not None and 500 <= status < 600:
        return MLError(
            "provider_error",
            f"Model sağlayıcı hatası (HTTP {status}) — birazdan tekrar deneyin.",
        )
    low = str(exc).lower()
    if "timeout" in low or "timed out" in low:
        return MLError(
            "timeout", "Bağlantı zaman aşımına uğradı — ağ ya da model yavaş; tekrar deneyin."
        )
    if any(
        s in low for s in ("connection", "network", "resolve", "getaddrinfo", "name or service")
    ):
        return MLError("network", "Ağ hatası — internet bağlantısını kontrol edin.")
    detail = _scrub_exception_text(exc)
    if detail:
        return MLError("unknown", f"ML katmanı beklenmeyen bir hata verdi: {detail[:160]}")
    return MLError("unknown", "ML katmanı beklenmeyen bir hata verdi.")


class _InferenceAPIAdapter:
    """Runs grammar correction against the Hugging Face Inference API.

    Two transport modes are dispatched off ``ModelSpec.conversational``:

    * ``conversational=True`` — uses ``client.chat_completion`` with a
      role-based ``messages`` array. Required by Qwen-Instruct,
      Llama-Instruct and other models the HF gateway routes to
      "conversational"-only providers (text_generation 404s for them).
    * ``conversational=False`` (default) — uses raw
      ``client.text_generation`` with the zero-shot prompt string. Works
      for the mt0 / mt5 / GPT-2 family.

    Both modes share the same zero-shot instruction and the same strict
    near-deterministic decoding (``temperature=0.1``, ``do_sample=False``).
    No local model weights are downloaded. Picks up ``HF_TOKEN`` /
    ``HUGGINGFACE_API_TOKEN`` from the environment when no token is set on
    the config.

    Transport failures are RAISED (after token-scrubbed logging), not
    swallowed: the caller classifies them via :func:`_classify_ml_error`.
    A ``None`` return means the model produced empty/blank output. Keeping
    no per-request error state on the (shared, cached) adapter avoids a
    cross-request race on the failure reason.
    """

    def __init__(self, spec: ModelSpec, config: GrammarConfig) -> None:
        from huggingface_hub import InferenceClient

        token = (
            config.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
        )
        self.spec = spec
        self.client = InferenceClient(
            model=spec.hf_name,
            token=token,
            timeout=config.inference_api_timeout_seconds,
        )
        self._has_token = bool(token)

    def generate(self, words: List[str], config: GrammarConfig) -> Optional[str]:
        if self.spec.conversational:
            return self._generate_chat(words, config)
        return self._generate_text(words, config)


    def _generate_chat(self, words: List[str], config: GrammarConfig) -> Optional[str]:
        messages = build_zero_shot_messages(words, config.prompt_version)
        try:
            result = self.client.chat_completion(
                messages=messages,
                max_tokens=config.max_new_tokens,
                temperature=max(config.temperature, 0.01),
            )
        except Exception as e:
            logger.warning(
                "HF Inference API chat_completion failed for %s (token=%s): %s",
                self.spec.hf_name,
                self._has_token,
                _scrub_exception_text(e),
            )
            raise
        content = _extract_chat_content(result)
        if not content:
            return None
        # Chat models sometimes return multi-line answers; keep the
        # first non-empty line (the corrected sentence).
        for line in content.strip().splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return None

    def _generate_text(self, words: List[str], config: GrammarConfig) -> Optional[str]:
        prompt = build_zero_shot_prompt(words, config.prompt_version)
        # Strict near-deterministic decoding so the remote model can't
        # hallucinate from the zero-shot prompt. ``do_sample=False`` is
        # the strongest signal; ``temperature`` is kept as a defensive
        # fallback for InferenceClient versions that don't forward
        # ``do_sample`` to every backend.
        kwargs = dict(
            max_new_tokens=config.max_new_tokens,
            temperature=max(config.temperature, 0.01),
            do_sample=False,
            return_full_text=False,
        )
        try:
            text = self.client.text_generation(prompt, **kwargs)
        except TypeError:
            # Older huggingface_hub releases reject ``do_sample`` on the
            # text_generation call. Retry without it — temperature=0.1
            # still keeps generation close to deterministic.
            kwargs.pop("do_sample", None)
            try:
                text = self.client.text_generation(prompt, **kwargs)
            except Exception as e:
                logger.warning(
                    "HF Inference API text_generation failed for %s (token=%s): %s",
                    self.spec.hf_name,
                    self._has_token,
                    _scrub_exception_text(e),
                )
                raise
        except Exception as e:
            logger.warning(
                "HF Inference API text_generation failed for %s (token=%s): %s",
                self.spec.hf_name,
                self._has_token,
                _scrub_exception_text(e),
            )
            raise
        text = (text or "").strip()
        text = text.split("\n", 1)[0].strip()
        return text or None


def _build_adapter(spec: ModelSpec, config: GrammarConfig) -> ModelAdapter:
    if spec.arch == "seq2seq":
        return _Seq2SeqAdapter(spec, config)
    if spec.arch == "causal":
        return _CausalAdapter(spec, config)
    if spec.arch == "inference-api":
        return _InferenceAPIAdapter(spec, config)
    raise ValueError(f"Unsupported architecture: {spec.arch!r}")
