"""Standalone CLI to try the hybrid grammar pipeline.

Usage:
    python -m text_processing.demo                       # rule-based only
    python -m text_processing.demo --use-ml              # default qwen2.5-7b-api (cloud)
    python -m text_processing.demo --use-ml --model mt0-small   # local seq2seq
    python -m text_processing.demo --use-ml --model turkish-gpt2-medium
    python -m text_processing.demo --use-ml --model-name my/model --model-arch causal
    python -m text_processing.demo --words "merhaba ben su istemek"
    python -m text_processing.demo --list-models
"""

from __future__ import annotations

import argparse
import sys

from . import (
    DEFAULT_MODEL_KEY,
    GrammarConfig,
    PipelineConfig,
    SignTextPipeline,
    TTSConfig,
    list_available_models,
)

DEFAULT_PHRASES = [
    "merhaba ben su istemek",
    "sen okul gitmek",
    "ben kitap okumak",
    "biz yemik yemek",
    "o ev gelmek",
]


def _print_models() -> None:
    print("Mevcut modeller:")
    for spec in list_available_models():
        tags = []
        if spec.instruction_tuned:
            tags.append("instruction-tuned")
        if spec.turkish_native:
            tags.append("Turkish-native")
        tags.append(spec.arch)
        suffix = f"  [{', '.join(tags)}]"
        print(f"  - {spec.key:<22} {spec.hf_name:<40} ~{spec.approx_size_mb} MB{suffix}")
        if spec.notes:
            print(f"      {spec.notes}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SignTurk hybrid grammar demo")
    parser.add_argument("--use-ml", action="store_true", help="enable the HF model layer")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, help="model key from the registry")
    parser.add_argument("--model-name", default=None, help="raw HF model name (overrides --model)")
    parser.add_argument(
        "--model-arch",
        default=None,
        choices=["seq2seq", "causal", "inference-api"],
        help="architecture for --model-name (inferred from the name when omitted)",
    )
    parser.add_argument("--words", default=None, help="space-separated words to feed")
    parser.add_argument(
        "--no-audio", action="store_true", help="skip TTS synthesis (gTTS and Piper)"
    )
    parser.add_argument("--list-models", action="store_true", help="print model registry and exit")
    args = parser.parse_args(argv)

    if args.list_models:
        _print_models()
        return 0

    pipeline = SignTextPipeline(
        PipelineConfig(
            grammar=GrammarConfig(
                use_ml=args.use_ml,
                model_key=args.model,
                model_name_override=args.model_name,
                model_arch_override=args.model_arch,
            ),
            tts=TTSConfig(),
            synthesize_audio=not args.no_audio,
        )
    )

    phrases = [args.words] if args.words else DEFAULT_PHRASES
    for phrase in phrases:
        words = phrase.split()
        result = pipeline.correct(words)
        print(f"\nGirdi  : {words}")
        print(f"Cümle  : {result.sentence}")
        print(f"Kaynak : {result.grammar_source}")
        if result.ml_latency_ms:
            print(f"ML süre: {result.ml_latency_ms:.0f} ms")
        if result.rejected_candidate:
            print(
                f"Reddedilen ML çıktısı ({result.rejection_reason}): {result.rejected_candidate!r}"
            )
        if result.audio_path:
            print(f"Ses    : {result.audio_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
