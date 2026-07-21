"""Small labelled eval set + practical metrics for the grammar layer.

Why this exists: the repo asserted "SOTA" with no evidence. Rather than
BLEU/chrF (which need reference corpora and reward surface overlap), we
track the properties that actually matter for TİD → Turkish:

  * exact_match        — normalized equality with the gold sentence
  * root_preservation  — fraction of input lemmas surviving into the output
  * intent_ok          — no fabricated intent/necessity verb when the input
                         had none (the canonical hallucination failure)
  * question_ok        — interrogatives stay interrogative ("?")
  * negation_ok        — negation/existential stays negative

Run::

    python -m text_processing.eval                 # rule-based engine
    python -m text_processing.eval --use-ml        # + ML layer (needs HF_TOKEN/model)
    python -m text_processing.eval --json out.json # machine-readable report

All metrics reuse the production heuristics in ``text_processing.grammar``
so the eval measures the same notions the arbiter enforces at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..grammar import (
    _CRITICAL_NEGATION_TOKENS,
    _CRITICAL_QUESTION_TOKENS,
    GrammarConfig,
    GrammarCorrector,
    _input_signals_intent,
    _ml_contains_hallucinated_intent,
    _root_preservation_score,
    normalize_words,
    turkish_lower,
)

DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"

# Negative present-continuous infix (gelmiyorum, gitmiyorsun, okumuyorum, …)
# plus the copular/existential negators. Used only as a preservation check.
_NEG_RE = re.compile(r"m[iıuü]yor")


@dataclass
class Example:
    words: List[str]
    expected: str
    tags: List[str] = field(default_factory=list)


@dataclass
class CaseResult:
    words: List[str]
    expected: str
    output: str
    source: str
    tags: List[str]
    ml_latency_ms: float
    exact: bool
    root_preservation: float
    intent_ok: bool
    question_ok: Optional[bool]
    negation_ok: Optional[bool]


def load_dataset(path: Path = DATASET_PATH) -> List[Example]:
    examples: List[Example] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                Example(words=obj["words"], expected=obj["expected"], tags=obj.get("tags", []))
            )
    return examples

def _normalize_sentence(text: str) -> str:
    """Lowercase (Turkish-aware), drop terminal punctuation, collapse spaces."""
    lowered = turkish_lower(text.strip())
    lowered = re.sub(r"[.?!,]", " ", lowered)
    return " ".join(lowered.split())


def _has_question(words: List[str]) -> bool:
    return any(w in _CRITICAL_QUESTION_TOKENS for w in normalize_words(words))


def _has_negation(words: List[str]) -> bool:
    return any(w in _CRITICAL_NEGATION_TOKENS for w in normalize_words(words))


def _negation_in_output(output: str) -> bool:
    lower = turkish_lower(output)
    return bool(_NEG_RE.search(lower)) or "değil" in lower or "yok" in lower or "hayır" in lower


def _intent_ok(norm_words: List[str], output: str) -> bool:
    """Intent vocab must be present iff the input asked for it.

    Reuses the production detector symmetrically: ``_ml_contains_hallucinated_intent``
    flags intent/necessity words (``isti*``/``gerek*``/``lazım``) in the output —
    which is exactly the signal we want present when the input has an intent
    lemma and absent when it does not (catches the hallucination failure mode).
    """
    intent_in_output = _ml_contains_hallucinated_intent(output)
    if _input_signals_intent(norm_words):
        return intent_in_output
    return not intent_in_output


def evaluate(examples: List[Example], corrector: GrammarCorrector) -> List[CaseResult]:
    results: List[CaseResult] = []
    for ex in examples:
        detail = corrector.correct_detailed(ex.words)
        output = detail.sentence
        # Normalize once so the metric helpers see the same token shape the
        # runtime arbiter does.
        norm = normalize_words(ex.words)
        has_q = _has_question(ex.words) or "question" in ex.tags
        # Negation is expected only when the input actually carries a negation
        # token (yok/değil/hayır) or is explicitly tagged negative — NOT for the
        # "existential" tag alone, which also covers positive "var".
        has_neg = _has_negation(ex.words) or any(
            t in ex.tags for t in ("negation", "nominal_negation")
        )

        results.append(
            CaseResult(
                words=ex.words,
                expected=ex.expected,
                output=output,
                source=detail.source,
                tags=ex.tags,
                ml_latency_ms=detail.ml_latency_ms,
                exact=_normalize_sentence(output) == _normalize_sentence(ex.expected),
                root_preservation=_root_preservation_score(norm, output),
                intent_ok=_intent_ok(norm, output),
                question_ok=(output.strip().endswith("?") if has_q else None),
                negation_ok=(_negation_in_output(output) if has_neg else None),
            )
        )
    return results


def _rate(values: Iterable[Optional[bool]]) -> Optional[float]:
    applicable = [v for v in values if v is not None]
    if not applicable:
        return None
    return round(sum(1 for v in applicable if v) / len(applicable), 3)


def aggregate(results: List[CaseResult], use_ml: bool) -> Dict[str, object]:
    n = len(results)
    latencies = [r.ml_latency_ms for r in results if r.ml_latency_ms]
    report: Dict[str, object] = {
        "n": n,
        "exact_match": _rate([r.exact for r in results]),
        "mean_root_preservation": round(sum(r.root_preservation for r in results) / n, 3)
        if n
        else 0.0,
        "intent_ok": _rate([r.intent_ok for r in results]),
        "question_ok": _rate([r.question_ok for r in results]),
        "negation_ok": _rate([r.negation_ok for r in results]),
    }
    if use_ml:
        ml_used = [r for r in results if r.source.startswith("ml:")]
        report["ml_used"] = len(ml_used)
        report["fallback_rate"] = round(1 - len(ml_used) / n, 3) if n else 0.0
        report["avg_ml_latency_ms"] = (
            round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        )
    # Per-tag exact-match breakdown.
    tags = sorted({t for r in results for t in r.tags})
    per_tag: Dict[str, object] = {}
    for tag in tags:
        subset = [r for r in results if tag in r.tags]
        per_tag[tag] = {"n": len(subset), "exact_match": _rate([r.exact for r in subset])}
    report["per_tag"] = per_tag
    return report

def format_report(report: Dict[str, object], results: List[CaseResult]) -> str:
    lines = ["", "=== TİD → Türkçe eval ===", f"examples: {report['n']}"]
    for key in ("exact_match", "mean_root_preservation", "intent_ok", "question_ok", "negation_ok"):
        lines.append(f"  {key:<24} {report.get(key)}")
    if "fallback_rate" in report:
        lines.append(f"  {'ml_used':<24} {report['ml_used']}")
        lines.append(f"  {'fallback_rate':<24} {report['fallback_rate']}")
        lines.append(f"  {'avg_ml_latency_ms':<24} {report['avg_ml_latency_ms']}")
    lines.append("\nPer-tag exact_match:")
    per_tag = report["per_tag"]
    assert isinstance(per_tag, dict)
    for tag, info in per_tag.items():
        lines.append(f"  {tag:<20} n={info['n']:<3} exact={info['exact_match']}")
    mismatches = [r for r in results if not r.exact]
    if mismatches:
        lines.append(f"\nMismatches ({len(mismatches)}):")
        for r in mismatches:
            lines.append(
                f"  {r.words}\n    expected: {r.expected!r}\n    got     : {r.output!r}  [{r.source}]"
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TİD → Türkçe grammar eval")
    parser.add_argument(
        "--use-ml", action="store_true", help="enable the ML layer (needs HF token/model)"
    )
    parser.add_argument(
        "--model", default=None, help="model key (defaults to the ML default when --use-ml)"
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="prompt template version to test (see PROMPT_TEMPLATES; matters with --use-ml)",
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="path to a .jsonl eval set")
    parser.add_argument("--json", default=None, help="write the report as JSON to this path")
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N examples"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="CI gate: exit non-zero if exact_match falls below --min-exact",
    )
    parser.add_argument(
        "--min-exact", type=float, default=0.98, help="exact_match floor for --check"
    )
    args = parser.parse_args(argv)

    examples = load_dataset(Path(args.dataset))
    if args.limit is not None:
        examples = examples[: args.limit]

    cfg_kwargs = {"use_ml": args.use_ml}
    if args.model:
        cfg_kwargs["model_key"] = args.model
    if args.prompt_version:
        cfg_kwargs["prompt_version"] = args.prompt_version
    corrector = GrammarCorrector(GrammarConfig(**cfg_kwargs))

    t0 = time.monotonic()
    results = evaluate(examples, corrector)
    wall_s = time.monotonic() - t0

    report = aggregate(results, use_ml=args.use_ml)
    report["wall_seconds"] = round(wall_s, 2)
    print(format_report(report, results))

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote JSON report to {args.json}")

    if args.check:
        score = report["exact_match"] or 0.0
        passed = score >= args.min_exact
        print(
            f"\n[check] exact_match={score} min={args.min_exact} -> {'PASS' if passed else 'FAIL'}"
        )
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
