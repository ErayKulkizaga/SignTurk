# text_processing

Turns a stream of recognized sign-gloss words into fluent, grammatical
Turkish (and optional speech). A deterministic **rule-based engine** is the
reliable primary; an optional **ML layer** (HF Inference API or local
transformers) proposes a candidate that is validated and arbitrated against
the rule-based output per request.

## Architecture

Dependencies point inward — pure domain at the centre, I/O at the edges.

```
buffer ─▶ pipeline ─▶ grammar ─▶ tts
                        │
        web (FastAPI) ──┘
```

### `grammar/` (was a single ~2000-line module)

| module          | responsibility                                            |
| --------------- | --------------------------------------------------------- |
| `linguistics`   | Turkish phonology & morphology — pure domain, no I/O      |
| `registry`      | `ModelSpec` / `MODEL_REGISTRY` / `GrammarConfig`          |
| `prompts`       | versioned zero-shot prompt templates                      |
| `validation`    | sanity checks on ML candidate sentences                   |
| `rules`         | `RuleBasedCorrector` — the primary engine                 |
| `adapters`      | model backends + `_classify_ml_error`                     |
| `arbiter`       | per-request ML-vs-rule quality comparator                 |
| `telemetry`     | in-process decision log (health endpoint)                 |
| `corrector`     | `GrammarCorrector` — the public hybrid facade             |

`grammar/__init__.py` re-exports the full public API, so
`from text_processing.grammar import …` is unchanged.

### `web/`

`schemas` (Pydantic) · `cache` (bounded LRU+TTL pipeline cache) ·
`jobs` (async job store + worker pool) · `router` (the `/api/text/*`
endpoints). Applications import the router directly from
`text_processing.web.router`.

## ML observability

ML failures are no longer swallowed into a bare fallback. Each failure is
classified (`quota` / `auth` / `not_served` / `rate_limit` / `timeout` /
`network` / `deps` …) and surfaced as an actionable Turkish message on
`GrammarResult.ml_error` → `CorrectResponse.ml_error`. `POST /api/text/ping`
probes a model's connectivity before a real correction.

## Conventions

- **Code and docstrings are written in English.**
- **User-facing strings are Turkish** — UI copy, model `notes`, and the
  `ml_error` / `ping` messages a user reads.
- The rule-based engine is the safe floor and the fallback for every ML
  error path; it is never removed.

## Validation

The deterministic engine can be validated without loading the CV models or
making network requests.

```bash
python -m unittest discover -s tests -v
python -m text_processing.eval --check --min-exact 0.98   # rule-based floor
python -m compileall text_processing
```

The repository CI runs these checks on every push and pull request.

For focused API development without TensorFlow, MediaPipe, or OpenCV:

```bash
python -m text_processing.devserver
# Open http://127.0.0.1:8000/docs
```
