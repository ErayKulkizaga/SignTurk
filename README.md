---
title: SignTurk
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Real-time Turkish Sign Language recognition and translation
---

# SignTurk

Real-time Turkish Sign Language (TİD) recognition, sentence assembly, speech output, and 3D sign visualization in one FastAPI application.

> **Outstanding Graduation Project · 2025–2026**
> Eastern Mediterranean University, Department of Computer Engineering

## See SignTurk in Motion

Perform a TİD sign, watch the 32-frame buffer fill, and see the live prediction
settle in real time. Click any preview to play the full recording.

<table>
  <tr>
    <td width="50%" align="center">
      <a href="docs/videos/cay.mp4?raw=1">
        <img src="docs/images/signs/cay-preview.gif" alt="SignTurk recognizing the Turkish Sign Language sign for çay (tea)" width="100%">
      </a>
      <br><strong>ÇAY</strong> · tea<br>
      <sub><a href="docs/videos/cay.mp4?raw=1">▶ Play the full video</a></sub>
    </td>
    <td width="50%" align="center">
      <a href="docs/videos/kardes.mp4?raw=1">
        <img src="docs/images/signs/kardes-preview.gif" alt="SignTurk recognizing the Turkish Sign Language sign for kardeş (sibling)" width="100%">
      </a>
      <br><strong>KARDEŞ</strong> · sibling<br>
      <sub><a href="docs/videos/kardes.mp4?raw=1">▶ Play the full video</a></sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="docs/videos/degistirmek.mp4?raw=1">
        <img src="docs/images/signs/degistirmek-preview.gif" alt="SignTurk recognizing the Turkish Sign Language sign for değiştirmek (change)" width="100%">
      </a>
      <br><strong>DEĞİŞTİRMEK</strong> · change<br>
      <sub><a href="docs/videos/degistirmek.mp4?raw=1">▶ Play the full video</a></sub>
    </td>
    <td width="50%" align="center">
      <a href="docs/videos/kopek.mp4?raw=1">
        <img src="docs/images/signs/kopek-preview.gif" alt="SignTurk recognizing the Turkish Sign Language sign for köpek (dog)" width="100%">
      </a>
      <br><strong>KÖPEK</strong> · dog<br>
      <sub><a href="docs/videos/kopek.mp4?raw=1">▶ Play the full video</a></sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="docs/videos/kopru.mp4?raw=1">
        <img src="docs/images/signs/kopru-preview.gif" alt="SignTurk recognizing the Turkish Sign Language sign for köprü (bridge)" width="100%">
      </a>
      <br><strong>KÖPRÜ</strong> · bridge<br>
      <sub><a href="docs/videos/kopru.mp4?raw=1">▶ Play the full video</a></sub>
    </td>
    <td width="50%" align="center">
      <a href="docs/videos/ben.mp4?raw=1">
        <img src="docs/images/signs/ben-preview.gif" alt="SignTurk recognizing the Turkish Sign Language sign for ben (I/me)" width="100%">
      </a>
      <br><strong>BEN</strong> · I / me<br>
      <sub><a href="docs/videos/ben.mp4?raw=1">▶ Play the full video</a></sub>
    </td>
  </tr>
</table>

<p align="center">
  <img src="docs/images/product-overview.png" alt="SignTurk product overview" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ErayKulkizaga/SignTurk/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ErayKulkizaga/SignTurk/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/ErayKulkizaga/SignTurk/actions/workflows/docker.yml"><img alt="Docker build" src="https://github.com/ErayKulkizaga/SignTurk/actions/workflows/docker.yml/badge.svg"></a>
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.18-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-black">
</p>

> **Project status:** maintained academic prototype and portfolio case study. The
> repository is runnable locally, but it is not presented as a safety-critical or
> production interpreting service.

## Contribution Scope

SignTurk was co-developed as a graduation-project team effort. Eray Kulkizaga led
the machine-learning work across model design, training, and evaluation, and
implemented nearly all of the FastAPI backend, including real-time inference,
WebSocket communication, authentication, persistence, and operational APIs.
Interface, data preparation, integration, and 3D animation work was shared across
the team.

## Why SignTurk

SignTurk turns isolated TİD signs captured by a standard RGB webcam into approved words and readable Turkish sentences. The same platform can replay supported words on a Three.js avatar and optionally synthesize the assembled sentence through gTTS.

- **Live recognition:** MediaPipe hand landmarks streamed to FastAPI over WebSocket
- **Human-in-the-loop translation:** predictions enter the sentence only after user approval
- **Turkish sentence engine:** deterministic morphology and grammar rules, with an optional ML-assisted path
- **Speech output:** opt-in gTTS with an optional offline Piper adapter
- **3D visualization:** 226 landmark sequences mapped to a browser avatar
- **Persistence:** SQLAlchemy with PostgreSQL/Supabase support and a zero-config SQLite fallback
- **Operations UI:** account access, prediction history, settings, model status, and administrator views

## Model Strategy

Two recognition pipelines and multiple ensemble configurations were evaluated.
Accuracy alone did not determine the live product choice: the extractor and
feature contract also had to fit the browser/WebSocket pipeline.

| Evaluation | Input and architecture | Vocabulary | Top-1 | Top-3 | Top-5 | Macro-F1 | Evidence status |
|---|---|---:|---:|---:|---:|---:|---|
| **Live application** | 16 frames · 156 MediaPipe hand features · BiLSTM + attention | 179 | **85.65%** | **93.96%** | **95.59%** | — | Bundled application model |
| **Final research ensemble** | 32 RGB frames · RTMW/RTMPose · validation-selected four-stream ensemble | 226 | **94.17%** | **98.88%** | **99.49%** | **93.99%** | Final held-out test evaluation with weights frozen after validation |

All metrics are offline results. The live and research rows use different
extractors, feature contracts, sequence lengths, and vocabularies.
See the [226-class model card](research/model_226/MODEL_CARD.md) for the exact
split, weights, limitations, and provenance.

### What is actually bundled

| Artifact | Included | Runtime purpose |
|---|:---:|---|
| `demo_assets_179/` | Metadata in Git; weight in Release | Current 179-class MediaPipe/BiLSTM live application |
| `model_assets/` | Metadata in Git; weight in Release | Legacy 184-class color/depth compatibility path used by `/ws` |
| `research/model_226/` | Config, model card, notebooks, and evaluation evidence in Git; weights in Release | Separate four-stream research runtime |
| AUTSL videos | No | Governed by the dataset owner and intentionally not redistributed |

The legacy 184-class bundle is not one of the two headline evaluation tracks.
It remains only to keep the older color/depth WebSocket demonstration working.
`tools/download_models.py` retrieves versioned weights and verifies both their
size and SHA-256 digest. Asset-contract tests validate the manifests, class
counts, feature dimensions, label maps, and ensemble configuration without
downloading hundreds of megabytes in CI.

### Live preprocessing

```text
RGB webcam frame
  → MediaPipe Hands (21 × 3 coordinates × 2 hands)
  → wrist-relative, scale-normalized coordinates (126 features)
  → finger-joint angles (+30 features)
  → Z-score normalization
  → 16 × 156 sequence
  → BiLSTM + temporal attention
  → Top-k sign predictions
```

### Research pipeline

```text
32 RGB frames
  → RTMW / RTMPose WholeBody
  → pose + left hand + right hand landmarks
  → joint, bone, motion, and geometric streams
  → recurrent temporal models + attention
  → validation-selected probability ensemble
  → 226-class prediction
```

<p align="center">
  <img src="research/model_226/evaluation/four_stream_error_summary.png" alt="Four-stream 226-class per-class accuracy distribution and most frequent confusion pairs" width="100%">
</p>

The complete confusion-matrix generator is included at
`research/evaluate_predictions.py`. The preserved chart is explicitly an error
summary; regenerating the full 226×226 matrix requires the original probability
and label arrays or a fresh official-test-split run.

## Product Tour

### Live translation

<p align="center">
  <img src="docs/images/live-translation.png" alt="SignTurk live translation workspace" width="100%">
</p>

Camera frames are buffered into 16-frame windows. A confident prediction becomes a pending candidate; the user approves or rejects it before it is persisted or added to the sentence.

### Recognition feedback

<p align="center">
  <img src="docs/images/recognition-result.png" alt="SignTurk recognition result and confidence" width="100%">
</p>

### 3D avatar

<p align="center">
  <img src="docs/images/avatar-animation.png" alt="SignTurk 3D avatar animation" width="49%">
  <img src="docs/images/avatar-studio.png" alt="SignTurk avatar studio" width="49%">
</p>

The submitted interface used **TSL Nexus** as an internal UI codename. The public project and current source use the final name **SignTurk**; the screenshots are retained as an authentic record of the evaluated graduation-project build.

## Architecture

```text
Browser UI
  ├─ camera frames ───────────────┐
  ├─ approval / rejection actions │
  └─ avatar and TTS requests      │
                                  ▼
FastAPI
  ├─ WebSocket inference ── MediaPipe ── TensorFlow/Keras
  ├─ sentence engine ────── rule-based Turkish grammar
  ├─ optional speech ────── gTTS / Piper
  ├─ avatar API ─────────── smoothed landmark sequences
  └─ account/history API ── SQLAlchemy ── SQLite or PostgreSQL
```

## Repository Layout

```text
backend.py                 FastAPI app, WebSockets, auth, inference, and APIs
database.py                Environment-based database config with SQLite fallback
models.py                  SQLAlchemy models
live_pipeline.py           Pure 179/184-class preprocessing helpers
frontend/                  Product UI modules and Three.js avatar assets
demo_assets_179/           179-class live preprocessing metadata
model_assets/              Legacy 184-class compatibility metadata
model-assets.json          Versioned model URLs, sizes, destinations, and SHA-256 hashes
signturk_runtime/          Modular 226-class feature, extractor, ensemble, and stream runtime
research/model_226/        Model card, configuration, notebooks, and evaluation evidence
research/evaluate_predictions.py  Full metrics and confusion-matrix generator
dataset/landmarks/         Per-word landmark sequences for avatar playback
text_processing/           Turkish sentence, grammar, evaluation, and TTS modules
docs/images/               Product screenshots used in this README
docs/images/signs/         Animated previews and thumbnails for the sign video gallery
docs/videos/               Compressed H.264 product demonstration videos
tests/                     Lightweight grammar and model-asset contract checks
extract_landmarks.py       Offline AUTSL landmark extraction utility
tools/download_models.py   Checksum-verified model asset installer
```

## Quick Start

Python **3.11** is recommended.

```bash
git clone https://github.com/ErayKulkizaga/SignTurk.git
cd SignTurk
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
python tools/download_models.py --bundle live
uvicorn backend:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Configuration

The application works locally without external infrastructure: when `DATABASE_URL` is empty, it creates `local_dev.db` with SQLite.

```bash
cp .env.example .env
```

For PostgreSQL/Supabase, set a SQLAlchemy-compatible connection string in `.env`. Credentials are never stored in source control.

```env
DATABASE_URL=postgresql+psycopg://user:password@host:5432/postgres
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```

To create the first administrator, provide `SIGNTURK_ADMIN_EMAIL` and an 8+ character `SIGNTURK_ADMIN_PASSWORD` before the first startup. There are no hard-coded public demo credentials.

### Docker

```bash
docker compose up --build
```

The Docker image downloads and verifies the versioned live weights during the
build; no host-side model volume is required.

## Main API Surface

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Model, MediaPipe, database, and UI readiness |
| `POST` | `/api/auth/register` | Create a user account |
| `POST` | `/api/auth/login` | Receive a bearer access token |
| `WS` | `/api/predict/live` | Authenticated live sign inference and approval flow |
| `POST` | `/api/predict/sequence` | One-shot 16-frame prediction |
| `GET` | `/api/history` | Authenticated prediction history |
| `GET/POST` | `/api/settings` | Authenticated runtime settings |
| `GET` | `/api/dictionary` | Sign label registry |
| `GET` | `/signs` | Available avatar words |
| `GET` | `/landmark/{word}` | Smoothed animation landmarks |
| `POST` | `/api/text/correct` | Turkish sentence correction and optional speech |

FastAPI also exposes interactive API documentation at `/docs` while the server is running.

The older `/ws` endpoint is retained for the 184-class color/depth compatibility
demo. New integrations should use `/api/predict/live`.

## Security and Publication Hygiene

- Database credentials and optional API tokens are read only from environment variables.
- Passwords are hashed with PBKDF2-SHA256.
- History, settings, model telemetry, and administrator endpoints require bearer authentication.
- Administrator-only endpoints verify the stored user role server-side.
- CORS defaults to the two local development origins and can be overridden explicitly.
- No default password, local database, generated speech, or `.env` file is committed.

The in-memory bearer-token store is appropriate for the local graduation-project runtime. A multi-instance production deployment should replace it with expiring, signed tokens or a centralized session store.

## Validation

Run the lightweight sentence-engine checks without loading TensorFlow:

```bash
python -m unittest discover -s tests -v
python -m text_processing.eval --check --min-exact 0.98
python -m compileall backend.py database.py models.py live_pipeline.py text_processing signturk_runtime research tools
```

Runtime health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Scope and Limitations

- SignTurk recognizes **isolated signs**, not unrestricted continuous sign language.
- The 179-class live and 226-class research results belong to different input pipelines and must not be compared as drop-in replacements.
- The final four-stream research ensemble achieved 94.17% Top-1 on the official held-out test split after its weights were selected on validation and frozen before test evaluation.
- Regional and signer variation can reduce recognition quality; predictions are not suitable for safety-critical communication.
- gTTS requires network access. The deterministic text path remains available when speech or optional ML services are unavailable.
- The platform is an academic prototype and does not replace a qualified interpreter.

## Dataset

Models were developed with [AUTSL](https://cvml.ankara.edu.tr/datasets/), a signer-independent Turkish Sign Language benchmark containing 226 isolated-sign classes. Dataset videos are not redistributed in this repository.

## Intentionally Excluded

- AUTSL source videos and generated training tensors
- AUTSL-derived feature tensors, extraction caches, and raw per-sample prediction exports
- Local databases, generated speech, uploads, environment files, and editor state
- Third-party RTMW/YOLOX checkpoints in Git history (the verified downloader fetches them from their upstream hosts)

Keeping these artifacts out of the repository makes the difference between the
published application and the separate research track explicit.

## License

Source code in this repository is released under the [MIT License](LICENSE).
Dataset, model, font, and asset terms remain with their respective owners; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md).
