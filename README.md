# SignTurk

Real-time Turkish Sign Language (TİD) recognition, sentence assembly, speech output, and 3D sign visualization in one FastAPI application.

> **Outstanding Graduation Project · 2025–2026**
> Eastern Mediterranean University, Department of Computer Engineering

<p align="center">
  <img src="docs/images/product-overview.png" alt="SignTurk product overview" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ErayKulkizaga/SignTurk/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ErayKulkizaga/SignTurk/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.18-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-black">
</p>

> **Project status:** maintained academic prototype and portfolio case study. The
> repository is runnable locally, but it is not presented as a safety-critical or
> production interpreting service.

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

Two recognition systems were developed and evaluated. Accuracy alone did not determine the live product choice: the inference contract also had to match the browser/WebSocket pipeline.

| Model | Input and architecture | Vocabulary | Top-1 | Top-5 | Macro-F1 | Role |
|---|---|---:|---:|---:|---:|---|
| **Live model** | 16 frames · 156 MediaPipe hand features · BiLSTM + temporal attention | 179 | **85.65%** | **95.59%** | — | Selected for the application because its lightweight feature contract integrates cleanly with real-time webcam inference |
| **Research model** | 32 RGB frames · RTMW/RTMPose whole-body landmarks · multi-stream temporal ensemble | 226 | **94.17%** | **99.49%** | **94.00%** | Higher-accuracy experiment; retained as the research track because it requires a different extractor and multi-stream runtime |

The live model also reached **93.96% Top-3** on its cross-subject evaluation.
All metrics above are offline evaluation results; the 226-class result is
reported separately so it is not mistaken for the model bundled with the live
MediaPipe pipeline.

### What is actually bundled

| Artifact | Included | Runtime purpose |
|---|:---:|---|
| `demo_assets_179/` | Yes | Current 179-class MediaPipe/BiLSTM live application |
| `model_assets/` | Yes | Legacy 184-class color/depth compatibility path used by `/ws` |
| 226-class RTMW research checkpoint | No | Research result documented above; its extractor and ensemble weights are not required by the live application |
| AUTSL videos | No | Governed by the dataset owner and intentionally not redistributed |

The legacy 184-class bundle is not one of the two headline evaluation tracks.
It remains only to keep the older color/depth WebSocket demonstration working.
The asset-contract tests verify each bundled model's class count, feature
dimension, label map, normalization statistics, and checkpoint presence.

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
frontend/                  Single-page product UI and Three.js avatar assets
demo_assets_179/           Bundled 179-class live model and preprocessing metadata
model_assets/              Legacy 184-class compatibility model for the /ws endpoint
dataset/landmarks/         Per-word landmark sequences for avatar playback
text_processing/           Turkish sentence, grammar, evaluation, and TTS modules
docs/images/               Product screenshots used in this README
tests/                     Lightweight grammar and model-asset contract checks
extract_landmarks.py       Offline AUTSL landmark extraction utility
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
python -m compileall backend.py database.py models.py text_processing
```

Runtime health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Scope and Limitations

- SignTurk recognizes **isolated signs**, not unrestricted continuous sign language.
- The 179-class live and 226-class research results belong to different input pipelines and must not be compared as drop-in replacements.
- Regional and signer variation can reduce recognition quality; predictions are not suitable for safety-critical communication.
- gTTS requires network access. The deterministic text path remains available when speech or optional ML services are unavailable.
- The platform is an academic prototype and does not replace a qualified interpreter.

## Dataset

Models were developed with [AUTSL](https://cvml.ankara.edu.tr/datasets/), a signer-independent Turkish Sign Language benchmark containing 226 isolated-sign classes. Dataset videos are not redistributed in this repository.

## Intentionally Excluded

- AUTSL source videos and generated training tensors
- The 226-class RTMW research ensemble checkpoint and extraction cache
- Local databases, generated speech, uploads, environment files, and editor state
- Experimental notebooks that are not part of the reproducible runtime

Keeping these artifacts out of the repository makes the difference between the
published application and the separate research track explicit.

## License

Source code in this repository is released under the [MIT License](LICENSE). Dataset, model, font, and third-party asset licenses remain with their respective owners.
