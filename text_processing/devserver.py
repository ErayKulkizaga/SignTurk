"""Standalone dev/test server for the text_processing pipeline.

Lets you test the text/LLM pipeline WITHOUT the heavy ML/CV stack (no
tensorflow / mediapipe / opencv / torch and no webcam):

    pip install fastapi uvicorn
    python -m text_processing.devserver
    # or: uvicorn text_processing.devserver:app --reload --port 8000
    # then open  http://127.0.0.1:8000

Optional extras (only if you want to exercise those paths):
    pip install gtts                 # hear audio (synthesize_audio / Ses üret)
    pip install huggingface_hub      # + export HF_TOKEN  -> test cloud ML (Qwen)

It serves the dedicated test UI at "/" and mounts the real
``/api/text/*`` router, so what you test here is exactly what runs in
production.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from text_processing_routes import router

app = FastAPI(title="SignAI · text_processing test server")
app.include_router(router)

_TEST_UI = Path(__file__).resolve().parent.parent / "frontend" / "text_processing_test.html"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if _TEST_UI.is_file():
        return HTMLResponse(_TEST_UI.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>text_processing_test.html bulunamadı</h1>"
        "<p>frontend/text_processing_test.html dosyasının yerinde olduğundan emin ol.</p>",
        status_code=404,
    )


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  SignAI text_processing test → http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
