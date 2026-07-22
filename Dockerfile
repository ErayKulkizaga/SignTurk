FROM python:3.11-slim

WORKDIR /app

LABEL org.opencontainers.image.source="https://github.com/ErayKulkizaga/SignTurk" \
      org.opencontainers.image.title="SignTurk" \
      org.opencontainers.image.description="Real-time Turkish Sign Language research application"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model-assets.json .
COPY tools/download_models.py tools/download_models.py
COPY demo_assets_179/ demo_assets_179/
COPY model_assets/ model_assets/
ARG MODEL_BUNDLE=live
RUN python tools/download_models.py --bundle "$MODEL_BUNDLE"

COPY backend.py .
COPY database.py .
COPY models.py .
COPY live_pipeline.py .
COPY landmark_smoother.py .
COPY text_processing/ text_processing/
COPY frontend/ frontend/
COPY dataset/landmarks/ dataset/landmarks/

RUN useradd --create-home --uid 10001 sign-turk \
    && chown -R sign-turk:sign-turk /app
USER sign-turk

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
