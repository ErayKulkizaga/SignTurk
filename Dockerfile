FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend.py .
COPY database.py .
COPY models.py .
COPY landmark_smoother.py .
COPY text_processing/ text_processing/
COPY frontend/ frontend/
COPY demo_assets_179/ demo_assets_179/
COPY model_assets/ model_assets/
COPY dataset/landmarks/ dataset/landmarks/

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
