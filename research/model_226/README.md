# 226-Class Research Track

This directory keeps the higher-accuracy RTMW/RTMPose experiment separate from
the lightweight 179-class MediaPipe model used by the main live interface.

## Contents

- `MODEL_CARD.md` — intended use, metrics, data, limitations, and evidence status
- `config/` — exact four-stream composition and 226-class vocabulary
- `evaluation/` — machine-readable aggregate metrics and preserved error analysis
- `notebooks/` — output-free Colab notebooks retained as training provenance
- `../evaluate_predictions.py` — deterministic evaluation and confusion-matrix tool

The notebook filenames reflect the original project sequence. They use Google
Drive paths and are not the canonical quick-start path for the web application.

## Re-evaluate predictions

Given a probability array shaped `(N, 226)` and model-index labels shaped `(N,)`:

```bash
python research/evaluate_predictions.py \
  --probabilities val_or_test_probabilities.npy \
  --labels val_or_test_labels.npy \
  --output-dir evaluation/reproduced
```

The command writes `metrics.json`, `confusion_matrix.npy`,
`confusion_matrix.csv`, `confusion_matrix_normalized.png`,
`per_class_accuracy.csv`, `top_confusions.csv`, and `error_summary.png`.

No benchmark data is committed to this repository.
