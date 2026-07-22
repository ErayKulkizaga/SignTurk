# SignTurk 226-Class Ensemble Model Card

## Model summary

This research track classifies an isolated Turkish Sign Language sign from a
32-frame RGB sequence. RTMW/RTMPose WholeBody converts each frame to a
75-landmark representation. Joint, bone, temporal-motion, geometric, and
hand-only features are then fused by four recurrent temporal classifiers.

The model is an academic prototype. It is not a continuous-sign-language
translator and must not be used for emergency, medical, legal, or other
safety-critical communication.

## Evaluation records

Two results are deliberately kept separate:

| Evaluation | Streams | Top-1 | Top-3 | Top-5 | Macro-F1 | Evidence status |
|---|---:|---:|---:|---:|---:|---|
| Audited report result | 3 | 94.09% | 98.69% | 99.33% | 0.9391 | Validation-selected ensemble independently reproduced for the final report |
| Four-stream runtime record | 4 | 94.17% | 98.88% | 99.49% | 0.9399 | Aggregate metrics and error summary preserved locally; raw per-sample exports are not in this repository |

Both rows use the official AUTSL held-out test split of 3,742 samples. The
four-stream value must not be described as independently reproduced until the
prediction export is recovered or the test split is rerun. Machine-readable
values are in [`evaluation/four_stream_metrics.json`](evaluation/four_stream_metrics.json).

![Four-stream per-class accuracy and most frequent confusion pairs](evaluation/four_stream_error_summary.png)

The weakest recorded class was original class 82 at 45.5% accuracy. The most
frequent error was class 182 predicted as class 203 nine times. The preserved
top pairs are in
[`evaluation/four_stream_top_confusions.csv`](evaluation/four_stream_top_confusions.csv).
This chart is an error-analysis summary, not a replacement for the full
226×226 matrix.

## Data and splits

- Dataset: AUTSL, 226 isolated signs, RGB input only
- Train: 28,142 samples
- Validation: 4,418 samples
- Test: 3,742 samples
- Split policy: the official signer-independent CSV splits supplied with AUTSL
- Vocabulary: `config/labels/`
- Random seed in the archived final training notebook: 42

AUTSL videos and generated feature tensors are intentionally excluded. Obtain
the dataset from its owner and comply with its terms.

## Inputs and preprocessing

1. Uniformly sample or pad a clip to 32 RGB frames.
2. Run RTMW/RTMPose WholeBody and select the primary person.
3. Map pose and hand detections to 75 landmarks with `(x, y, z, confidence)`.
4. Build joint `(32, 300)`, bone `(32, 144)`, joint-motion `(32, 300)`,
   bone-motion `(32, 144)`, and extra-geometry `(32, 39)` streams.
5. Build the two 168-dimensional hand streams plus the geometric stream.
6. Fuse model probability vectors with weights `0.10 / 0.21 / 0.55 / 0.14`.

The source implementation lives in `signturk_runtime/`; configuration is in
`config/backend_config.json`.

## Intended use

- Research on isolated-sign recognition from standard RGB video
- Reproducing the project evaluation with the official AUTSL split
- Comparing landmark and temporal-feature approaches
- Portfolio and educational demonstration

## Limitations and risks

- Isolated signs only; no continuous signing, co-articulation, or discourse context
- Accuracy varies by class, signer, camera framing, lighting, and pose quality
- Similar motion patterns remain confusable and low-frequency classes can be weaker
- AUTSL is a benchmark dataset and does not establish performance for every TİD user
- Offline benchmark accuracy does not equal live webcam accuracy
- The exact four-stream result still needs independent re-verification from raw predictions

## Reproducibility

Download model and extractor files with:

```bash
python tools/download_models.py --bundle research
pip install -r requirements-research.txt
```

The archived notebooks document the original Colab workflow. They intentionally
retain their Drive-oriented paths as provenance; configure those paths before
running. Use `research/evaluate_predictions.py` to regenerate metrics, the full
confusion matrix, per-class accuracy, and confusion-pair plots from recovered
probabilities and labels.

## Versioning

- Asset release: `model-assets-v1`
- Model format: TensorFlow/Keras
- Research runtime: 32-frame RTMW/RTMPose feature contract
- Published name: SignTurk (earlier internal material may say SignaTurk or TSL Nexus)
