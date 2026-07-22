"""Evaluate 226-class prediction exports and generate confusion artifacts.

Labels and probability columns must use the contiguous model index (0..225).
The vocabulary files map those indices back to original AUTSL class IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "research" / "model_226" / "config"


def load_vocabulary(label_map_path: Path, display_names_path: Path) -> list[dict[str, object]]:
    label_to_index = json.loads(label_map_path.read_text(encoding="utf-8"))
    display_names = json.loads(display_names_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object] | None] = [None] * len(label_to_index)
    for original_id, model_index in label_to_index.items():
        rows[int(model_index)] = {
            "model_index": int(model_index),
            "original_class_id": int(original_id),
            "label": display_names.get(str(original_id), str(original_id)),
        }
    if any(row is None for row in rows):
        raise ValueError("label map does not define a contiguous model index")
    return [row for row in rows if row is not None]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    per_class = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator != 0,
    )
    return float(per_class.mean())


def top_k_accuracy(probabilities: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.any(top == y_true[:, None], axis=1).mean())


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    output_dir: Path,
    matrix: np.ndarray,
    per_class_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    top1: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plot generation requires matplotlib") from exc

    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        support,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=support != 0,
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set(title="SignTurk 226-class normalized confusion matrix", xlabel="Predicted class", ylabel="True class")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, label="Row-normalized rate", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_normalized.png", dpi=180)
    plt.close(fig)

    accuracies = np.array([float(row["accuracy"]) * 100 for row in per_class_rows])
    weakest = min(per_class_rows, key=lambda row: float(row["accuracy"]))
    pairs = confusion_rows[:15]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(accuracies, bins=np.arange(0, 102, 5), color="#4c78a8", edgecolor="white")
    axes[0].axvline(float(weakest["accuracy"]) * 100, color="#e45756", linestyle="--")
    axes[0].set(title="Per-class accuracy distribution", xlabel="Accuracy (%)", ylabel="Class count")
    labels = [f"{row['true_original_class_id']} → {row['pred_original_class_id']}" for row in reversed(pairs)]
    counts = [int(row["count"]) for row in reversed(pairs)]
    axes[1].barh(labels, counts, color="#f58518")
    axes[1].set(title="Most frequent confusion pairs", xlabel="Count")
    fig.suptitle(
        f"Test Top-1: {top1 * 100:.2f}% | Weakest: "
        f"{weakest['original_class_id']} ({float(weakest['accuracy']) * 100:.1f}%)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "error_summary.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", type=Path, required=True, help="N×226 .npy probability array")
    parser.add_argument("--labels", type=Path, required=True, help="N-element .npy model-index labels")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_CONFIG / "labels" / "label_map.json")
    parser.add_argument(
        "--display-names", type=Path, default=DEFAULT_CONFIG / "labels" / "class_display_names.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probabilities = np.asarray(np.load(args.probabilities), dtype=np.float64)
    y_true = np.asarray(np.load(args.labels), dtype=np.int64).reshape(-1)
    vocabulary = load_vocabulary(args.label_map, args.display_names)
    class_count = len(vocabulary)
    if probabilities.shape != (len(y_true), class_count):
        raise ValueError(
            f"expected probabilities shape ({len(y_true)}, {class_count}), got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if np.any((y_true < 0) | (y_true >= class_count)):
        raise ValueError("labels contain an out-of-range model index")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, class_count)
    top1 = float((y_pred == y_true).mean())
    metrics = {
        "samples": int(len(y_true)),
        "classes": class_count,
        "top_1_accuracy": top1,
        "top_3_accuracy": top_k_accuracy(probabilities, y_true, 3),
        "top_5_accuracy": top_k_accuracy(probabilities, y_true, 5),
        "macro_f1": macro_f1(matrix),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.save(args.output_dir / "confusion_matrix.npy", matrix)
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")

    supports = matrix.sum(axis=1)
    correct = np.diag(matrix)
    per_class_rows: list[dict[str, object]] = []
    for meta, count, hit in zip(vocabulary, supports, correct):
        per_class_rows.append(
            {
                **meta,
                "support": int(count),
                "correct": int(hit),
                "accuracy": float(hit / count) if count else 0.0,
            }
        )
    write_csv(
        args.output_dir / "per_class_accuracy.csv",
        ["model_index", "original_class_id", "label", "support", "correct", "accuracy"],
        per_class_rows,
    )

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    confusion_rows: list[dict[str, object]] = []
    for true_index, pred_index in np.argwhere(off_diagonal > 0):
        true_meta = vocabulary[int(true_index)]
        pred_meta = vocabulary[int(pred_index)]
        confusion_rows.append(
            {
                "true_model_index": int(true_index),
                "true_original_class_id": true_meta["original_class_id"],
                "true_label": true_meta["label"],
                "pred_model_index": int(pred_index),
                "pred_original_class_id": pred_meta["original_class_id"],
                "pred_label": pred_meta["label"],
                "count": int(off_diagonal[true_index, pred_index]),
            }
        )
    confusion_rows.sort(key=lambda row: (-int(row["count"]), int(row["true_model_index"])))
    write_csv(
        args.output_dir / "top_confusions.csv",
        [
            "true_model_index",
            "true_original_class_id",
            "true_label",
            "pred_model_index",
            "pred_original_class_id",
            "pred_label",
            "count",
        ],
        confusion_rows,
    )
    save_plots(args.output_dir, matrix, per_class_rows, confusion_rows, top1)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
