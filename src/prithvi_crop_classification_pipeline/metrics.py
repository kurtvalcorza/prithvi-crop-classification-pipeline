"""Pixel-level multi-class segmentation metrics for the crop-classification tutorial and its constant baseline.

All numbers are pooled over the labelled pixels of the chips scored together (the ignore index excluded); nothing
here estimates dispersion. Mean IoU and mean class accuracy average over the classes that occur in the labels or
the predictions of the scored set (mmseg's `nanmean` convention), so a class absent from a small sample does not
pull the mean to zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def confusion_matrix(predictions: Sequence[Any], labels: Sequence[Any], *, num_classes: int, ignore_index: int) -> Any:
    """(num_classes, num_classes) int64 matrix, rows = label, columns = prediction, ignored pixels excluded."""
    import numpy as np

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(predictions, labels, strict=True):
        pred = np.asarray(pred).reshape(-1).astype(np.int64)
        label = np.asarray(label).reshape(-1).astype(np.int64)
        if pred.shape != label.shape:
            raise ValueError(f"prediction shape {pred.shape} != label shape {label.shape}")
        valid = label != ignore_index
        if valid.any():
            index = label[valid] * num_classes + pred[valid]
            matrix += np.bincount(index, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return matrix


def metrics_from_confusion(matrix: Any, class_names: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.int64)
    total = int(matrix.sum())
    tp = np.diag(matrix).astype(np.float64)
    label_count = matrix.sum(axis=1).astype(np.float64)
    pred_count = matrix.sum(axis=0).astype(np.float64)
    union = label_count + pred_count - tp
    present = union > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(present, tp / np.where(union > 0, union, 1), np.nan)
        recall = np.where(label_count > 0, tp / np.where(label_count > 0, label_count, 1), np.nan)
        precision = np.where(pred_count > 0, tp / np.where(pred_count > 0, pred_count, 1), np.nan)
        # F1 is defined for every class that occurs in the labels or the predictions; a class the model never
        # predicts (precision undefined) or that never occurs (recall undefined) scores 0 on the missing side.
        p0, r0 = np.nan_to_num(precision, nan=0.0), np.nan_to_num(recall, nan=0.0)
        f1 = np.where(present, 2 * p0 * r0 / np.where((p0 + r0) > 0, p0 + r0, 1), np.nan)
    return {
        "pixels": total,
        "accuracy": round(float(tp.sum() / total), 4) if total else None,
        "mean_iou": round(float(np.nanmean(iou)), 4) if present.any() else None,
        "mean_accuracy": round(float(np.nanmean(recall)), 4) if (label_count > 0).any() else None,
        "mean_f1": round(float(np.nanmean(f1)), 4) if present.any() else None,
        "iou": {name: (round(float(v), 4) if np.isfinite(v) else None) for name, v in zip(class_names, iou, strict=True)},
        "recall": {name: (round(float(v), 4) if np.isfinite(v) else None) for name, v in zip(class_names, recall, strict=True)},
        "label_fraction": {
            name: round(float(v / total), 4) if total else None for name, v in zip(class_names, label_count, strict=True)
        },
        "classes_scored": int(present.sum()),
    }


def segmentation_metrics(
    predictions: Sequence[Any], labels: Sequence[Any], *, class_names: Sequence[str], ignore_index: int
) -> dict[str, Any]:
    """Per-class IoU / recall, mean IoU, mean class accuracy, mean F1 and overall pixel accuracy."""
    matrix = confusion_matrix(predictions, labels, num_classes=len(class_names), ignore_index=ignore_index)
    return metrics_from_confusion(matrix, class_names)


def majority_baseline(labels: Sequence[Any], *, class_names: Sequence[str], ignore_index: int) -> dict[str, Any]:
    """The constant predictor that names every pixel with the most frequent class of the scored labels — the best
    any constant map can do on these pixels — scored on the same pixels as the model."""
    import numpy as np

    counts = np.zeros(len(class_names), dtype=np.int64)
    for label in labels:
        label = np.asarray(label).reshape(-1).astype(np.int64)
        valid = label[label != ignore_index]
        counts += np.bincount(valid, minlength=len(class_names))[: len(class_names)]
    majority = int(counts.argmax())
    predictions = [np.full(np.asarray(label).shape, majority, dtype=np.int64) for label in labels]
    report = segmentation_metrics(predictions, labels, class_names=class_names, ignore_index=ignore_index)
    report["majority_class"] = class_names[majority]
    return report
