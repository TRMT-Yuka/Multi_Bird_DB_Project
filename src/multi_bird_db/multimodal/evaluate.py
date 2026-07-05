from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import MultimodalFeatureMatrix, MultimodalSampleRow


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """Metrics and per-row predictions for one split."""

    split_name: str
    row_count: int
    accuracy: float
    macro_f1: float
    support_by_label: dict[str, int]
    predictions: list[dict[str, str]]


def select_rows_by_qids(matrix: MultimodalFeatureMatrix, qids: set[str]) -> MultimodalFeatureMatrix:
    """Filter a feature matrix to rows whose QIDs belong to the requested split."""

    indices = [index for index, row in enumerate(matrix.rows) if row.qid in qids]
    selected_rows = [matrix.rows[index] for index in indices]
    if indices:
        selected_embeddings = matrix.embeddings[indices]
    else:
        selected_embeddings = np.zeros((0, matrix.embeddings.shape[1]), dtype=np.float32)
    metadata = dict(matrix.metadata)
    metadata['row_count'] = len(selected_rows)
    return MultimodalFeatureMatrix(rows=selected_rows, embeddings=selected_embeddings, metadata=metadata)


def labels_from_rows(rows: list[MultimodalSampleRow]) -> list[str]:
    return [row.target_label for row in rows]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _per_label_f1(true_labels: list[str], predicted_labels: list[str], classes: list[str]) -> dict[str, float]:
    f1_by_label: dict[str, float] = {}
    for label in classes:
        tp = sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=False) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=False) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=False) if truth == label and pred != label)
        precision = _safe_divide(float(tp), float(tp + fp))
        recall = _safe_divide(float(tp), float(tp + fn))
        f1_by_label[label] = _safe_divide(2.0 * precision * recall, precision + recall)
    return f1_by_label


def evaluate_predictions(
    *,
    split_name: str,
    rows: list[MultimodalSampleRow],
    predicted_labels: list[str],
    classes: list[str],
) -> SplitEvaluation:
    """Compute minimal split metrics and row-level prediction records."""

    true_labels = labels_from_rows(rows)
    if len(true_labels) != len(predicted_labels):
        raise ValueError('Prediction count must match row count.')
    accuracy = _safe_divide(
        float(sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=False) if truth == pred)),
        float(len(true_labels)),
    )
    support_by_label = {label: true_labels.count(label) for label in classes if label in true_labels}
    macro_f1 = 0.0
    if classes:
        per_label = _per_label_f1(true_labels, predicted_labels, classes)
        macro_f1 = float(sum(per_label.values()) / len(classes))
    predictions = [
        {
            'sample_id': row.sample_id,
            'qid': row.qid,
            'split': split_name,
            'target_rank': row.target_rank,
            'true_label': row.target_label,
            'predicted_label': predicted_label,
            'modality_pattern': row.modality_pattern,
        }
        for row, predicted_label in zip(rows, predicted_labels, strict=False)
    ]
    return SplitEvaluation(
        split_name=split_name,
        row_count=len(rows),
        accuracy=accuracy,
        macro_f1=macro_f1,
        support_by_label=support_by_label,
        predictions=predictions,
    )


def write_split_predictions_tsv(output_path: Path, evaluations: list[SplitEvaluation]) -> None:
    """Write a flat row-level prediction table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = [
            'sample_id',
            'qid',
            'split',
            'target_rank',
            'true_label',
            'predicted_label',
            'modality_pattern',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='	')
        writer.writeheader()
        for evaluation in evaluations:
            for row in evaluation.predictions:
                writer.writerow(row)


def write_metrics_tsv(output_path: Path, evaluations: list[SplitEvaluation]) -> None:
    """Write one metric row per split."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = ['split', 'row_count', 'accuracy', 'macro_f1']
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='	')
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow(
                {
                    'split': evaluation.split_name,
                    'row_count': evaluation.row_count,
                    'accuracy': f'{evaluation.accuracy:.6f}',
                    'macro_f1': f'{evaluation.macro_f1:.6f}',
                }
            )


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
