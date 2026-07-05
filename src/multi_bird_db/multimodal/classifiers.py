from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SoftmaxClassifierConfig:
    """Configuration for the initial numpy-only softmax baseline."""

    learning_rate: float = 0.05
    num_epochs: int = 200
    l2_weight: float = 1e-4
    seed: int = 42
    fit_intercept: bool = True


@dataclass(frozen=True, slots=True)
class SoftmaxClassifierModel:
    """Trained multinomial linear classifier."""

    classes: list[str]
    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    training_trace: list[dict[str, float]]


def _require_2d_float_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {matrix.shape}")
    return matrix


def _require_1d_labels(label_indices: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_indices, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"Expected a 1D label vector, got shape {labels.shape}")
    return labels


def _standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
    normalized = ((features - mean) / scale).astype(np.float32, copy=False)
    return normalized, mean, scale


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted, dtype=np.float64)
    return (exp_shifted / exp_shifted.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)


def _cross_entropy(probabilities: np.ndarray, label_indices: np.ndarray) -> float:
    row_indices = np.arange(label_indices.shape[0], dtype=np.int64)
    clipped = np.clip(probabilities[row_indices, label_indices], 1e-8, 1.0)
    return float(-np.log(clipped, dtype=np.float64).mean())


def _encode_labels(labels: list[str]) -> tuple[np.ndarray, list[str], dict[str, int]]:
    classes = sorted({str(label) for label in labels})
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.asarray([class_to_index[str(label)] for label in labels], dtype=np.int64)
    return encoded, classes, class_to_index


def fit_softmax_classifier(
    features: np.ndarray,
    labels: list[str],
    *,
    validation_features: np.ndarray | None = None,
    validation_labels: list[str] | None = None,
    config: SoftmaxClassifierConfig | None = None,
) -> SoftmaxClassifierModel:
    """Fit a simple multinomial logistic-regression baseline with full-batch SGD."""

    resolved_config = config or SoftmaxClassifierConfig()
    train_x = _require_2d_float_matrix(features)
    train_y, classes, class_to_index = _encode_labels(labels)
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError('Feature rows and label count must match.')
    if len(classes) < 2:
        raise ValueError('At least two classes are required for classification.')

    normalized_train_x, feature_mean, feature_scale = _standardize_features(train_x)
    num_rows, feature_dim = normalized_train_x.shape
    num_classes = len(classes)

    rng = np.random.default_rng(resolved_config.seed)
    weights = rng.normal(loc=0.0, scale=0.01, size=(feature_dim, num_classes)).astype(np.float32)
    bias = np.zeros((num_classes,), dtype=np.float32)

    normalized_validation_x: np.ndarray | None = None
    validation_y: np.ndarray | None = None
    if validation_features is not None and validation_labels is not None:
        validation_matrix = _require_2d_float_matrix(validation_features)
        normalized_validation_x = ((validation_matrix - feature_mean) / feature_scale).astype(np.float32, copy=False)
        validation_y = _require_1d_labels(
            np.asarray([class_to_index[str(label)] for label in validation_labels], dtype=np.int64)
        )

    training_trace: list[dict[str, float]] = []
    targets = np.eye(num_classes, dtype=np.float32)[train_y]
    for epoch in range(resolved_config.num_epochs):
        logits = normalized_train_x @ weights
        if resolved_config.fit_intercept:
            logits = logits + bias
        probabilities = _softmax(logits)
        residual = (probabilities - targets) / float(num_rows)
        gradient_w = normalized_train_x.T @ residual
        if resolved_config.l2_weight > 0.0:
            gradient_w = gradient_w + (resolved_config.l2_weight * weights)
        weights = (weights - (resolved_config.learning_rate * gradient_w)).astype(np.float32, copy=False)
        if resolved_config.fit_intercept:
            gradient_b = residual.sum(axis=0)
            bias = (bias - (resolved_config.learning_rate * gradient_b)).astype(np.float32, copy=False)

        train_loss = _cross_entropy(probabilities, train_y)
        trace_item: dict[str, float] = {
            'epoch': float(epoch + 1),
            'train_loss': train_loss,
            'train_accuracy': float((probabilities.argmax(axis=1) == train_y).mean()),
        }
        if normalized_validation_x is not None and validation_y is not None:
            validation_logits = normalized_validation_x @ weights
            if resolved_config.fit_intercept:
                validation_logits = validation_logits + bias
            validation_probabilities = _softmax(validation_logits)
            trace_item['validation_loss'] = _cross_entropy(validation_probabilities, validation_y)
            trace_item['validation_accuracy'] = float((validation_probabilities.argmax(axis=1) == validation_y).mean())
        training_trace.append(trace_item)

    return SoftmaxClassifierModel(
        classes=classes,
        weights=weights,
        bias=bias,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        training_trace=training_trace,
    )


def predict_probabilities(model: SoftmaxClassifierModel, features: np.ndarray) -> np.ndarray:
    """Return class probabilities for each feature row."""

    matrix = _require_2d_float_matrix(features)
    normalized = ((matrix - model.feature_mean) / model.feature_scale).astype(np.float32, copy=False)
    logits = normalized @ model.weights + model.bias
    return _softmax(logits)


def predict_label_indices(model: SoftmaxClassifierModel, features: np.ndarray) -> np.ndarray:
    """Return argmax class indices for each feature row."""

    probabilities = predict_probabilities(model, features)
    return probabilities.argmax(axis=1).astype(np.int64, copy=False)


def predict_labels(model: SoftmaxClassifierModel, features: np.ndarray) -> list[str]:
    """Return predicted class labels for each feature row."""

    indices = predict_label_indices(model, features).tolist()
    return [model.classes[index] for index in indices]
