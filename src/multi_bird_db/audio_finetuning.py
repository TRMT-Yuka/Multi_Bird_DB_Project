from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .audio_embeddings import (
    DEFAULT_CACHE_DIR,
    DEFAULT_EXTENSIONS,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TARGET_SAMPLE_RATE,
    _finish_progress_line,
    _render_progress_line,
    _timestamp_utc,
    _write_json_atomic,
    _write_tsv_atomic,
    discover_audio_files,
    infer_qid,
    load_audio_file,
    resolve_torch_device,
)
from .config import get_project_paths

DEFAULT_OUTPUT_DIR = get_project_paths().root / "data" / "external" / "models" / "audio" / "wav2vec2-finetuned"
DEFAULT_NUM_FOLDS = 5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 4
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_SEED = 42
DEFAULT_RECORDING_MAP_PATH = get_project_paths().xeno_canto_recording_map_json
DEFAULT_AUDIO_REPAIR_REPORT_PATH = DEFAULT_OUTPUT_DIR / "xeno_canto_audio_repair.tsv"
ALLOWED_REPAIR_STATUSES = {"valid", "repaired"}

MANIFEST_COLUMNS = [
    "qid",
    "recording_id",
    "audio_path",
    "relative_path",
    "fold_index",
    "train_only_all_folds",
    "xeno_canto_species_id",
    "download_url",
]


@dataclass(frozen=True)
class FineTuneExample:
    qid: str
    recording_id: str
    audio_path: str
    relative_path: str
    xeno_canto_species_id: str
    download_url: str
    fold_index: int | None
    train_only_all_folds: bool


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_examples: list[FineTuneExample]
    test_examples: list[FineTuneExample]
    singleton_train_examples: list[FineTuneExample]


class AudioClassificationCollator:
    def __init__(
        self,
        *,
        feature_extractor: Any,
        label2id: dict[str, int],
        sample_rate: int,
        max_seconds: float,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds

    def __call__(self, batch: list[FineTuneExample]) -> dict[str, Any]:
        import torch

        waveforms = []
        labels = []
        for item in batch:
            waveform, _ = load_audio_file(
                Path(item.audio_path),
                target_sample_rate=self.sample_rate,
                max_seconds=self.max_seconds,
            )
            waveforms.append(np.asarray(waveform, dtype=np.float32))
            labels.append(self.label2id[item.qid])
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs


def _write_manifest(path: Path, rows: list[FineTuneExample]) -> None:
    payload = [
        {
            "qid": row.qid,
            "recording_id": row.recording_id,
            "audio_path": row.audio_path,
            "relative_path": row.relative_path,
            "fold_index": "" if row.fold_index is None else str(row.fold_index),
            "train_only_all_folds": "1" if row.train_only_all_folds else "0",
            "xeno_canto_species_id": row.xeno_canto_species_id,
            "download_url": row.download_url,
        }
        for row in rows
    ]
    _write_tsv_atomic(path, payload, MANIFEST_COLUMNS)


def _load_recording_index(recording_map_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not recording_map_path.exists():
        return {}
    payload = json.loads(recording_map_path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], dict[str, str]] = {}
    if not isinstance(payload, list):
        return index
    for row in payload:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip()
        recording_ids = row.get("recording_ids", [])
        download_urls = row.get("download_urls", [])
        species_id = str(row.get("xeno_canto_species_id", "")).strip()
        if not qid or not isinstance(recording_ids, list):
            continue
        for offset, recording_id in enumerate(recording_ids):
            normalized = str(recording_id).strip()
            if not normalized:
                continue
            download_url = ""
            if isinstance(download_urls, list) and offset < len(download_urls):
                download_url = str(download_urls[offset])
            index[(qid, normalized)] = {
                "xeno_canto_species_id": species_id,
                "download_url": download_url,
            }
    return index


def build_finetune_examples(
    input_dir: Path,
    recording_map_path: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> list[FineTuneExample]:
    recording_index = _load_recording_index(recording_map_path)
    examples: list[FineTuneExample] = []
    for path in discover_audio_files(input_dir, extensions=extensions):
        qid = infer_qid(path, input_dir)
        recording_id = path.stem
        metadata = recording_index.get((qid, recording_id), {})
        relative_path = str(path.relative_to(input_dir)) if path.is_relative_to(input_dir) else str(path)
        examples.append(
            FineTuneExample(
                qid=qid,
                recording_id=recording_id,
                audio_path=str(path),
                relative_path=relative_path,
                xeno_canto_species_id=str(metadata.get("xeno_canto_species_id", "")),
                download_url=str(metadata.get("download_url", "")),
                fold_index=None,
                train_only_all_folds=False,
            )
        )
    return examples


def _load_excluded_audio_keys(repair_report_path: Path | None) -> set[tuple[str, str]]:
    if repair_report_path is None or not repair_report_path.exists():
        return set()
    excluded: set[tuple[str, str]] = set()
    with repair_report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            status = str(row.get("status", "")).strip()
            if not status or status in ALLOWED_REPAIR_STATUSES:
                continue
            qid = str(row.get("qid", "")).strip()
            recording_id = str(row.get("recording_id", "")).strip()
            if qid and recording_id:
                excluded.add((qid, recording_id))
    return excluded


def _filter_repair_failed_examples(
    examples: list[FineTuneExample],
    repair_report_path: Path | None,
) -> tuple[list[FineTuneExample], list[FineTuneExample]]:
    excluded_keys = _load_excluded_audio_keys(repair_report_path)
    if not excluded_keys:
        return examples, []
    valid_examples: list[FineTuneExample] = []
    excluded_examples: list[FineTuneExample] = []
    for item in examples:
        if (item.qid, item.recording_id) in excluded_keys:
            excluded_examples.append(item)
        else:
            valid_examples.append(item)
    return valid_examples, excluded_examples


def assign_crossval_folds(
    examples: list[FineTuneExample],
    *,
    num_folds: int = DEFAULT_NUM_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[FineTuneExample]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    grouped: dict[str, list[FineTuneExample]] = defaultdict(list)
    for item in examples:
        grouped[item.qid].append(item)

    rng = np.random.default_rng(seed)
    assigned: list[FineTuneExample] = []
    for qid in sorted(grouped):
        group = list(grouped[qid])
        order = rng.permutation(len(group)).tolist()
        shuffled = [group[index] for index in order]
        if len(shuffled) == 1:
            assigned.append(replace(shuffled[0], fold_index=None, train_only_all_folds=True))
            continue
        for offset, item in enumerate(shuffled):
            assigned.append(replace(item, fold_index=offset % num_folds, train_only_all_folds=False))
    return sorted(assigned, key=lambda row: (row.qid, row.relative_path))


def build_fold_splits(examples: list[FineTuneExample], num_folds: int) -> list[FoldSplit]:
    singleton_train_examples = [item for item in examples if item.train_only_all_folds]
    regular_examples = [item for item in examples if not item.train_only_all_folds]
    splits: list[FoldSplit] = []
    for fold_index in range(num_folds):
        test_examples = [item for item in regular_examples if item.fold_index == fold_index]
        train_examples = [item for item in regular_examples if item.fold_index != fold_index] + singleton_train_examples
        splits.append(
            FoldSplit(
                fold_index=fold_index,
                train_examples=sorted(train_examples, key=lambda row: (row.qid, row.relative_path)),
                test_examples=sorted(test_examples, key=lambda row: (row.qid, row.relative_path)),
                singleton_train_examples=singleton_train_examples,
            )
        )
    return splits


def _write_training_curves(output_dir: Path, history: list[dict[str, Any]]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [row.get("train_loss") for row in history]
    eval_loss = [row.get("eval_loss") for row in history]
    eval_accuracy = [row.get("eval_accuracy") for row in history]

    loss_curve_path = output_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, marker="o", label="train_loss")
    if any(value is not None for value in eval_loss):
        ax.plot(epochs, eval_loss, marker="s", label="eval_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("wav2vec2 fine-tuning loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_curve_path, dpi=150)
    plt.close(fig)

    accuracy_curve_path = output_dir / "accuracy_curve.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    accuracy_epochs = [epoch for epoch, value in zip(epochs, eval_accuracy) if value is not None]
    accuracy_values = [value for value in eval_accuracy if value is not None]
    if accuracy_values:
        ax.plot(accuracy_epochs, accuracy_values, marker="o", label="eval_accuracy")
        ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("wav2vec2 fine-tuning eval accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(accuracy_curve_path, dpi=150)
    plt.close(fig)

    return {
        "loss_curve_png": str(loss_curve_path),
        "accuracy_curve_png": str(accuracy_curve_path),
    }


def _count_parameters(model: Any) -> dict[str, int]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def _evaluate_model(model: Any, data_loader: Any, device: str) -> dict[str, float | int | None]:
    import torch

    if len(data_loader) == 0:
        return {
            "loss": None,
            "accuracy": None,
            "example_count": 0,
        }
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            labels = batch["labels"]
            predictions = logits.argmax(dim=-1)
            batch_size = int(labels.shape[0])
            total_examples += batch_size
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_correct += int((predictions == labels).sum().detach().cpu().item())
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "example_count": total_examples,
    }


def _train_one_fold(
    *,
    fold: FoldSplit,
    output_dir: Path,
    base_model_name: str,
    cache_dir: Path | None,
    device: str,
    batch_size: int,
    eval_batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_seconds: float,
    freeze_feature_encoder: bool,
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification

    class FineTuneDataset(Dataset):
        def __init__(self, rows: list[FineTuneExample]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> FineTuneExample:
            return self.rows[index]

    cache_path = str(cache_dir) if cache_dir else None
    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name, cache_dir=cache_path)
    sample_rate = int(getattr(feature_extractor, "sampling_rate", DEFAULT_TARGET_SAMPLE_RATE))
    config = AutoConfig.from_pretrained(
        base_model_name,
        cache_dir=cache_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={str(key): value for key, value in id2label.items()},
        finetuning_task="multi_bird_db_audio_qid",
    )
    model = AutoModelForAudioClassification.from_pretrained(
        base_model_name,
        config=config,
        cache_dir=cache_path,
        ignore_mismatched_sizes=True,
    )
    if freeze_feature_encoder:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        elif hasattr(model, "freeze_feature_extractor"):
            model.freeze_feature_extractor()
    model.to(device)

    collator = AudioClassificationCollator(
        feature_extractor=feature_extractor,
        label2id=label2id,
        sample_rate=sample_rate,
        max_seconds=max_seconds,
    )
    train_dataset = FineTuneDataset(fold.train_examples)
    test_dataset = FineTuneDataset(fold.test_examples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, Any]] = []
    start_time = time.monotonic()

    for epoch_index in range(num_epochs):
        model.train()
        running_loss = 0.0
        seen_examples = 0
        step_count = max(len(train_loader), 1)
        for step_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            batch_size_value = int(batch["labels"].shape[0])
            seen_examples += batch_size_value
            running_loss += float(loss.detach().cpu().item()) * batch_size_value
            _render_progress_line(
                f"wav2vec2 fold {fold.fold_index} | epoch {epoch_index + 1}/{num_epochs} | "
                f"step {step_index}/{step_count} | train_examples {seen_examples}/{len(train_dataset)}"
            )

        train_loss = running_loss / max(seen_examples, 1)
        eval_metrics = _evaluate_model(model, test_loader, device)
        epoch_metrics = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss,
            "eval_loss": eval_metrics["loss"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_example_count": eval_metrics["example_count"],
        }
        history.append(epoch_metrics)
        eval_accuracy = eval_metrics["accuracy"]
        eval_accuracy_text = "n/a" if eval_accuracy is None else f"{float(eval_accuracy):.4f}"
        _finish_progress_line(
            f"wav2vec2 fold {fold.fold_index} | epoch {epoch_index + 1}/{num_epochs} | "
            f"train_loss {train_loss:.4f} | eval_accuracy {eval_accuracy_text}"
        )

    model_dir = output_dir / f"wav2vec2-model_{fold.fold_index}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    feature_extractor.save_pretrained(model_dir)
    _write_manifest(model_dir / "train_manifest.tsv", fold.train_examples)
    _write_manifest(model_dir / "test_manifest.tsv", fold.test_examples)

    curve_paths = _write_training_curves(model_dir, history)
    parameter_counts = _count_parameters(model)
    elapsed_seconds = time.monotonic() - start_time
    summary = {
        "model_dir": str(model_dir),
        "fold_index": fold.fold_index,
        "base_model_name": base_model_name,
        "train_example_count": len(fold.train_examples),
        "test_example_count": len(fold.test_examples),
        "singleton_train_only_count": len(fold.singleton_train_examples),
        "label_count": len(label2id),
        "sample_rate": sample_rate,
        "device": device,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_seconds": max_seconds,
        "freeze_feature_encoder": freeze_feature_encoder,
        "parameter_count": parameter_counts,
        "history": history,
        "output_files": curve_paths,
        "created_at_utc": _timestamp_utc(),
        "elapsed_seconds": elapsed_seconds,
    }
    _write_json_atomic(model_dir / "summary.json", summary)
    return summary


def finetune_wav2vec2_crossval(
    *,
    input_dir: Path,
    recording_map_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    device: str = "auto",
    num_folds: int = DEFAULT_NUM_FOLDS,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    freeze_feature_encoder: bool = True,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    seed: int = DEFAULT_SEED,
    audio_repair_report: Path | None = DEFAULT_AUDIO_REPAIR_REPORT_PATH,
) -> dict[str, Any]:
    raw_examples = build_finetune_examples(input_dir=input_dir, recording_map_path=recording_map_path, extensions=extensions)
    if not raw_examples:
        raise FileNotFoundError(f"No audio files were found under: {input_dir}")

    raw_examples, excluded_examples = _filter_repair_failed_examples(raw_examples, audio_repair_report)
    if not raw_examples:
        raise RuntimeError(f"No usable audio files remain after applying repair report: {audio_repair_report}")

    assigned_examples = assign_crossval_folds(
        raw_examples,
        num_folds=num_folds,
        seed=seed,
    )

    qids = sorted({item.qid for item in assigned_examples})
    label2id = {qid: index for index, qid in enumerate(qids)}
    id2label = {index: qid for qid, index in label2id.items()}
    resolved_device = resolve_torch_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(output_dir / "fold_assignments.tsv", assigned_examples)
    _write_manifest(output_dir / "excluded_audio_files.tsv", excluded_examples)

    folds = build_fold_splits(assigned_examples, num_folds=num_folds)
    fold_summaries = []
    for fold in folds:
        fold_summaries.append(
            _train_one_fold(
                fold=fold,
                output_dir=output_dir,
                base_model_name=model_name,
                cache_dir=cache_dir,
                device=resolved_device,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_seconds=max_seconds,
                freeze_feature_encoder=freeze_feature_encoder,
                label2id=label2id,
                id2label=id2label,
            )
        )

    summary = {
        "kind": "wav2vec2_crossval_finetune",
        "created_at_utc": _timestamp_utc(),
        "input_dir": str(input_dir),
        "recording_map_path": str(recording_map_path),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "cache_dir": "" if cache_dir is None else str(cache_dir),
        "requested_device": device,
        "resolved_device": resolved_device,
        "num_folds": num_folds,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_seconds": max_seconds,
        "freeze_feature_encoder": freeze_feature_encoder,
        "seed": seed,
        "audio_file_count": len(assigned_examples),
        "excluded_audio_file_count": len(excluded_examples),
        "audio_repair_report": "" if audio_repair_report is None else str(audio_repair_report),
        "label_count": len(label2id),
        "singleton_train_only_count": sum(1 for row in assigned_examples if row.train_only_all_folds),
        "fold_summaries": fold_summaries,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Fine-tune wav2vec2 models across 5 cross-validation folds.")
    parser.add_argument("--input-dir", default=str(paths.xeno_canto_raw_dir))
    parser.add_argument("--recording-map", default=str(DEFAULT_RECORDING_MAP_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-folds", type=int, default=DEFAULT_NUM_FOLDS)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--freeze-feature-encoder", action="store_true", default=True)
    parser.add_argument("--no-freeze-feature-encoder", dest="freeze_feature_encoder", action="store_false")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audio-repair-report", default=str(DEFAULT_AUDIO_REPAIR_REPORT_PATH))
    parser.add_argument("--no-audio-repair-report", dest="audio_repair_report", action="store_const", const="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extensions = tuple(ext.strip() for ext in args.extensions.split(",") if ext.strip())
    summary = finetune_wav2vec2_crossval(
        input_dir=Path(args.input_dir),
        recording_map_path=Path(args.recording_map),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        device=args.device,
        num_folds=args.num_folds,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_seconds=args.max_seconds,
        freeze_feature_encoder=args.freeze_feature_encoder,
        extensions=extensions,
        seed=args.seed,
        audio_repair_report=Path(args.audio_repair_report) if args.audio_repair_report else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
