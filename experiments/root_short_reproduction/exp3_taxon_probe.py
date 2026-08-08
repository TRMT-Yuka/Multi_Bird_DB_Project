from __future__ import annotations

import argparse
import csv
import json
import itertools
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exp1_similarity_matrices import (  # noqa: E402
    _load_json,
    _load_qids,
    _read_tsv,
)
from exp3_sub1_audio_pretraining import load_audio_items  # noqa: E402
from multi_bird_db.config import get_project_paths  # noqa: E402
from multi_bird_db.embeddings import load_graph  # noqa: E402
from multi_bird_db.multimodal.evaluate import evaluate_predictions  # noqa: E402
from multi_bird_db.multimodal.labels import assign_labels_for_qids  # noqa: E402
from multi_bird_db.multimodal.types import MultimodalSampleRow  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp3_taxon_probe"
DEFAULT_SELECTED_RUNS_PATH = PROJECT_ROOT / "data" / "external" / "embeddings" / "selected_runs.json"
DEFAULT_TARGET_RANKS = "family,order"


@dataclass(frozen=True)
class VectorItem:
    qid: str
    source_id: str
    vector: np.ndarray


@dataclass(frozen=True)
class RunBundle:
    modality: str
    label: str
    path: Path
    items_by_qid: dict[str, list[VectorItem]]


@dataclass(frozen=True)
class EvaluationSpec:
    modalities: str
    run_labels: tuple[str, ...]
    bundles: tuple[RunBundle, ...]


@dataclass(frozen=True)
class ProbeSample:
    row: MultimodalSampleRow
    vector: np.ndarray


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: Iterable[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _normalize_filter_values(values: list[str]) -> set[str] | None:
    normalized: set[str] = set()
    for value in values:
        for part in value.split(","):
            item = part.strip()
            if item:
                normalized.add(item)
    return normalized or None


def _parse_target_ranks(value: str) -> list[str]:
    ranks: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        rank = part.strip().lower()
        if not rank or rank in seen:
            continue
        ranks.append(rank)
        seen.add(rank)
    if not ranks:
        raise ValueError("At least one target rank is required.")
    return ranks


def _selected_runs_by_modality(path: Path) -> dict[str, dict[str, Path]]:
    payload = _load_json(path)
    runs = payload.get("runs", {}) if isinstance(payload, dict) else {}
    if not isinstance(runs, dict):
        raise ValueError("selected_runs.json must contain a runs object.")

    result: dict[str, dict[str, Path]] = {"graph": {}, "language": {}, "audio": {}}
    for modality in result:
        entries = runs.get(modality, {})
        if not isinstance(entries, dict):
            continue
        for label, value in entries.items():
            run_dir = Path(str(value)).expanduser()
            if not run_dir.is_absolute():
                run_dir = PROJECT_ROOT / run_dir
            result[modality][str(label)] = run_dir
    return result


def _load_graph_bundle(label: str, path: Path) -> RunBundle:
    qids = _load_qids(path)
    vectors = np.asarray(np.load(path / "embeddings.npy"), dtype=np.float32)
    if len(qids) != len(vectors):
        raise ValueError(f"qids/embedding row mismatch in {path}: {len(qids)} != {len(vectors)}")
    items: dict[str, list[VectorItem]] = {}
    for qid, vector in zip(qids, vectors, strict=True):
        if not qid:
            continue
        items.setdefault(qid, []).append(VectorItem(qid=qid, source_id=qid, vector=np.asarray(vector, dtype=np.float32)))
    return RunBundle(modality="G", label=label, path=path, items_by_qid=items)


def _language_row_sort_key(row: dict[str, str], fallback_index: int) -> tuple[str, int, str]:
    try:
        ordinal = int(row.get("ordinal", ""))
    except ValueError:
        ordinal = fallback_index
    return (row.get("qid", ""), ordinal, row.get("surface_id", ""))


def _load_language_bundle(label: str, path: Path) -> RunBundle:
    qids = _load_qids(path)
    vectors = np.asarray(np.load(path / "embeddings.npy"), dtype=np.float32)
    if len(qids) != len(vectors):
        raise ValueError(f"qids/embedding row mismatch in {path}: {len(qids)} != {len(vectors)}")
    rows = _read_tsv(path / "surface_manifest.tsv") if (path / "surface_manifest.tsv").exists() else []
    if len(rows) == len(vectors):
        order = sorted(range(len(rows)), key=lambda index: _language_row_sort_key(rows[index], index))
        source_ids = [rows[index].get("surface_id", str(index)) for index in range(len(rows))]
    else:
        order = list(range(len(vectors)))
        source_ids = [str(index) for index in range(len(vectors))]

    items: dict[str, list[VectorItem]] = {}
    for index in order:
        qid = qids[index]
        if not qid:
            continue
        items.setdefault(qid, []).append(
            VectorItem(qid=qid, source_id=str(source_ids[index]), vector=np.asarray(vectors[index], dtype=np.float32))
        )
    for qid in items:
        items[qid].sort(key=lambda item: item.source_id)
    return RunBundle(modality="L", label=label, path=path, items_by_qid=items)


def _load_audio_bundle(label: str, path: Path) -> RunBundle:
    run = type("AudioRunProxy", (), {"name": label, "path": path})()
    audio_items = load_audio_items(run)
    items: dict[str, list[VectorItem]] = {}
    for item in audio_items:
        items.setdefault(item.qid, []).append(
            VectorItem(qid=item.qid, source_id=item.relative_path, vector=np.asarray(item.vector, dtype=np.float32))
        )
    for qid in items:
        items[qid].sort(key=lambda item: item.source_id)
    return RunBundle(modality="A", label=label, path=path, items_by_qid=items)


def load_bundles(
    selected_runs_path: Path,
    *,
    graph_filter: set[str] | None,
    language_filter: set[str] | None,
    audio_filter: set[str] | None,
) -> dict[str, list[RunBundle]]:
    selected = _selected_runs_by_modality(selected_runs_path)
    bundles: dict[str, list[RunBundle]] = {"G": [], "L": [], "A": []}

    for label, path in sorted(selected["graph"].items()):
        if graph_filter is None or label in graph_filter:
            bundles["G"].append(_load_graph_bundle(label, path))
    for label, path in sorted(selected["language"].items()):
        if language_filter is None or label in language_filter:
            bundles["L"].append(_load_language_bundle(label, path))
    for label, path in sorted(selected["audio"].items()):
        if audio_filter is None or label in audio_filter:
            bundles["A"].append(_load_audio_bundle(label, path))
    return bundles


def _parse_modalities(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        for part in value.split(","):
            item = part.strip().upper()
            if not item:
                continue
            if any(char not in "GLA" for char in item):
                raise ValueError(f"Unsupported modality pattern: {item}")
            normalized = "".join(char for char in "GLA" if char in item)
            if normalized and normalized not in parsed:
                parsed.append(normalized)
    return parsed


def build_evaluation_specs(modality_patterns: list[str], bundles: dict[str, list[RunBundle]]) -> list[EvaluationSpec]:
    specs: list[EvaluationSpec] = []
    for pattern in modality_patterns:
        bundle_groups = [bundles[modality] for modality in pattern]
        if any(not group for group in bundle_groups):
            continue
        for bundle_tuple in itertools.product(*bundle_groups):
            run_labels = tuple(f"{bundle.modality}:{bundle.label}" for bundle in bundle_tuple)
            specs.append(EvaluationSpec(modalities=pattern, run_labels=run_labels, bundles=tuple(bundle_tuple)))
    return specs


def _representative_item(items: list[VectorItem]) -> VectorItem:
    if not items:
        raise ValueError("Cannot select a representative from an empty item list.")
    return items[0]


def build_probe_samples(spec: EvaluationSpec, taxonomy_graph, target_rank: str) -> list[ProbeSample]:
    common_qids: set[str] | None = None
    for bundle in spec.bundles:
        qids = set(bundle.items_by_qid)
        common_qids = qids if common_qids is None else common_qids & qids
    if not common_qids:
        return []

    assignments = {assignment.qid: assignment for assignment in assign_labels_for_qids(taxonomy_graph, sorted(common_qids), target_rank)}
    if not assignments:
        return []

    samples: list[ProbeSample] = []
    for qid in sorted(assignments):
        chosen_items = [_representative_item(bundle.items_by_qid[qid]) for bundle in spec.bundles]
        vector = np.concatenate([item.vector for item in chosen_items]).astype(np.float32)
        sample_id = "|".join(f"{bundle.modality}:{item.source_id}" for bundle, item in zip(spec.bundles, chosen_items, strict=True))
        assignment = assignments[qid]
        row = MultimodalSampleRow(
            sample_id=f"{qid}|{sample_id}",
            qid=qid,
            graph_embedding_index=None,
            audio_embedding_index=None,
            language_embedding_index=None,
            modality_pattern=spec.modalities,
            target_rank=assignment.target_rank,
            target_label=assignment.label_name,
        )
        samples.append(ProbeSample(row=row, vector=vector))
    return samples


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _majority_vote(labels: list[str], similarities: list[float]) -> tuple[str, dict[str, int]]:
    counts = Counter(labels)
    best_count = max(counts.values())
    tied_labels = [label for label, count in counts.items() if count == best_count]
    if len(tied_labels) == 1:
        return tied_labels[0], dict(counts)

    best_label = tied_labels[0]
    best_similarity = float("-inf")
    for label in tied_labels:
        first_similarity = next(similarity for voted_label, similarity in zip(labels, similarities, strict=True) if voted_label == label)
        if first_similarity > best_similarity or (first_similarity == best_similarity and label < best_label):
            best_label = label
            best_similarity = first_similarity
    return best_label, dict(counts)


def _ranking_metrics(
    neighbor_labels: list[str],
    true_label: str,
    *,
    top_k_values: tuple[int, ...] = (1, 5, 10),
) -> tuple[int | None, float, dict[int, bool]]:
    first_rank: int | None = None
    for index, label in enumerate(neighbor_labels, start=1):
        if label == true_label:
            first_rank = index
            break

    reciprocal_rank = 0.0 if first_rank is None else 1.0 / float(first_rank)
    top_hits = {k: (first_rank is not None and first_rank <= k) for k in top_k_values}
    return first_rank, reciprocal_rank, top_hits


def predict_knn_majority(
    samples: list[ProbeSample],
    *,
    k: int,
    batch_size: int,
) -> tuple[list[str], list[dict[str, object]]]:
    if len(samples) < 2:
        return [], []

    effective_k = min(max(1, k), len(samples) - 1)
    vectors = _normalize_rows(np.stack([sample.vector for sample in samples], axis=0).astype(np.float32))
    rows = [sample.row for sample in samples]
    predicted_labels: list[str] = []
    per_query_rows: list[dict[str, object]] = []

    for start in range(0, len(samples), batch_size):
        stop = min(len(samples), start + batch_size)
        similarity_block = vectors[start:stop] @ vectors.T
        for row_offset, query_index in enumerate(range(start, stop)):
            similarities = np.array(similarity_block[row_offset], copy=True)
            similarities[query_index] = -np.inf
            neighbor_count = min(effective_k, len(samples) - 1)
            if neighbor_count <= 0:
                predicted_labels.append("")
                per_query_rows.append(
                    {
                        "sample_id": rows[query_index].sample_id,
                        "qid": rows[query_index].qid,
                        "split": "all",
                        "target_rank": rows[query_index].target_rank,
                        "true_label": rows[query_index].target_label,
                        "predicted_label": "",
                        "modality_pattern": rows[query_index].modality_pattern,
                        "k": k,
                    }
                )
                continue

            if neighbor_count == len(samples) - 1:
                neighbor_indices = np.flatnonzero(np.isfinite(similarities))
            else:
                neighbor_indices = np.argpartition(-similarities, kth=neighbor_count - 1)[:neighbor_count]
            neighbor_indices = neighbor_indices[np.argsort(-similarities[neighbor_indices], kind="mergesort")]
            neighbor_labels = [rows[index].target_label for index in neighbor_indices]
            neighbor_similarities = [float(similarities[index]) for index in neighbor_indices]
            predicted_label, vote_counts = _majority_vote(neighbor_labels, neighbor_similarities)
            first_rank, reciprocal_rank, top_hits = _ranking_metrics(neighbor_labels, rows[query_index].target_label)
            predicted_labels.append(predicted_label)
            per_query_rows.append(
                {
                    "sample_id": rows[query_index].sample_id,
                    "qid": rows[query_index].qid,
                    "split": "all",
                    "target_rank": rows[query_index].target_rank,
                    "true_label": rows[query_index].target_label,
                    "predicted_label": predicted_label,
                    "modality_pattern": rows[query_index].modality_pattern,
                    "k": k,
                    "first_correct_rank": "" if first_rank is None else first_rank,
                    "reciprocal_rank": f"{reciprocal_rank:.10f}",
                    "top1_hit": int(top_hits[1]),
                    "top5_hit": int(top_hits[5]),
                    "top10_hit": int(top_hits[10]),
                    "neighbor_labels": "|".join(neighbor_labels),
                    "vote_counts": json.dumps(vote_counts, ensure_ascii=False, sort_keys=True),
                }
            )
    return predicted_labels, per_query_rows


def run(args: argparse.Namespace) -> None:
    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    target_ranks = _parse_target_ranks(args.target_ranks)
    modality_patterns = _parse_modalities(args.modalities)
    graph_filter = _normalize_filter_values(args.graph_runs)
    language_filter = _normalize_filter_values(args.language_runs)
    audio_filter = _normalize_filter_values(args.audio_runs)

    bundles = load_bundles(
        selected_runs_path,
        graph_filter=graph_filter,
        language_filter=language_filter,
        audio_filter=audio_filter,
    )
    specs = build_evaluation_specs(modality_patterns, bundles)
    if not specs:
        raise SystemExit("No runnable EXP3 specs. Check --modalities and run filters.")

    taxonomy_graph_path = Path(args.taxonomy_graph).expanduser() if args.taxonomy_graph else get_project_paths().taxonomy_graph_pkl
    if not taxonomy_graph_path.is_absolute():
        taxonomy_graph_path = PROJECT_ROOT / taxonomy_graph_path
    taxonomy_graph = load_graph(taxonomy_graph_path)

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    metadata_specs: list[dict[str, object]] = []
    written_dirs: set[str] = set()

    for spec in specs:
        for target_rank in target_ranks:
            samples = build_probe_samples(spec, taxonomy_graph, target_rank)
            if len(samples) < 2:
                continue
            predicted_labels, per_query_rows = predict_knn_majority(samples, k=args.k, batch_size=args.batch_size)
            rows = [sample.row for sample in samples]
            classes = sorted({row.target_label for row in rows})
            evaluation = evaluate_predictions(
                split_name="all",
                rows=rows,
                predicted_labels=predicted_labels,
                classes=classes,
            )
            reciprocal_ranks = [float(row["reciprocal_rank"]) for row in per_query_rows]
            top1_hits = [int(row["top1_hit"]) for row in per_query_rows]
            top5_hits = [int(row["top5_hit"]) for row in per_query_rows]
            top10_hits = [int(row["top10_hit"]) for row in per_query_rows]
            run_name = "__".join([spec.modalities, target_rank, *(_safe_name(label) for label in spec.run_labels)])
            spec_output_dir = output_dir / target_rank / spec.modalities / _safe_name("__".join(spec.run_labels))
            summary_rows.append(
                {
                    "run_name": run_name,
                    "modalities": spec.modalities,
                    "target_rank": target_rank,
                    "run_labels": ",".join(spec.run_labels),
                    "k": args.k,
                    "sample_count": len(samples),
                    "class_count": len(classes),
                    "accuracy": f"{evaluation.accuracy:.10f}",
                    "macro_f1": f"{evaluation.macro_f1:.10f}",
                    "mrr": f"{float(np.mean(reciprocal_ranks)):.10f}" if reciprocal_ranks else "0.0000000000",
                    "top1_accuracy": f"{float(np.mean(top1_hits)):.10f}" if top1_hits else "0.0000000000",
                    "top5_accuracy": f"{float(np.mean(top5_hits)):.10f}" if top5_hits else "0.0000000000",
                    "top10_accuracy": f"{float(np.mean(top10_hits)):.10f}" if top10_hits else "0.0000000000",
                }
            )
            metadata_specs.append(
                {
                    "run_name": run_name,
                    "modalities": spec.modalities,
                    "target_rank": target_rank,
                    "run_labels": list(spec.run_labels),
                    "paths": [str(bundle.path) for bundle in spec.bundles],
                    "sample_count": len(samples),
                    "class_count": len(classes),
                }
            )
            _write_tsv(
                spec_output_dir / "metrics.tsv",
                [
                    {
                        "run_name": run_name,
                        "modalities": spec.modalities,
                        "target_rank": target_rank,
                        "run_labels": ",".join(spec.run_labels),
                        "k": args.k,
                        "sample_count": len(samples),
                        "class_count": len(classes),
                        "accuracy": f"{evaluation.accuracy:.10f}",
                        "macro_f1": f"{evaluation.macro_f1:.10f}",
                        "mrr": f"{float(np.mean(reciprocal_ranks)):.10f}" if reciprocal_ranks else "0.0000000000",
                        "top1_accuracy": f"{float(np.mean(top1_hits)):.10f}" if top1_hits else "0.0000000000",
                        "top5_accuracy": f"{float(np.mean(top5_hits)):.10f}" if top5_hits else "0.0000000000",
                        "top10_accuracy": f"{float(np.mean(top10_hits)):.10f}" if top10_hits else "0.0000000000",
                    }
                ],
                [
                    "run_name",
                    "modalities",
                    "target_rank",
                    "run_labels",
                    "k",
                    "sample_count",
                    "class_count",
                    "accuracy",
                    "macro_f1",
                    "mrr",
                    "top1_accuracy",
                    "top5_accuracy",
                    "top10_accuracy",
                ],
            )
            _write_tsv(
                spec_output_dir / "predictions.tsv",
                (
                    {
                        "run_name": run_name,
                        **row,
                    }
                    for row in per_query_rows
                ),
                [
                    "run_name",
                    "sample_id",
                    "qid",
                    "split",
                    "target_rank",
                    "true_label",
                    "predicted_label",
                    "modality_pattern",
                    "k",
                    "first_correct_rank",
                    "reciprocal_rank",
                    "top1_hit",
                    "top5_hit",
                    "top10_hit",
                    "neighbor_labels",
                    "vote_counts",
                ],
            )
            _write_json(
                spec_output_dir / "metadata.json",
                {
                    "experiment": "EXP3-taxon-probe",
                    "probe": "leave-one-out k-nearest-neighbor majority vote over frozen embeddings",
                    "k": args.k,
                    "batch_size": args.batch_size,
                    "selected_runs": str(selected_runs_path),
                    "taxonomy_graph": str(taxonomy_graph_path),
                    "modalities": [spec.modalities],
                    "graph_runs": sorted(graph_filter) if graph_filter else "all selected",
                    "language_runs": sorted(language_filter) if language_filter else "all selected",
                    "audio_runs": sorted(audio_filter) if audio_filter else "all selected",
                    "target_ranks": [target_rank],
                    "spec": {
                        "run_name": run_name,
                        "modalities": spec.modalities,
                        "target_rank": target_rank,
                        "run_labels": list(spec.run_labels),
                        "paths": [str(bundle.path) for bundle in spec.bundles],
                        "sample_count": len(samples),
                        "class_count": len(classes),
                    },
                },
            )
            written_dirs.add(str(spec_output_dir))
            for row in per_query_rows:
                prediction_rows.append(
                    {
                        "run_name": run_name,
                        **row,
                    }
                )

    if not summary_rows:
        raise SystemExit("No EXP3 taxon probe results were produced.")

    _write_tsv(
        output_dir / "metrics.tsv",
        summary_rows,
        [
            "run_name",
            "modalities",
            "target_rank",
            "run_labels",
            "k",
            "sample_count",
            "class_count",
            "accuracy",
            "macro_f1",
            "mrr",
            "top1_accuracy",
            "top5_accuracy",
            "top10_accuracy",
        ],
    )
    _write_tsv(
        output_dir / "predictions.tsv",
        prediction_rows,
        [
            "run_name",
            "sample_id",
            "qid",
            "split",
            "target_rank",
            "true_label",
            "predicted_label",
            "modality_pattern",
            "k",
            "first_correct_rank",
            "reciprocal_rank",
            "top1_hit",
            "top5_hit",
            "top10_hit",
            "neighbor_labels",
            "vote_counts",
        ],
    )
    _write_json(
        output_dir / "metadata.json",
        {
            "experiment": "EXP3-taxon-probe",
            "probe": "leave-one-out k-nearest-neighbor majority vote over frozen embeddings",
            "k": args.k,
            "batch_size": args.batch_size,
            "selected_runs": str(selected_runs_path),
            "taxonomy_graph": str(taxonomy_graph_path),
            "modalities": modality_patterns,
            "graph_runs": sorted(graph_filter) if graph_filter else "all selected",
            "language_runs": sorted(language_filter) if language_filter else "all selected",
            "audio_runs": sorted(audio_filter) if audio_filter else "all selected",
            "target_ranks": target_ranks,
            "specs": metadata_specs,
            "written_dirs": sorted(written_dirs),
        },
    )

    print(f"specs: {len(summary_rows)}")
    print(f"output_dir: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EXP3 taxon classification with k-nearest majority vote.")
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument("--taxonomy-graph", default=None, help="Path to bird taxonomy graph PKL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--modalities",
        action="append",
        default=["G,L,GL,A,GA,LA,GLA"],
        help="Comma-separated modality patterns, defaulting to all combinations: G,L,GL,A,GA,LA,GLA.",
    )
    parser.add_argument("--graph-runs", action="append", default=[], help="Limit graph runs, e.g. node2vec,gcn.")
    parser.add_argument("--language-runs", action="append", default=[], help="Limit language runs, e.g. en.")
    parser.add_argument("--audio-runs", action="append", default=[], help="Limit audio runs, e.g. wav2vec2_base.")
    parser.add_argument("--target-ranks", default=DEFAULT_TARGET_RANKS, help="Comma-separated target ranks, e.g. family,order.")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors used for majority vote.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for similarity computation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
