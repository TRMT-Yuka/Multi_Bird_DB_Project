from __future__ import annotations

import argparse
import csv
import json
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable


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
from multi_bird_db.multimodal.types import MultimodalSampleRow  # noqa: E402
from multi_bird_db.taxon_labels import load_cached_taxon_labels  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp3_taxon_probe"
DEFAULT_SELECTED_RUNS_PATH = PROJECT_ROOT / "data" / "external" / "embeddings" / "selected_runs.json"
DEFAULT_TAXON_LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "taxonomy" / "qid_taxon_labels.tsv"
DEFAULT_TARGET_RANKS = "family,order"
DEFAULT_TOP_K = 10


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


@dataclass(frozen=True)
class RankingSummary:
    query_count: int
    evaluated_query_count: int
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    f1_at_1: float
    f1_at_5: float
    f1_at_10: float
    mean_average_precision_at_10: float
    mean_reciprocal_rank: float


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


def build_probe_samples(
    spec: EvaluationSpec,
    *,
    labels_by_qid: dict[str, dict[str, dict[str, object] | None]],
    target_rank: str,
) -> list[ProbeSample]:
    common_qids: set[str] | None = None
    for bundle in spec.bundles:
        qids = set(bundle.items_by_qid)
        common_qids = qids if common_qids is None else common_qids & qids
    if not common_qids:
        return []

    assignments = {
        qid: labels_by_qid.get(qid, {}).get(target_rank)
        for qid in sorted(common_qids)
        if labels_by_qid.get(qid, {}).get(target_rank)
    }
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
            target_rank=target_rank,
            target_label=str(assignment["label_name"]),
        )
        samples.append(ProbeSample(row=row, vector=vector))
    return samples


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _prefix_precision_recall(
    relevance: list[int],
    total_relevant: int,
    top_k_values: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict[int, float], dict[int, float], int | None, float, float]:
    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    cumulative_relevant = 0
    first_relevant_rank: int | None = None
    average_precision_at_10 = 0.0

    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            cumulative_relevant += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank
            average_precision_at_10 += cumulative_relevant / float(rank)
        if rank in top_k_values:
            precision_at_k[rank] = cumulative_relevant / float(rank)
            recall_at_k[rank] = 0.0 if total_relevant == 0 else cumulative_relevant / float(total_relevant)

    for k in top_k_values:
        precision_at_k.setdefault(k, 0.0)
        recall_at_k.setdefault(k, 0.0 if total_relevant == 0 else cumulative_relevant / float(total_relevant))

    if total_relevant > 0:
        average_precision_at_10 /= float(total_relevant)
    else:
        average_precision_at_10 = 0.0
    reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / float(first_relevant_rank)
    return precision_at_k, recall_at_k, first_relevant_rank, reciprocal_rank, average_precision_at_10


def _f1_score(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator <= 0.0:
        return 0.0
    return 2.0 * precision * recall / denominator


def evaluate_ranked_candidates(
    samples: list[ProbeSample],
    *,
    top_k: int,
    batch_size: int,
) -> tuple[list[dict[str, object]], RankingSummary]:
    if len(samples) < 2:
        return [], RankingSummary(
            query_count=len(samples),
            evaluated_query_count=0,
            precision_at_1=0.0,
            precision_at_5=0.0,
            precision_at_10=0.0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            f1_at_1=0.0,
            f1_at_5=0.0,
            f1_at_10=0.0,
            mean_average_precision_at_10=0.0,
            mean_reciprocal_rank=0.0,
        )

    effective_k = min(max(1, top_k), len(samples) - 1)
    vectors = _normalize_rows(np.stack([sample.vector for sample in samples], axis=0).astype(np.float32))
    rows = [sample.row for sample in samples]
    per_query_rows: list[dict[str, object]] = []
    precision_at_1_values: list[float] = []
    precision_at_5_values: list[float] = []
    precision_at_10_values: list[float] = []
    recall_at_1_values: list[float] = []
    recall_at_5_values: list[float] = []
    recall_at_10_values: list[float] = []
    f1_at_1_values: list[float] = []
    f1_at_5_values: list[float] = []
    f1_at_10_values: list[float] = []
    average_precision_values: list[float] = []
    reciprocal_rank_values: list[float] = []
    evaluated_query_count = 0

    batch_iter = range(0, len(samples), batch_size)
    for start in tqdm(batch_iter, desc="ranking batches", leave=False):
        stop = min(len(samples), start + batch_size)
        similarity_block = vectors[start:stop] @ vectors.T
        for row_offset, query_index in enumerate(range(start, stop)):
            similarities = np.array(similarity_block[row_offset], copy=True)
            similarities[query_index] = -np.inf
            if effective_k == len(samples) - 1:
                ranked_indices = np.flatnonzero(np.isfinite(similarities))
            else:
                ranked_indices = np.argpartition(-similarities, kth=effective_k - 1)[:effective_k]
            ranked_indices = ranked_indices[np.argsort(-similarities[ranked_indices], kind="mergesort")]
            ranked_labels = [rows[index].target_label for index in ranked_indices]
            relevance = [1 if label == rows[query_index].target_label else 0 for label in ranked_labels[:effective_k]]
            total_relevant = sum(
                1 for index in np.flatnonzero(np.isfinite(similarities)) if rows[index].target_label == rows[query_index].target_label
            )
            if total_relevant > 0:
                evaluated_query_count += 1
            precision_at_k, recall_at_k, first_rank, reciprocal_rank, average_precision_at_10 = _prefix_precision_recall(
                relevance,
                total_relevant,
            )
            precision_at_1_values.append(precision_at_k[1])
            precision_at_5_values.append(precision_at_k[5])
            precision_at_10_values.append(precision_at_k[10])
            recall_at_1_values.append(recall_at_k[1])
            recall_at_5_values.append(recall_at_k[5])
            recall_at_10_values.append(recall_at_k[10])
            f1_at_1_values.append(_f1_score(precision_at_k[1], recall_at_k[1]))
            f1_at_5_values.append(_f1_score(precision_at_k[5], recall_at_k[5]))
            f1_at_10_values.append(_f1_score(precision_at_k[10], recall_at_k[10]))
            average_precision_values.append(average_precision_at_10)
            reciprocal_rank_values.append(reciprocal_rank)
            per_query_rows.append(
                {
                    "sample_id": rows[query_index].sample_id,
                    "qid": rows[query_index].qid,
                    "split": "all",
                    "target_rank": rows[query_index].target_rank,
                    "target_label": rows[query_index].target_label,
                    "modality_pattern": rows[query_index].modality_pattern,
                    "top_k": top_k,
                    "relevant_count": total_relevant,
                    "first_relevant_rank": "" if first_rank is None else first_rank,
                    "reciprocal_rank": f"{reciprocal_rank:.10f}",
                    "precision_at_1": f"{precision_at_k[1]:.10f}",
                    "precision_at_5": f"{precision_at_k[5]:.10f}",
                    "precision_at_10": f"{precision_at_k[10]:.10f}",
                    "recall_at_1": f"{recall_at_k[1]:.10f}",
                    "recall_at_5": f"{recall_at_k[5]:.10f}",
                    "recall_at_10": f"{recall_at_k[10]:.10f}",
                    "f1_at_1": f"{_f1_score(precision_at_k[1], recall_at_k[1]):.10f}",
                    "f1_at_5": f"{_f1_score(precision_at_k[5], recall_at_k[5]):.10f}",
                    "f1_at_10": f"{_f1_score(precision_at_k[10], recall_at_k[10]):.10f}",
                    "hit_at_1": int(recall_at_k[1] > 0.0),
                    "hit_at_5": int(recall_at_k[5] > 0.0),
                    "hit_at_10": int(recall_at_k[10] > 0.0),
                    "ranked_labels": "|".join(ranked_labels),
                    "ranked_relevance": "|".join(str(value) for value in relevance),
                }
            )

    summary = RankingSummary(
        query_count=len(samples),
        evaluated_query_count=evaluated_query_count,
        precision_at_1=float(np.mean(precision_at_1_values)) if precision_at_1_values else 0.0,
        precision_at_5=float(np.mean(precision_at_5_values)) if precision_at_5_values else 0.0,
        precision_at_10=float(np.mean(precision_at_10_values)) if precision_at_10_values else 0.0,
        recall_at_1=float(np.mean(recall_at_1_values)) if recall_at_1_values else 0.0,
        recall_at_5=float(np.mean(recall_at_5_values)) if recall_at_5_values else 0.0,
        recall_at_10=float(np.mean(recall_at_10_values)) if recall_at_10_values else 0.0,
        f1_at_1=float(np.mean(f1_at_1_values)) if f1_at_1_values else 0.0,
        f1_at_5=float(np.mean(f1_at_5_values)) if f1_at_5_values else 0.0,
        f1_at_10=float(np.mean(f1_at_10_values)) if f1_at_10_values else 0.0,
        mean_average_precision_at_10=float(np.mean(average_precision_values)) if average_precision_values else 0.0,
        mean_reciprocal_rank=float(np.mean(reciprocal_rank_values)) if reciprocal_rank_values else 0.0,
    )
    return per_query_rows, summary


def run(args: argparse.Namespace) -> None:
    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    target_ranks = _parse_target_ranks(args.target_ranks)
    modality_patterns = _parse_modalities(args.modalities or ["G,L,GL,A,GA,LA,GLA"])
    graph_filter = _normalize_filter_values(args.graph_runs)
    language_filter = _normalize_filter_values(args.language_runs)
    audio_filter = _normalize_filter_values(args.audio_runs)
    taxon_labels_path = Path(args.taxon_labels).expanduser()
    if not taxon_labels_path.is_absolute():
        taxon_labels_path = PROJECT_ROOT / taxon_labels_path
    if not taxon_labels_path.exists():
        raise SystemExit(
            f"Taxon label cache does not exist: {taxon_labels_path}\n"
            "Run `make build-taxon-labels` before EXP3."
        )
    labels_by_qid = load_cached_taxon_labels(taxon_labels_path, target_ranks)

    bundles = load_bundles(
        selected_runs_path,
        graph_filter=graph_filter,
        language_filter=language_filter,
        audio_filter=audio_filter,
    )
    specs = build_evaluation_specs(modality_patterns, bundles)
    if not specs:
        raise SystemExit("No runnable EXP3 specs. Check --modalities and run filters.")

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    metadata_specs: list[dict[str, object]] = []
    written_dirs: set[str] = set()

    total_runs = len(specs) * len(target_ranks)
    run_index = 0
    for spec in tqdm(specs, desc="EXP3 specs"):
        for target_rank in target_ranks:
            run_index += 1
            tqdm.write(f"[{run_index}/{total_runs}] modalities={spec.modalities} target_rank={target_rank} runs={','.join(spec.run_labels)}")
            samples = build_probe_samples(spec, labels_by_qid=labels_by_qid, target_rank=target_rank)
            if len(samples) < 2:
                continue
            per_query_rows, ranking_summary = evaluate_ranked_candidates(samples, top_k=DEFAULT_TOP_K, batch_size=args.batch_size)
            rows = [sample.row for sample in samples]
            run_name = "__".join([spec.modalities, target_rank, *(_safe_name(label) for label in spec.run_labels)])
            spec_output_dir = output_dir / target_rank / spec.modalities / _safe_name("__".join(spec.run_labels))
            summary_rows.append(
                {
                    "run_name": run_name,
                    "modalities": spec.modalities,
                    "target_rank": target_rank,
                    "run_labels": ",".join(spec.run_labels),
                    "top_k": DEFAULT_TOP_K,
                    "sample_count": ranking_summary.query_count,
                    "evaluated_query_count": ranking_summary.evaluated_query_count,
                    "precision_at_1": f"{ranking_summary.precision_at_1:.10f}",
                    "precision_at_5": f"{ranking_summary.precision_at_5:.10f}",
                    "precision_at_10": f"{ranking_summary.precision_at_10:.10f}",
                    "recall_at_1": f"{ranking_summary.recall_at_1:.10f}",
                    "recall_at_5": f"{ranking_summary.recall_at_5:.10f}",
                    "recall_at_10": f"{ranking_summary.recall_at_10:.10f}",
                    "f1_at_1": f"{ranking_summary.f1_at_1:.10f}",
                    "f1_at_5": f"{ranking_summary.f1_at_5:.10f}",
                    "f1_at_10": f"{ranking_summary.f1_at_10:.10f}",
                    "map_at_10": f"{ranking_summary.mean_average_precision_at_10:.10f}",
                    "mrr": f"{ranking_summary.mean_reciprocal_rank:.10f}",
                }
            )
            metadata_specs.append(
                {
                    "run_name": run_name,
                    "modalities": spec.modalities,
                    "target_rank": target_rank,
                    "run_labels": list(spec.run_labels),
                    "paths": [str(bundle.path) for bundle in spec.bundles],
                    "sample_count": ranking_summary.query_count,
                    "evaluated_query_count": ranking_summary.evaluated_query_count,
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
                        "top_k": DEFAULT_TOP_K,
                        "sample_count": ranking_summary.query_count,
                        "evaluated_query_count": ranking_summary.evaluated_query_count,
                        "precision_at_1": f"{ranking_summary.precision_at_1:.10f}",
                        "precision_at_5": f"{ranking_summary.precision_at_5:.10f}",
                        "precision_at_10": f"{ranking_summary.precision_at_10:.10f}",
                        "recall_at_1": f"{ranking_summary.recall_at_1:.10f}",
                        "recall_at_5": f"{ranking_summary.recall_at_5:.10f}",
                        "recall_at_10": f"{ranking_summary.recall_at_10:.10f}",
                        "f1_at_1": f"{ranking_summary.f1_at_1:.10f}",
                        "f1_at_5": f"{ranking_summary.f1_at_5:.10f}",
                        "f1_at_10": f"{ranking_summary.f1_at_10:.10f}",
                        "map_at_10": f"{ranking_summary.mean_average_precision_at_10:.10f}",
                        "mrr": f"{ranking_summary.mean_reciprocal_rank:.10f}",
                    }
                ],
                [
                    "run_name",
                    "modalities",
                    "target_rank",
                    "run_labels",
                    "top_k",
                    "sample_count",
                    "evaluated_query_count",
                    "precision_at_1",
                    "precision_at_5",
                    "precision_at_10",
                    "recall_at_1",
                    "recall_at_5",
                    "recall_at_10",
                    "f1_at_1",
                    "f1_at_5",
                    "f1_at_10",
                    "map_at_10",
                    "mrr",
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
                    "target_label",
                    "modality_pattern",
                    "top_k",
                    "relevant_count",
                    "first_relevant_rank",
                    "reciprocal_rank",
                    "precision_at_1",
                    "precision_at_5",
                    "precision_at_10",
                    "recall_at_1",
                    "recall_at_5",
                    "recall_at_10",
                    "f1_at_1",
                    "f1_at_5",
                    "f1_at_10",
                    "hit_at_1",
                    "hit_at_5",
                    "hit_at_10",
                    "ranked_labels",
                    "ranked_relevance",
                ],
            )
            _write_json(
                spec_output_dir / "metadata.json",
                {
                    "experiment": "EXP3-taxon-probe",
                    "probe": "ranked-candidate evaluation over frozen embeddings",
                    "top_k": DEFAULT_TOP_K,
                    "batch_size": args.batch_size,
                    "selected_runs": str(selected_runs_path),
                    "taxon_labels": str(taxon_labels_path),
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
            "top_k",
            "sample_count",
            "evaluated_query_count",
            "precision_at_1",
            "precision_at_5",
            "precision_at_10",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "f1_at_1",
            "f1_at_5",
            "f1_at_10",
            "map_at_10",
            "mrr",
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
            "target_label",
            "modality_pattern",
            "top_k",
            "relevant_count",
            "first_relevant_rank",
            "reciprocal_rank",
            "precision_at_1",
            "precision_at_5",
            "precision_at_10",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "f1_at_1",
            "f1_at_5",
            "f1_at_10",
            "hit_at_1",
            "hit_at_5",
            "hit_at_10",
            "ranked_labels",
            "ranked_relevance",
        ],
    )
    _write_json(
        output_dir / "metadata.json",
        {
            "experiment": "EXP3-taxon-probe",
            "probe": "ranked-candidate evaluation over frozen embeddings",
            "top_k": DEFAULT_TOP_K,
            "batch_size": args.batch_size,
            "selected_runs": str(selected_runs_path),
            "taxon_labels": str(taxon_labels_path),
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
    parser = argparse.ArgumentParser(description="Run EXP3 taxon ranking evaluation.")
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument(
        "--taxon-labels",
        default=str(DEFAULT_TAXON_LABELS_PATH),
        help="Path to cached qid_taxon_labels.tsv generated by build-taxon-labels.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--modalities",
        action="append",
        default=[],
        help="Comma-separated modality patterns, defaulting to all combinations: G,L,GL,A,GA,LA,GLA.",
    )
    parser.add_argument("--graph-runs", action="append", default=[], help="Limit graph runs, e.g. node2vec,gcn.")
    parser.add_argument("--language-runs", action="append", default=[], help="Limit language runs, e.g. en.")
    parser.add_argument("--audio-runs", action="append", default=[], help="Limit audio runs, e.g. wav2vec2_base.")
    parser.add_argument("--target-ranks", default=DEFAULT_TARGET_RANKS, help="Comma-separated target ranks, e.g. family,order.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for similarity computation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
