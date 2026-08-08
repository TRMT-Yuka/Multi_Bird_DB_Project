from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from exp1_similarity_matrices import (
    PROJECT_ROOT,
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_SELECTED_RUNS_PATH,
    EmbeddingRun,
    _load_embeddings,
    _load_json,
    _load_qids,
    _read_tsv,
    load_selected_runs,
)
from exp3_sub1_audio_pretraining import load_audio_items


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp3_search"
NDCG_K_VALUES = (10, 50, 100)


@dataclass(frozen=True)
class VectorItem:
    qid: str
    source_id: str
    vector: np.ndarray


@dataclass(frozen=True)
class SearchSample:
    qid: str
    sample_id: str
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
    vectors = _load_embeddings(path)
    if len(qids) != len(vectors):
        raise ValueError(f"qids/embedding row mismatch in {path}: {len(qids)} != {len(vectors)}")
    items: dict[str, list[VectorItem]] = {}
    for qid, vector in zip(qids, vectors, strict=True):
        if not qid:
            continue
        items.setdefault(qid, []).append(VectorItem(qid=qid, source_id=qid, vector=np.asarray(vector, dtype=np.float32)))
    return RunBundle(modality="G", label=label, path=path, items_by_qid=items)


def _language_sort_key(row: dict[str, str], fallback_index: int) -> tuple[str, int, str]:
    try:
        ordinal = int(row.get("ordinal", ""))
    except ValueError:
        ordinal = fallback_index
    return (row.get("qid", ""), ordinal, row.get("surface_id", ""))


def _load_language_bundle(label: str, path: Path) -> RunBundle:
    qids = _load_qids(path)
    vectors = _load_embeddings(path)
    if len(qids) != len(vectors):
        raise ValueError(f"qids/embedding row mismatch in {path}: {len(qids)} != {len(vectors)}")
    rows = _read_tsv(path / "surface_manifest.tsv") if (path / "surface_manifest.tsv").exists() else []
    if len(rows) == len(vectors):
        order = sorted(range(len(rows)), key=lambda index: _language_sort_key(rows[index], index))
        source_ids = [rows[index].get("surface_id", str(index)) for index in range(len(rows))]
    else:
        order = list(range(len(vectors)))
        source_ids = [str(index) for index in range(len(vectors))]

    items: dict[str, list[VectorItem]] = {}
    for index in order:
        qid = qids[index]
        if not qid:
            continue
        source_id = source_ids[index]
        items.setdefault(qid, []).append(
            VectorItem(qid=qid, source_id=source_id, vector=np.asarray(vectors[index], dtype=np.float32))
        )
    return RunBundle(modality="L", label=label, path=path, items_by_qid=items)


def _load_audio_bundle(label: str, path: Path) -> RunBundle:
    # Reuse the EXP3-sub1 loader: window-level rows are averaged per audio file.
    run = type("AudioRunProxy", (), {"name": label, "path": path})()
    audio_items = load_audio_items(run)
    items: dict[str, list[VectorItem]] = {}
    for item in audio_items:
        items.setdefault(item.qid, []).append(
            VectorItem(qid=item.qid, source_id=item.relative_path, vector=np.asarray(item.vector, dtype=np.float32))
        )
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


def build_samples(spec: EvaluationSpec, *, max_samples_per_qid: int | None = None) -> list[SearchSample]:
    common_qids: set[str] | None = None
    for bundle in spec.bundles:
        qids = set(bundle.items_by_qid)
        common_qids = qids if common_qids is None else common_qids & qids
    if not common_qids:
        return []

    samples: list[SearchSample] = []
    for qid in sorted(common_qids):
        item_groups = [bundle.items_by_qid[qid] for bundle in spec.bundles]
        count_for_qid = 0
        for combination in itertools.product(*item_groups):
            sample_id = "|".join(f"{bundle.modality}:{item.source_id}" for bundle, item in zip(spec.bundles, combination, strict=True))
            vector = np.concatenate([item.vector for item in combination]).astype(np.float32)
            samples.append(SearchSample(qid=qid, sample_id=f"{qid}|{sample_id}", vector=vector))
            count_for_qid += 1
            if max_samples_per_qid is not None and count_for_qid >= max_samples_per_qid:
                break
    return samples


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _average_precision(relevance: np.ndarray, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    relevant_seen = 0
    precision_sum = 0.0
    for index, is_relevant in enumerate(relevance, start=1):
        if not is_relevant:
            continue
        relevant_seen += 1
        precision_sum += relevant_seen / index
    return precision_sum / total_relevant


def _reciprocal_rank(relevance: np.ndarray) -> float:
    positions = np.flatnonzero(relevance)
    if len(positions) == 0:
        return 0.0
    return 1.0 / float(positions[0] + 1)


def _ndcg_at_k(relevance: np.ndarray, total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    top = relevance[:k].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2, dtype=np.float64))
    dcg = float(np.sum(top * discounts))
    ideal_len = min(total_relevant, k)
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2, dtype=np.float64))
    idcg = float(np.sum(ideal_discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def evaluate_samples(
    samples: list[SearchSample],
    *,
    allow_self_hit: bool,
    max_query_items: int | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(samples) < 2:
        return _empty_metrics(len(samples)), []

    if max_query_items is not None:
        query_indices = list(range(min(max_query_items, len(samples))))
    else:
        query_indices = list(range(len(samples)))

    vectors = _normalize_rows(np.stack([sample.vector for sample in samples], axis=0).astype(np.float32))
    qids = np.array([sample.qid for sample in samples], dtype=object)
    sample_ids = np.array([sample.sample_id for sample in samples], dtype=object)

    per_query: list[dict[str, object]] = []
    ap_values: list[float] = []
    rr_values: list[float] = []
    ndcg_values = {k: [] for k in NDCG_K_VALUES}

    for query_index in query_indices:
        relevance_mask = qids == qids[query_index]
        candidate_mask = np.ones(len(samples), dtype=bool)
        if not allow_self_hit:
            candidate_mask[query_index] = False
        relevant_count = int(np.sum(relevance_mask & candidate_mask))
        if relevant_count == 0:
            continue

        similarities = vectors @ vectors[query_index]
        candidate_indices = np.flatnonzero(candidate_mask)
        ranked = candidate_indices[np.argsort(-similarities[candidate_indices], kind="mergesort")]
        relevance = relevance_mask[ranked]

        average_precision = _average_precision(relevance, relevant_count)
        reciprocal_rank = _reciprocal_rank(relevance)
        ndcg = {k: _ndcg_at_k(relevance, relevant_count, k) for k in NDCG_K_VALUES}
        ap_values.append(average_precision)
        rr_values.append(reciprocal_rank)
        for k, value in ndcg.items():
            ndcg_values[k].append(value)
        per_query.append(
            {
                "qid": str(qids[query_index]),
                "sample_id": str(sample_ids[query_index]),
                "relevant_count": relevant_count,
                "average_precision": average_precision,
                "reciprocal_rank": reciprocal_rank,
                "nDCG@10": ndcg[10],
                "nDCG@50": ndcg[50],
                "nDCG@100": ndcg[100],
            }
        )

    metrics = {
        "sample_count": len(samples),
        "query_count": len(query_indices),
        "evaluated_query_count": len(per_query),
        "mAP": float(np.mean(ap_values)) if ap_values else 0.0,
        "MRR": float(np.mean(rr_values)) if rr_values else 0.0,
        "nDCG@10": float(np.mean(ndcg_values[10])) if ndcg_values[10] else 0.0,
        "nDCG@50": float(np.mean(ndcg_values[50])) if ndcg_values[50] else 0.0,
        "nDCG@100": float(np.mean(ndcg_values[100])) if ndcg_values[100] else 0.0,
    }
    return metrics, per_query


def _empty_metrics(sample_count: int) -> dict[str, object]:
    return {
        "sample_count": sample_count,
        "query_count": 0,
        "evaluated_query_count": 0,
        "mAP": 0.0,
        "MRR": 0.0,
        "nDCG@10": 0.0,
        "nDCG@50": 0.0,
        "nDCG@100": 0.0,
    }


def run(args: argparse.Namespace) -> None:
    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

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

    summary_rows: list[dict[str, object]] = []
    metadata_specs: list[dict[str, object]] = []
    for spec in specs:
        samples = build_samples(spec, max_samples_per_qid=args.max_samples_per_qid)
        allow_self_hit = args.allow_self_hit or spec.modalities == "G"
        metrics, per_query = evaluate_samples(samples, allow_self_hit=allow_self_hit, max_query_items=args.max_query_items)
        run_name = "__".join([spec.modalities, *(_safe_name(label) for label in spec.run_labels)])
        row = {
            "run_name": run_name,
            "modalities": spec.modalities,
            "run_labels": ",".join(spec.run_labels),
            **metrics,
        }
        summary_rows.append(row)
        metadata_specs.append(
            {
                "run_name": run_name,
                "modalities": spec.modalities,
                "run_labels": list(spec.run_labels),
                "paths": [str(bundle.path) for bundle in spec.bundles],
                "sample_count": metrics["sample_count"],
            }
        )
        if args.write_per_query:
            _write_tsv(
                output_dir / f"{run_name}_per_query.tsv",
                (
                    {
                        **query,
                        "average_precision": f"{float(query['average_precision']):.10f}",
                        "reciprocal_rank": f"{float(query['reciprocal_rank']):.10f}",
                        "nDCG@10": f"{float(query['nDCG@10']):.10f}",
                        "nDCG@50": f"{float(query['nDCG@50']):.10f}",
                        "nDCG@100": f"{float(query['nDCG@100']):.10f}",
                    }
                    for query in per_query
                ),
                ["qid", "sample_id", "relevant_count", "average_precision", "reciprocal_rank", "nDCG@10", "nDCG@50", "nDCG@100"],
            )

    _write_tsv(
        output_dir / "metrics.tsv",
        (
            {
                **row,
                "mAP": f"{float(row['mAP']):.10f}",
                "MRR": f"{float(row['MRR']):.10f}",
                "nDCG@10": f"{float(row['nDCG@10']):.10f}",
                "nDCG@50": f"{float(row['nDCG@50']):.10f}",
                "nDCG@100": f"{float(row['nDCG@100']):.10f}",
            }
            for row in summary_rows
        ),
        [
            "run_name",
            "modalities",
            "run_labels",
            "sample_count",
            "query_count",
            "evaluated_query_count",
            "mAP",
            "MRR",
            "nDCG@10",
            "nDCG@50",
            "nDCG@100",
        ],
    )
    _write_json(
        output_dir / "metadata.json",
        {
            "experiment": "EXP3-main",
            "selected_runs": str(selected_runs_path),
            "modalities": modality_patterns,
            "graph_runs": sorted(graph_filter) if graph_filter else "all selected",
            "language_runs": sorted(language_filter) if language_filter else "all selected",
            "audio_runs": sorted(audio_filter) if audio_filter else "all selected",
            "relevance": "Candidates with the same QID as the query are relevant.",
            "self_hit_policy": "Self-hit is allowed for G-only by default; otherwise exact sample self-hit is excluded unless --allow-self-hit is set.",
            "specs": metadata_specs,
        },
    )

    print(f"specs: {len(specs)}")
    print(f"output_dir: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EXP3 search performance evaluation for selected modality combinations.")
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--modalities",
        action="append",
        default=["G,L,GL"],
        help="Comma-separated modality patterns, e.g. G,L,GL,A,GA,LA,GLA.",
    )
    parser.add_argument("--graph-runs", action="append", default=[], help="Limit graph runs, e.g. node2vec,gcn.")
    parser.add_argument("--language-runs", action="append", default=[], help="Limit language runs, e.g. en.")
    parser.add_argument("--audio-runs", action="append", default=[], help="Limit audio runs, e.g. wav2vec2_base.")
    parser.add_argument("--max-samples-per-qid", type=int, default=None)
    parser.add_argument("--max-query-items", type=int, default=None)
    parser.add_argument("--allow-self-hit", action="store_true")
    parser.add_argument("--write-per-query", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
