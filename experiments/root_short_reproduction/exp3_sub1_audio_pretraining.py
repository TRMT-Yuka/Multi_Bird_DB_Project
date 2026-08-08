from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMBEDDING_ROOT = PROJECT_ROOT / "data" / "external" / "embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp3_sub1_audio"
DEFAULT_SELECTED_RUNS_PATH = DEFAULT_EMBEDDING_ROOT / "selected_runs.json"
NDCG_K_VALUES = (10, 50, 100)


@dataclass(frozen=True)
class AudioRun:
    name: str
    path: Path


@dataclass(frozen=True)
class AudioItem:
    qid: str
    relative_path: str
    vector: np.ndarray
    row_count: int


@dataclass(frozen=True)
class QueryMetric:
    qid: str
    relative_path: str
    relevant_count: int
    average_precision: float
    reciprocal_rank: float
    ndcg_at_10: float
    ndcg_at_50: float
    ndcg_at_100: float


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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




def load_selected_audio_runs(path: Path) -> list[AudioRun]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selected_runs.json must contain an object.")
    runs = payload.get("runs", {})
    if not isinstance(runs, dict):
        raise ValueError("selected_runs.json must contain a runs object.")
    audio_entries = runs.get("audio", {})
    if not isinstance(audio_entries, dict):
        raise ValueError("selected_runs.json runs.audio must be an object.")

    selected: list[AudioRun] = []
    for label, value in audio_entries.items():
        run_dir = Path(str(value)).expanduser()
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        selected.append(AudioRun(name=str(label), path=run_dir))
    return selected

def discover_audio_runs(root: Path = DEFAULT_EMBEDDING_ROOT) -> list[AudioRun]:
    audio_root = root / "audio"
    runs: list[AudioRun] = []
    if not audio_root.exists():
        return runs
    for run_dir in sorted(audio_root.glob("*/*/*")):
        if (run_dir / "embeddings.npy").exists() and (run_dir / "audio_manifest.tsv").exists():
            backend = run_dir.parents[1].name
            model = run_dir.parent.name
            tag = run_dir.name
            runs.append(AudioRun(name=f"{backend}__{model}__{tag}", path=run_dir))
    return runs


def parse_audio_run_paths(values: list[str]) -> list[AudioRun]:
    runs: list[AudioRun] = []
    for value in values:
        run_dir = Path(value).expanduser()
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        runs.append(AudioRun(name=f"{run_dir.parents[1].name}__{run_dir.parent.name}__{run_dir.name}", path=run_dir))
    return runs


def load_audio_items(run: AudioRun) -> list[AudioItem]:
    vectors = np.load(run.path / "embeddings.npy")
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embeddings.npy: {run.path / 'embeddings.npy'}")
    rows = _read_tsv(run.path / "audio_manifest.tsv")
    if len(rows) != len(vectors):
        raise ValueError(f"audio manifest/embedding row mismatch in {run.path}: {len(rows)} != {len(vectors)}")

    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for row, vector in zip(rows, vectors, strict=True):
        qid = row.get("qid", "").strip()
        relative_path = row.get("relative_path", "").strip()
        if not qid or not relative_path:
            continue
        grouped.setdefault((qid, relative_path), []).append(np.asarray(vector, dtype=np.float32))

    items = [
        AudioItem(
            qid=qid,
            relative_path=relative_path,
            vector=np.mean(np.stack(group_vectors, axis=0), axis=0).astype(np.float32),
            row_count=len(group_vectors),
        )
        for (qid, relative_path), group_vectors in grouped.items()
    ]
    return sorted(items, key=lambda item: (item.qid, item.relative_path))


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
    relevant_positions = np.flatnonzero(relevance)
    if len(relevant_positions) == 0:
        return 0.0
    return 1.0 / float(relevant_positions[0] + 1)


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


def evaluate_audio_run(items: list[AudioItem]) -> list[QueryMetric]:
    if len(items) < 2:
        return []

    vectors = _normalize_rows(np.stack([item.vector for item in items], axis=0).astype(np.float32))
    qids = np.array([item.qid for item in items], dtype=object)
    paths = np.array([item.relative_path for item in items], dtype=object)
    similarities = vectors @ vectors.T

    metrics: list[QueryMetric] = []
    for query_index, item in enumerate(items):
        candidate_mask = np.ones(len(items), dtype=bool)
        candidate_mask[query_index] = False
        relevance_mask = (qids == item.qid) & (paths != item.relative_path)
        relevant_count = int(np.sum(relevance_mask & candidate_mask))
        if relevant_count == 0:
            continue

        candidate_indices = np.flatnonzero(candidate_mask)
        ranked_candidate_indices = candidate_indices[np.argsort(-similarities[query_index, candidate_indices], kind="mergesort")]
        relevance = relevance_mask[ranked_candidate_indices]
        metrics.append(
            QueryMetric(
                qid=item.qid,
                relative_path=item.relative_path,
                relevant_count=relevant_count,
                average_precision=_average_precision(relevance, relevant_count),
                reciprocal_rank=_reciprocal_rank(relevance),
                ndcg_at_10=_ndcg_at_k(relevance, relevant_count, 10),
                ndcg_at_50=_ndcg_at_k(relevance, relevant_count, 50),
                ndcg_at_100=_ndcg_at_k(relevance, relevant_count, 100),
            )
        )
    return metrics


def summarize_run(run: AudioRun, items: list[AudioItem], query_metrics: list[QueryMetric]) -> dict[str, object]:
    qids = {item.qid for item in items}
    rows = len(items)
    if not query_metrics:
        return {
            "run_name": run.name,
            "path": str(run.path),
            "audio_item_count": rows,
            "qid_count": len(qids),
            "evaluated_query_count": 0,
            "mAP": 0.0,
            "MRR": 0.0,
            "nDCG@10": 0.0,
            "nDCG@50": 0.0,
            "nDCG@100": 0.0,
        }
    return {
        "run_name": run.name,
        "path": str(run.path),
        "audio_item_count": rows,
        "qid_count": len(qids),
        "evaluated_query_count": len(query_metrics),
        "mAP": float(np.mean([metric.average_precision for metric in query_metrics])),
        "MRR": float(np.mean([metric.reciprocal_rank for metric in query_metrics])),
        "nDCG@10": float(np.mean([metric.ndcg_at_10 for metric in query_metrics])),
        "nDCG@50": float(np.mean([metric.ndcg_at_50 for metric in query_metrics])),
        "nDCG@100": float(np.mean([metric.ndcg_at_100 for metric in query_metrics])),
    }


def write_query_metrics(path: Path, run_name: str, metrics: list[QueryMetric]) -> None:
    _write_tsv(
        path,
        (
            {
                "run_name": run_name,
                "qid": metric.qid,
                "relative_path": metric.relative_path,
                "relevant_count": metric.relevant_count,
                "average_precision": f"{metric.average_precision:.10f}",
                "reciprocal_rank": f"{metric.reciprocal_rank:.10f}",
                "nDCG@10": f"{metric.ndcg_at_10:.10f}",
                "nDCG@50": f"{metric.ndcg_at_50:.10f}",
                "nDCG@100": f"{metric.ndcg_at_100:.10f}",
            }
            for metric in metrics
        ),
        [
            "run_name",
            "qid",
            "relative_path",
            "relevant_count",
            "average_precision",
            "reciprocal_rank",
            "nDCG@10",
            "nDCG@50",
            "nDCG@100",
        ],
    )


def run(args: argparse.Namespace) -> None:
    embedding_root = Path(args.embedding_root).expanduser()
    if not embedding_root.is_absolute():
        embedding_root = PROJECT_ROOT / embedding_root
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path

    if args.use_selected_runs and selected_runs_path.exists():
        audio_runs = load_selected_audio_runs(selected_runs_path)
    else:
        audio_runs = discover_audio_runs(embedding_root) if args.auto_discover else []
    audio_runs.extend(parse_audio_run_paths(args.audio_run))
    if not audio_runs:
        raise SystemExit("No audio embedding runs found. Use --auto-discover or pass --audio-run.")

    summaries: list[dict[str, object]] = []
    for audio_run in audio_runs:
        items = load_audio_items(audio_run)
        if args.max_items is not None:
            items = items[: args.max_items]
        query_metrics = evaluate_audio_run(items)
        summary = summarize_run(audio_run, items, query_metrics)
        summaries.append(summary)
        write_query_metrics(output_dir / f"{_safe_name(audio_run.name)}_per_query.tsv", audio_run.name, query_metrics)

    _write_tsv(
        output_dir / "metrics.tsv",
        (
            {
                **summary,
                "mAP": f"{float(summary['mAP']):.10f}",
                "MRR": f"{float(summary['MRR']):.10f}",
                "nDCG@10": f"{float(summary['nDCG@10']):.10f}",
                "nDCG@50": f"{float(summary['nDCG@50']):.10f}",
                "nDCG@100": f"{float(summary['nDCG@100']):.10f}",
            }
            for summary in summaries
        ),
        [
            "run_name",
            "path",
            "audio_item_count",
            "qid_count",
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
            "experiment": "EXP3-sub1",
            "description": "Audio-only retrieval comparison for wav2vec pretraining/fine-tuning style Table 1 experiments.",
            "relevance": "A candidate is relevant when it has the same QID as the query and a different audio relative_path.",
            "same_source_exclusion": "The query audio file itself is excluded from candidates.",
            "window_policy": "Rows with the same QID and relative_path are averaged before retrieval evaluation.",
            "metrics": ["mAP", "MRR", "nDCG@10", "nDCG@50", "nDCG@100"],
            "runs": summaries,
        },
    )

    print(f"audio_runs: {len(audio_runs)}")
    print(f"output_dir: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EXP3-sub1 audio embedding retrieval evaluation.")
    parser.add_argument("--embedding-root", default=str(DEFAULT_EMBEDDING_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument("--use-selected-runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-discover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-run", action="append", default=[], help="Audio embedding run directory.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional debug cap after loading audio items.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
