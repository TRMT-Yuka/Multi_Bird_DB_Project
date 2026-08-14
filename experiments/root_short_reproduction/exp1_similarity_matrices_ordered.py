from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMP_CACHE_ROOT = PROJECT_ROOT / "temp" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_ROOT / "xdg_cache"))

from exp1_similarity_matrices import (
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_SELECTED_RUNS_PATH,
    EmbeddingRun,
    _load_json,
    _parse_run_paths,
    _safe_name,
    _write_json,
    _write_tsv,
    build_qid_vector_maps,
    cosine_similarity_matrix,
    discover_all_runs,
    load_selected_runs,
    plot_similarity_distribution,
    plot_similarity_heatmap,
)

DEFAULT_QID_ORDER_PATH = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp1_xeno_canto_num_recordings" / "top_100_qids.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp1_img_ordered"


def _load_qid_order(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"QID order file does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        if isinstance(payload, list):
            return [str(value).strip() for value in payload if str(value).strip()]
        raise ValueError(f"Unsupported JSON qid order format: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        return []
    if "qid" not in rows[0]:
        raise ValueError(f"QID order TSV must contain a qid column: {path}")
    return [str(row.get("qid", "")).strip() for row in rows if str(row.get("qid", "")).strip()]


def _save_run_outputs(
    *,
    run: EmbeddingRun,
    qids: list[str],
    vectors: np.ndarray,
    output_dir: Path,
    source_order_path: Path,
    source_order_count: int,
) -> None:
    run_dir = output_dir / _safe_name(run.name)
    run_dir.mkdir(parents=True, exist_ok=True)

    matrix = cosine_similarity_matrix(vectors)
    np.save(run_dir / "similarity.npy", matrix)
    _write_json(run_dir / "qids.json", qids)
    _write_tsv(
        run_dir / "qids.tsv",
        ({"rank": index + 1, "qid": qid} for index, qid in enumerate(qids)),
        ["rank", "qid"],
    )
    plot_similarity_heatmap(matrix, tuple(qids), run_dir / "heatmap.png", title=run.name)
    plot_similarity_distribution(matrix, run_dir / "distribution.png", title=run.name)
    _write_json(
        run_dir / "metadata.json",
        {
            "run_name": run.name,
            "modality": run.modality,
            "path": str(run.path),
            "qid_count": len(qids),
            "source_order_path": str(source_order_path),
            "source_order_count": source_order_count,
            "similarity": "cosine similarity",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build similarity matrices in a fixed QID order for Experiment 1.")
    parser.add_argument("--embedding-root", default=str(DEFAULT_EMBEDDING_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument("--qid-order", default=str(DEFAULT_QID_ORDER_PATH))
    parser.add_argument("--use-selected-runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-discover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-run", action="append", default=[], help="Graph embedding run directory.")
    parser.add_argument("--language-run", action="append", default=[], help="Language embedding run directory.")
    parser.add_argument("--audio-run", action="append", default=[], help="Audio embedding run directory.")
    parser.add_argument("--max-qids", type=int, default=None, help="Optional debug cap after applying the fixed order.")
    return parser


def run(args: argparse.Namespace) -> None:
    embedding_root = Path(args.embedding_root).expanduser()
    if not embedding_root.is_absolute():
        embedding_root = PROJECT_ROOT / embedding_root

    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path

    qid_order_path = Path(args.qid_order).expanduser()
    if not qid_order_path.is_absolute():
        qid_order_path = PROJECT_ROOT / qid_order_path

    if args.use_selected_runs and selected_runs_path.exists():
        runs = load_selected_runs(selected_runs_path)
    elif args.auto_discover:
        runs = discover_all_runs(embedding_root)
    else:
        runs = []
    runs.extend(_parse_run_paths(args.graph_run, "graph"))
    runs.extend(_parse_run_paths(args.language_run, "language"))
    runs.extend(_parse_run_paths(args.audio_run, "audio"))

    if not runs:
        raise SystemExit("No embedding runs found. Use --auto-discover or pass --graph-run/--language-run/--audio-run.")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    qid_order = _load_qid_order(qid_order_path)
    if args.max_qids is not None:
        qid_order = qid_order[: args.max_qids]
    if not qid_order:
        raise SystemExit("QID order is empty.")

    vector_maps, audio_choices = build_qid_vector_maps(runs)

    summary_rows: list[dict[str, object]] = []
    for run in runs:
        qid_map = vector_maps[run.name]
        ordered_qids = [qid for qid in qid_order if qid in qid_map]
        if not ordered_qids:
            continue
        vectors = np.stack([qid_map[qid] for qid in ordered_qids], axis=0).astype(np.float32)
        _save_run_outputs(
            run=run,
            qids=ordered_qids,
            vectors=vectors,
            output_dir=output_dir,
            source_order_path=qid_order_path,
            source_order_count=len(qid_order),
        )
        summary_rows.append(
            {
                "run_name": run.name,
                "modality": run.modality,
                "path": str(run.path),
                "ordered_qid_count": len(ordered_qids),
                "source_order_count": len(qid_order),
                "missing_in_run_count": len(qid_order) - len(ordered_qids),
            }
        )

    _write_tsv(
        output_dir / "summary.tsv",
        summary_rows,
        ["run_name", "modality", "path", "ordered_qid_count", "source_order_count", "missing_in_run_count"],
    )
    _write_json(
        output_dir / "metadata.json",
        {
            "experiment": "EXP1-similarity-matrices-ordered",
            "output_dir": str(output_dir),
            "qid_order_path": str(qid_order_path),
            "qid_order_count": len(qid_order),
            "run_count": len(summary_rows),
            "runs": summary_rows,
            "audio_representative_policy": (
                "For each QID, use audio files that exist in every audio embedding run, "
                "sort their relative paths lexicographically, select the first file, "
                "and average window-level vectors within that file when necessary."
            ),
        },
    )

    print(json.dumps({"qid_order_count": len(qid_order), "run_count": len(summary_rows)}, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"output_dir: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
