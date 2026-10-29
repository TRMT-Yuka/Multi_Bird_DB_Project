from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_CACHE_ROOT = PROJECT_ROOT / "temp" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_ROOT / "xdg_cache"))

from exp1_similarity_matrices import (  # noqa: E402
    DEFAULT_EMBEDDING_ROOT,
    DEFAULT_SELECTED_RUNS_PATH,
    _parse_run_paths,
    build_qid_vector_maps,
    cosine_similarity_matrix,
    discover_all_runs,
    load_selected_runs,
    plot_similarity_distribution,
    plot_similarity_heatmap,
)


DEFAULT_COMMON_QIDS_PATH = PROJECT_ROOT / "experiments" / "exp1" / "qid_coverage" / "common_qids.tsv"
DEFAULT_NUM_RECORDINGS_PATH = PROJECT_ROOT / "experiments" / "exp1" / "xeno_canto_num_recordings" / "qid_num_recordings.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp1" / "similarity_matrices_ordered"


def _read_qids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [str(row.get("qid", "")).strip() for row in rows if str(row.get("qid", "")).strip()]


def _load_num_recordings(path: Path) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    lookup: dict[str, dict[str, object]] = {}
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if qid:
            lookup[qid] = row
    return lookup


def _select_top_qids(common_qids_path: Path, num_recordings_path: Path, top_k: int) -> list[str]:
    common_qids = _read_qids(common_qids_path)
    num_recordings_lookup = _load_num_recordings(num_recordings_path)
    rows: list[tuple[int, str]] = []
    for qid in common_qids:
        row = num_recordings_lookup.get(qid)
        if not row:
            continue
        rows.append((int(row.get("num_recordings", 0) or 0), qid))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [qid for _, qid in rows[:top_k]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ordered similarity heatmaps for Experiment 1.")
    parser.add_argument("--embedding-root", default=str(DEFAULT_EMBEDDING_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument("--common-qids", default=str(DEFAULT_COMMON_QIDS_PATH))
    parser.add_argument("--num-recordings", default=str(DEFAULT_NUM_RECORDINGS_PATH))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--use-selected-runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-discover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-run", action="append", default=[], help="Graph embedding run directory.")
    parser.add_argument("--language-run", action="append", default=[], help="Language embedding run directory.")
    parser.add_argument("--audio-run", action="append", default=[], help="Audio embedding run directory.")
    return parser


def run(args: argparse.Namespace) -> None:
    embedding_root = Path(args.embedding_root).expanduser()
    if not embedding_root.is_absolute():
        embedding_root = PROJECT_ROOT / embedding_root

    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path

    common_qids_path = Path(args.common_qids).expanduser()
    if not common_qids_path.is_absolute():
        common_qids_path = PROJECT_ROOT / common_qids_path

    num_recordings_path = Path(args.num_recordings).expanduser()
    if not num_recordings_path.is_absolute():
        num_recordings_path = PROJECT_ROOT / num_recordings_path

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
    output_dir.mkdir(parents=True, exist_ok=True)

    qid_order = _select_top_qids(common_qids_path, num_recordings_path, args.top_k)
    if not qid_order:
        raise SystemExit("No common QIDs available for ordering.")

    vector_maps, _ = build_qid_vector_maps(runs)
    missing: list[tuple[str, list[str]]] = []

    for run in runs:
        qid_map = vector_maps[run.name]
        ordered_qids = [qid for qid in qid_order if qid in qid_map]
        if len(ordered_qids) != len(qid_order):
            missing_qids = [qid for qid in qid_order if qid not in qid_map]
            missing.append((run.name, missing_qids))
            continue
        vectors = np.stack([qid_map[qid] for qid in qid_order], axis=0).astype(np.float32)
        matrix = cosine_similarity_matrix(vectors)
        run_dir = output_dir / run.name.replace("/", "__")
        run_dir.mkdir(parents=True, exist_ok=True)
        plot_similarity_heatmap(matrix, tuple(qid_order), run_dir / "heatmap.png", title=run.name)
        plot_similarity_distribution(matrix, run_dir / "distribution.png", title=run.name)

    if missing:
        details = "; ".join(f"{run_name}:{len(qids)} missing" for run_name, qids in missing)
        raise SystemExit(f"Some runs are missing qids from the shared top-{len(qid_order)} order: {details}")

    print(f"ordered_qids: {len(qid_order)}")
    print(f"embedding_runs: {len(runs)}")
    print(f"output_dir: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
