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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp1_img"
DEFAULT_SELECTED_RUNS_PATH = DEFAULT_EMBEDDING_ROOT / "selected_runs.json"


@dataclass(frozen=True)
class EmbeddingRun:
    """One embedding output directory that can provide one vector per selected QID."""

    name: str
    modality: str
    path: Path


@dataclass(frozen=True)
class MatrixInput:
    """Aligned vectors for one embedding run."""

    run: EmbeddingRun
    qids: tuple[str, ...]
    vectors: np.ndarray


@dataclass(frozen=True)
class AudioFileChoice:
    qid: str
    relative_path: str


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: Iterable[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _load_qids(run_dir: Path) -> list[str]:
    payload = _load_json(run_dir / "qids.json")
    if isinstance(payload, list):
        return [str(value).strip() for value in payload]
    if isinstance(payload, dict) and isinstance(payload.get("qids"), list):
        return [str(value).strip() for value in payload["qids"]]
    raise ValueError(f"Unsupported qids.json format: {run_dir / 'qids.json'}")


def _load_embeddings(run_dir: Path) -> np.ndarray:
    matrix = np.load(run_dir / "embeddings.npy")
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embeddings.npy: {run_dir / 'embeddings.npy'}")
    return np.asarray(matrix, dtype=np.float32)




def _iter_selected_runs(payload: object) -> list[EmbeddingRun]:
    if not isinstance(payload, dict):
        raise ValueError("selected_runs.json must contain an object.")
    runs = payload.get("runs", {})
    if not isinstance(runs, dict):
        raise ValueError("selected_runs.json must contain a runs object.")

    selected: list[EmbeddingRun] = []
    for modality in ("graph", "language", "audio"):
        entries = runs.get(modality, {})
        if not isinstance(entries, dict):
            raise ValueError(f"selected_runs.json runs.{modality} must be an object.")
        for label, value in entries.items():
            run_dir = Path(str(value)).expanduser()
            if not run_dir.is_absolute():
                run_dir = PROJECT_ROOT / run_dir
            selected.append(EmbeddingRun(name=f"{modality}__{label}", modality=modality, path=run_dir))
    return selected


def load_selected_runs(path: Path) -> list[EmbeddingRun]:
    payload = _load_json(path)
    return _iter_selected_runs(payload)

def discover_graph_runs(root: Path = DEFAULT_EMBEDDING_ROOT) -> list[EmbeddingRun]:
    graph_root = root / "graph"
    runs: list[EmbeddingRun] = []
    if not graph_root.exists():
        return runs
    for run_dir in sorted(graph_root.glob("*/*")):
        if (run_dir / "embeddings.npy").exists() and (run_dir / "qids.json").exists():
            method = run_dir.parent.name
            tag = run_dir.name
            runs.append(EmbeddingRun(name=f"graph__{method}__{tag}", modality="graph", path=run_dir))
    return runs


def discover_language_runs(root: Path = DEFAULT_EMBEDDING_ROOT) -> list[EmbeddingRun]:
    language_root = root / "language"
    runs: list[EmbeddingRun] = []
    if not language_root.exists():
        return runs
    for run_dir in sorted(language_root.glob("*")):
        if (run_dir / "embeddings.npy").exists() and (run_dir / "qids.json").exists():
            runs.append(EmbeddingRun(name=f"language__{run_dir.name}", modality="language", path=run_dir))
    return runs


def discover_audio_runs(root: Path = DEFAULT_EMBEDDING_ROOT) -> list[EmbeddingRun]:
    audio_root = root / "audio"
    runs: list[EmbeddingRun] = []
    if not audio_root.exists():
        return runs
    for run_dir in sorted(audio_root.glob("*/*/*")):
        if (
            (run_dir / "embeddings.npy").exists()
            and (run_dir / "qids.json").exists()
            and (run_dir / "audio_manifest.tsv").exists()
        ):
            backend = run_dir.parents[1].name
            model = run_dir.parent.name
            tag = run_dir.name
            runs.append(EmbeddingRun(name=f"audio__{backend}__{model}__{tag}", modality="audio", path=run_dir))
    return runs


def discover_all_runs(root: Path = DEFAULT_EMBEDDING_ROOT) -> list[EmbeddingRun]:
    return discover_graph_runs(root) + discover_language_runs(root) + discover_audio_runs(root)


def load_graph_vectors(run: EmbeddingRun) -> dict[str, np.ndarray]:
    qids = _load_qids(run.path)
    vectors = _load_embeddings(run.path)
    if len(qids) != len(vectors):
        raise ValueError(f"qids/embedding row mismatch in {run.path}: {len(qids)} != {len(vectors)}")
    result: dict[str, np.ndarray] = {}
    for qid, vector in zip(qids, vectors, strict=True):
        if qid and qid not in result:
            result[qid] = vector
    return result


def _language_row_sort_key(row: dict[str, str], fallback_index: int) -> tuple[int, str, str]:
    ordinal = row.get("ordinal", "")
    try:
        ordinal_value = int(ordinal)
    except ValueError:
        ordinal_value = fallback_index
    return (ordinal_value, row.get("surface_id", ""), row.get("surface_text", ""))


def load_language_vectors(run: EmbeddingRun) -> dict[str, np.ndarray]:
    qids = _load_qids(run.path)
    vectors = _load_embeddings(run.path)
    if len(qids) != len(vectors):
        raise ValueError(f"qids/embedding row mismatch in {run.path}: {len(qids)} != {len(vectors)}")

    manifest_path = run.path / "surface_manifest.tsv"
    if manifest_path.exists():
        rows = _read_tsv(manifest_path)
        if len(rows) == len(vectors):
            order = sorted(range(len(rows)), key=lambda index: (rows[index].get("qid", ""), _language_row_sort_key(rows[index], index)))
        else:
            order = list(range(len(vectors)))
    else:
        order = list(range(len(vectors)))

    result: dict[str, np.ndarray] = {}
    for index in order:
        qid = qids[index]
        if qid and qid not in result:
            result[qid] = vectors[index]
    return result


def load_audio_file_vectors(run: EmbeddingRun) -> dict[tuple[str, str], np.ndarray]:
    """Return one vector per (QID, audio relative_path).

    BirdNET-style outputs can contain multiple windows per audio file. Those rows are
    averaged so each backend contributes one vector for the selected representative file.
    """

    vectors = _load_embeddings(run.path)
    rows = _read_tsv(run.path / "audio_manifest.tsv")
    if len(rows) != len(vectors):
        raise ValueError(f"audio manifest/embedding row mismatch in {run.path}: {len(rows)} != {len(vectors)}")

    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for row, vector in zip(rows, vectors, strict=True):
        qid = row.get("qid", "").strip()
        relative_path = row.get("relative_path", "").strip()
        if not qid or not relative_path:
            continue
        grouped.setdefault((qid, relative_path), []).append(vector)

    return {key: np.mean(np.stack(values, axis=0), axis=0).astype(np.float32) for key, values in grouped.items()}


def choose_representative_audio_files(audio_maps: dict[str, dict[tuple[str, str], np.ndarray]]) -> dict[str, AudioFileChoice]:
    if not audio_maps:
        return {}
    common_pairs: set[tuple[str, str]] | None = None
    for file_map in audio_maps.values():
        keys = set(file_map)
        common_pairs = keys if common_pairs is None else common_pairs & keys
    if not common_pairs:
        return {}

    by_qid: dict[str, list[str]] = {}
    for qid, relative_path in common_pairs:
        by_qid.setdefault(qid, []).append(relative_path)

    choices: dict[str, AudioFileChoice] = {}
    for qid, relative_paths in by_qid.items():
        choices[qid] = AudioFileChoice(qid=qid, relative_path=sorted(relative_paths)[0])
    return choices


def build_qid_vector_maps(runs: list[EmbeddingRun]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, AudioFileChoice]]:
    vector_maps: dict[str, dict[str, np.ndarray]] = {}
    audio_file_maps: dict[str, dict[tuple[str, str], np.ndarray]] = {}

    for run in runs:
        if run.modality == "graph":
            vector_maps[run.name] = load_graph_vectors(run)
        elif run.modality == "language":
            vector_maps[run.name] = load_language_vectors(run)
        elif run.modality == "audio":
            audio_file_maps[run.name] = load_audio_file_vectors(run)
        else:
            raise ValueError(f"Unsupported modality: {run.modality}")

    audio_choices = choose_representative_audio_files(audio_file_maps)
    for run_name, file_map in audio_file_maps.items():
        qid_vectors: dict[str, np.ndarray] = {}
        for qid, choice in audio_choices.items():
            vector = file_map.get((qid, choice.relative_path))
            if vector is not None:
                qid_vectors[qid] = vector
        vector_maps[run_name] = qid_vectors

    return vector_maps, audio_choices


def select_complete_qids(vector_maps: dict[str, dict[str, np.ndarray]]) -> list[str]:
    if not vector_maps:
        return []
    qids: set[str] | None = None
    for qid_map in vector_maps.values():
        current = set(qid_map)
        qids = current if qids is None else qids & current
    return sorted(qids or set())


def align_matrix_inputs(runs: list[EmbeddingRun], vector_maps: dict[str, dict[str, np.ndarray]], qids: list[str]) -> list[MatrixInput]:
    aligned: list[MatrixInput] = []
    for run in runs:
        qid_map = vector_maps[run.name]
        vectors = np.stack([qid_map[qid] for qid in qids], axis=0).astype(np.float32)
        aligned.append(MatrixInput(run=run, qids=tuple(qids), vectors=vectors))
    return aligned


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = vectors / norms
    matrix = normalized @ normalized.T
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32)


def save_similarity_outputs(inputs: list[MatrixInput], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in inputs:
        name = _safe_name(item.run.name)
        matrix = cosine_similarity_matrix(item.vectors)
        np.save(output_dir / f"{name}_similarity.npy", matrix)
        plot_similarity_heatmap(matrix, item.qids, output_dir / f"{name}_heatmap.png", title=item.run.name)
        plot_similarity_distribution(matrix, output_dir / f"{name}_distribution.png", title=item.run.name)


def plot_similarity_heatmap(matrix: np.ndarray, qids: tuple[str, ...], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("QID index")
    ax.set_ylabel("QID index")
    if len(qids) <= 50:
        ax.set_xticks(range(len(qids)))
        ax.set_yticks(range(len(qids)))
        ax.set_xticklabels(qids, rotation=90, fontsize=6)
        ax.set_yticklabels(qids, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_similarity_distribution(matrix: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    upper = matrix[np.triu_indices_from(matrix, k=1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(upper, bins=80, color="#3a6ea5", alpha=0.85)
    ax.set_title(f"{title} similarity distribution")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("pair count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parse_run_paths(values: list[str], modality: str) -> list[EmbeddingRun]:
    runs: list[EmbeddingRun] = []
    for value in values:
        run_dir = Path(value).expanduser()
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        runs.append(EmbeddingRun(name=f"{modality}__{run_dir.parent.name}__{run_dir.name}", modality=modality, path=run_dir))
    return runs


def _summarize_inputs(
    *,
    runs: list[EmbeddingRun],
    vector_maps: dict[str, dict[str, np.ndarray]],
    qids: list[str],
    audio_choices: dict[str, AudioFileChoice],
    output_dir: Path,
) -> None:
    rows = []
    for run in runs:
        vectors = vector_maps[run.name]
        example_vector = next(iter(vectors.values())) if vectors else None
        rows.append(
            {
                "run_name": run.name,
                "modality": run.modality,
                "path": str(run.path),
                "qid_count_before_intersection": len(vectors),
                "vector_dim": "" if example_vector is None else int(example_vector.shape[0]),
            }
        )
    _write_tsv(
        output_dir / "embedding_run_summary.tsv",
        rows,
        ["run_name", "modality", "path", "qid_count_before_intersection", "vector_dim"],
    )
    _write_json(output_dir / "matrix_qids.json", qids)
    _write_tsv(
        output_dir / "representative_audio_files.tsv",
        (
            {"qid": qid, "relative_path": choice.relative_path}
            for qid, choice in sorted(audio_choices.items())
            if qid in set(qids)
        ),
        ["qid", "relative_path"],
    )
    _write_json(
        output_dir / "metadata.json",
        {
            "complete_qid_count": len(qids),
            "run_count": len(runs),
            "runs": [
                {
                    "name": run.name,
                    "modality": run.modality,
                    "path": str(run.path),
                }
                for run in runs
            ],
            "audio_representative_policy": (
                "For each QID, use audio files that exist in every audio embedding run, "
                "sort their relative paths lexicographically, select the first file, "
                "and average window-level vectors within that file when necessary."
            ),
            "language_representative_policy": (
                "For each QID and language embedding run, select the first surface by "
                "surface_manifest ordinal/surface_id order when multiple surfaces exist."
            ),
            "similarity": "cosine similarity",
        },
    )


def run(args: argparse.Namespace) -> None:
    embedding_root = Path(args.embedding_root).expanduser()
    if not embedding_root.is_absolute():
        embedding_root = PROJECT_ROOT / embedding_root

    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path

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

    vector_maps, audio_choices = build_qid_vector_maps(runs)
    qids = select_complete_qids(vector_maps)
    if args.max_qids is not None:
        qids = qids[: args.max_qids]
    if not qids:
        raise SystemExit("No complete QIDs remain after intersecting all embedding runs.")

    aligned = align_matrix_inputs(runs, vector_maps, qids)
    _summarize_inputs(runs=runs, vector_maps=vector_maps, qids=qids, audio_choices=audio_choices, output_dir=output_dir)
    save_similarity_outputs(aligned, output_dir)

    print(f"complete_qids: {len(qids)}")
    print(f"embedding_runs: {len(runs)}")
    print(f"output_dir: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build QID-aligned similarity matrices for root_short Experiment 1. "
            "The script intersects all selected embedding runs and excludes QIDs "
            "missing any embedding."
        )
    )
    parser.add_argument("--embedding-root", default=str(DEFAULT_EMBEDDING_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument("--use-selected-runs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-discover", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-run", action="append", default=[], help="Graph embedding run directory.")
    parser.add_argument("--language-run", action="append", default=[], help="Language embedding run directory.")
    parser.add_argument("--audio-run", action="append", default=[], help="Audio embedding run directory.")
    parser.add_argument("--max-qids", type=int, default=None, help="Optional debug cap after complete-QID selection.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
