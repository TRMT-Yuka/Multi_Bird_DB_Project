from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from exp1_similarity_matrices import _load_json, _load_qids  # noqa: E402


DEFAULT_SELECTED_RUNS_PATH = PROJECT_ROOT / "data" / "external" / "embeddings" / "selected_runs.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp1_qid_coverage"


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


def _load_audio_qids(path: Path) -> set[str]:
    qids_path = path / "qids.json"
    if not qids_path.exists():
        qids_path = path / "qids.partial.json"
    if not qids_path.exists():
        raise FileNotFoundError(f"Missing qids manifest: {qids_path}")
    payload = _load_json(qids_path)
    if isinstance(payload, list):
        return {str(value).strip() for value in payload if str(value).strip()}
    if isinstance(payload, dict) and isinstance(payload.get("qids"), list):
        return {str(value).strip() for value in payload["qids"] if str(value).strip()}
    raise ValueError(f"Unsupported qids.json format: {qids_path}")


def load_run_qids(modality: str, path: Path) -> set[str]:
    if modality in {"graph", "language"}:
        return {qid for qid in _load_qids(path) if qid}
    if modality == "audio":
        return _load_audio_qids(path)
    raise ValueError(f"Unsupported modality: {modality}")


def build_report(selected_runs_path: Path) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    selected = _selected_runs_by_modality(selected_runs_path)
    run_sets: dict[str, dict[str, set[str]]] = {"graph": {}, "language": {}, "audio": {}}

    for modality, entries in selected.items():
        for label, path in sorted(entries.items()):
            run_sets[modality][label] = load_run_qids(modality, path)

    modality_common: dict[str, set[str]] = {}
    for modality, entries in run_sets.items():
        qid_sets = list(entries.values())
        if not qid_sets:
            modality_common[modality] = set()
        else:
            modality_common[modality] = set.intersection(*qid_sets)

    all_common: set[str]
    non_empty_common = [qids for qids in modality_common.values() if qids]
    if len(non_empty_common) == 3:
        all_common = set.intersection(*non_empty_common)
    else:
        all_common = set()

    run_rows: list[dict[str, object]] = []
    for modality, entries in run_sets.items():
        other_modalities = [modality_common[name] for name in run_sets if name != modality]
        other_common = set.intersection(*other_modalities) if all(other_modalities) else set()
        for label, qids in sorted(entries.items()):
            run_rows.append(
                {
                    "modality": modality,
                    "label": label,
                    "path": str(selected[modality][label]),
                    "qid_count": len(qids),
                    "complete_qid_count": len(qids & other_common),
                }
            )

    summary = {
        "experiment": "EXP0-qid-coverage",
        "selected_runs": str(selected_runs_path),
        "run_counts": {modality: len(entries) for modality, entries in run_sets.items()},
        "qid_counts_per_modality_common": {modality: len(qids) for modality, qids in modality_common.items()},
        "complete_qid_count": len(all_common),
    }
    return run_rows, summary, sorted(all_common)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report QID coverage across selected embeddings.")
    parser.add_argument("--selected-runs", default=str(DEFAULT_SELECTED_RUNS_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    selected_runs_path = Path(args.selected_runs).expanduser()
    if not selected_runs_path.is_absolute():
        selected_runs_path = PROJECT_ROOT / selected_runs_path
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    run_rows, summary, common_qids = build_report(selected_runs_path)
    _write_tsv(
        output_dir / "run_qid_counts.tsv",
        run_rows,
        ["modality", "label", "path", "qid_count", "complete_qid_count"],
    )
    _write_tsv(
        output_dir / "common_qids.tsv",
        ({"qid": qid} for qid in common_qids),
        ["qid"],
    )
    _write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
