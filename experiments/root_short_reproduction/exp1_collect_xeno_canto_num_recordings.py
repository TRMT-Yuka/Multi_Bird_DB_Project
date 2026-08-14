from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMP_CACHE_ROOT = PROJECT_ROOT / "temp" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEMP_CACHE_ROOT / "xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


DEFAULT_API_RECORDINGS_ROOT = PROJECT_ROOT / "data" / "interim" / "xeno-canto" / "api_recordings"
DEFAULT_XENO_CANTO_IDS_PATH = PROJECT_ROOT / "data" / "interim" / "wikidata" / "bird_xeno_canto_ids.tsv"
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "graph" / "bird_taxonomy_graph.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "root_short_reproduction" / "exp1_xeno_canto_num_recordings"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_xeno_canto_ids(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    mapping: dict[str, str] = {}
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        xeno_canto_species_id = str(row.get("xeno_canto_species_id", "")).strip()
        if qid:
            mapping[qid] = xeno_canto_species_id
    return mapping


def _read_japanese_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    lookup: dict[str, str] = {}
    for qid, attrs in getattr(graph, "nodes", lambda data=False: [])(data=True):
        label_ja = str((attrs or {}).get("label_ja") or (attrs or {}).get("ja_name") or "").strip()
        if qid and label_ja:
            lookup[str(qid)] = label_ja
    return lookup


def _collect_rows(
    api_recordings_root: Path,
    xeno_canto_ids_path: Path,
    graph_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    qid_to_species_id = _read_xeno_canto_ids(xeno_canto_ids_path)
    qid_to_japanese_name = _read_japanese_labels(graph_path)
    rows: list[dict[str, object]] = []
    missing_page001: list[str] = []

    if not api_recordings_root.exists():
        raise FileNotFoundError(f"API recordings root does not exist: {api_recordings_root}")

    for qid_dir in sorted(path for path in api_recordings_root.iterdir() if path.is_dir()):
        qid = qid_dir.name.strip()
        page001 = qid_dir / "page001.json"
        if not page001.exists():
            missing_page001.append(qid)
            continue
        payload = _load_json(page001)
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported JSON payload in {page001}")
        recordings = payload.get("recordings") or []
        if not isinstance(recordings, list):
            recordings = []
        num_recordings = int(payload.get("numRecordings", len(recordings)) or len(recordings))
        num_pages = int(payload.get("numPages", 1) or 1)
        rows.append(
            {
                "qid": qid,
                "xeno_canto_species_id": qid_to_species_id.get(qid, ""),
                "ja_name": qid_to_japanese_name.get(qid, ""),
                "num_recordings": num_recordings,
                "num_pages": num_pages,
                "page001_recordings": len(recordings),
                "page001_path": str(page001),
            }
        )

    rows.sort(key=lambda row: (-int(row["num_recordings"]), str(row["qid"])))
    summary = {
        "experiment": "EXP1-xeno-canto-num-recordings",
        "api_recordings_root": str(api_recordings_root),
        "xeno_canto_ids_path": str(xeno_canto_ids_path),
        "qid_count": len(rows),
        "total_num_recordings": sum(int(row["num_recordings"]) for row in rows),
        "max_num_recordings": max((int(row["num_recordings"]) for row in rows), default=0),
        "min_num_recordings": min((int(row["num_recordings"]) for row in rows), default=0),
        "missing_page001_count": len(missing_page001),
        "missing_page001_qids": missing_page001,
    }
    return rows, summary


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _plot_num_recordings_distribution(rows: list[dict[str, object]], output_path: Path) -> None:
    values = [int(row["num_recordings"]) for row in rows]
    if not values:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    ax.hist(values, bins=60, color="#4C78A8", edgecolor="white")
    ax.set_title("Distribution of xeno-canto numRecordings per QID")
    ax.set_xlabel("numRecordings")
    ax.set_ylabel("QID count")
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.12)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_num_recordings_distribution_log(rows: list[dict[str, object]], output_path: Path) -> None:
    values = [int(row["num_recordings"]) for row in rows if int(row["num_recordings"]) > 0]
    if not values:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    ax.hist(values, bins=60, color="#F58518", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Distribution of xeno-canto numRecordings per QID (log-scale)")
    ax.set_xlabel("numRecordings")
    ax.set_ylabel("QID count (log)")
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which="major", axis="both", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.12)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect xeno-canto numRecordings from local page001.json files.")
    parser.add_argument("--api-recordings-root", default=str(DEFAULT_API_RECORDINGS_ROOT))
    parser.add_argument("--xeno-canto-ids", default=str(DEFAULT_XENO_CANTO_IDS_PATH))
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    api_recordings_root = Path(args.api_recordings_root).expanduser()
    if not api_recordings_root.is_absolute():
        api_recordings_root = PROJECT_ROOT / api_recordings_root
    xeno_canto_ids_path = Path(args.xeno_canto_ids).expanduser()
    if not xeno_canto_ids_path.is_absolute():
        xeno_canto_ids_path = PROJECT_ROOT / xeno_canto_ids_path
    graph_path = Path(args.graph).expanduser()
    if not graph_path.is_absolute():
        graph_path = PROJECT_ROOT / graph_path
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    rows, summary = _collect_rows(api_recordings_root, xeno_canto_ids_path, graph_path)
    _write_tsv(
        output_dir / "qid_num_recordings.tsv",
        rows,
        [
            "qid",
            "xeno_canto_species_id",
            "ja_name",
            "num_recordings",
            "num_pages",
            "page001_recordings",
            "page001_path",
        ],
    )
    _write_json(output_dir / "summary.json", summary)
    _write_tsv(
        output_dir / "missing_page001_qids.tsv",
        ({"qid": qid} for qid in summary["missing_page001_qids"]),
        ["qid"],
    )
    _write_tsv(
        output_dir / "top_100_qids.tsv",
        (
            {
                "rank": index + 1,
                "qid": row["qid"],
                "xeno_canto_species_id": row["xeno_canto_species_id"],
                "ja_name": row["ja_name"],
                "num_recordings": row["num_recordings"],
                "page001_recordings": row["page001_recordings"],
                "num_pages": row["num_pages"],
            }
            for index, row in enumerate(rows[:100])
        ),
        [
            "rank",
            "qid",
            "xeno_canto_species_id",
            "ja_name",
            "num_recordings",
            "page001_recordings",
            "num_pages",
        ],
    )
    _plot_num_recordings_distribution(rows, output_dir / "num_recordings_distribution.png")
    _plot_num_recordings_distribution_log(rows, output_dir / "num_recordings_distribution_log.png")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
