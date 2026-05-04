from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

from .config import get_project_paths


def load_ontology(ontology_path: Path) -> list[dict[str, Any]]:
    """Load ontology PKL rows. / ontology PKL の行一覧を読む。"""

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file does not exist: {ontology_path}")
    with ontology_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Ontology PKL must contain a list, got: {type(data).__name__}")
    return data


def extract_rows(ontology_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract qid and Xeno-canto ID pairs. / qid と Xeno-canto ID の組を抜き出す。"""

    rows: list[dict[str, str]] = []
    for row in ontology_rows:
        qid = str(row.get("qid") or row.get("id") or "").strip()
        xeno_canto_species_id = str(row.get("xeno_canto_species_id") or "").strip()
        if not qid or not xeno_canto_species_id:
            continue
        rows.append({"qid": qid, "xeno_canto_species_id": xeno_canto_species_id})
    return rows


def extract_xeno_canto_ids(ontology_path: Path, output_path: Path) -> None:
    """Write qid-to-Xeno-canto ID pairs as TSV. / qid と Xeno-canto ID の対応を TSV で保存する。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = extract_rows(load_ontology(ontology_path))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["qid", "xeno_canto_species_id"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser. / CLI パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Extract Xeno-canto species IDs from bird ontology PKL.")
    parser.add_argument("--input", default=str(paths.ontology_pkl))
    parser.add_argument("--output", default=str(paths.xeno_canto_ids_tsv))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the extractor. / 抽出コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    extract_xeno_canto_ids(Path(args.input), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
