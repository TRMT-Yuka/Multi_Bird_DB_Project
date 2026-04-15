from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from qwikidata.json_dump import WikidataJsonDump

from .config import get_project_paths


def load_qids(input_path: Path) -> list[str]:
    """Read a one-column TSV file of QIDs. / 1 列 TSV から QID 一覧を読む。"""

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return [
            qid
            for row in csv.DictReader(handle, delimiter="\t")
            if (qid := (row.get("qid") or "").strip())
        ]


def qid_to_json_path(output_dir: Path, qid: str) -> Path:
    """Return the hierarchical JSON path for one QID. / 1 件の QID に対応する階層化 JSON パスを返す。"""

    digits = qid[1:]
    first_digit = digits[0] if len(digits) >= 1 else "_"
    second_digit = digits[1] if len(digits) >= 2 else "_"
    return output_dir / first_digit / second_digit / f"{qid}.json"


def write_json_atomically(output_path: Path, entity: dict) -> None:
    """Write one entity JSON atomically so interrupted runs do not leave partial files."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entity, ensure_ascii=False, separators=(",", ":"))
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def extract_entities_from_dump(input_path: Path, dump_path: Path, output_dir: Path) -> int:
    """Materialize per-QID JSON files by scanning the dump directly.

    / dump を直接走査して、QID ごとの JSON を生成する。
    """

    if not dump_path.exists():
        raise FileNotFoundError(f"Dump file does not exist: {dump_path}")

    target_qids = load_qids(input_path)
    if not target_qids:
        return 0

    remaining = set(target_qids)
    written = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for entity in WikidataJsonDump(str(dump_path)):
        qid = str(entity.get("id", "")).strip()
        if qid not in remaining:
            continue
        output_path = qid_to_json_path(output_dir, qid)
        if not output_path.exists():
            write_json_atomically(output_path, entity)
            written += 1
        remaining.discard(qid)
        if not remaining:
            break

    if remaining:
        missing_preview = ", ".join(sorted(remaining)[:20])
        raise ValueError(f"Some QIDs were not found in dump: {missing_preview}")

    return written


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for dump extraction. / dump 抽出コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Extract target Wikidata entity JSON files from the dump.")
    parser.add_argument("--input", default=str(paths.qids_tsv))
    parser.add_argument("--dump", default=str(paths.wikidata_dump_file))
    parser.add_argument("--output-dir", default=str(paths.json_dir))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dump extraction command. / dump 抽出コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    extract_entities_from_dump(Path(args.input), Path(args.dump), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
