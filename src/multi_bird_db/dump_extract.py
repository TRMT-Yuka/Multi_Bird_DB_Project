from __future__ import annotations

import argparse
import bz2
import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from .config import get_project_paths


def load_qids(input_path: Path) -> set[str]:
    """Load target QIDs from a one-column TSV. / 1 列 TSV から対象 QID 集合を読む。"""

    qids: set[str] = set()
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        next(handle, None)
        for line in handle:
            qid = line.strip().split("\t", 1)[0]
            if qid:
                qids.add(qid)
    return qids


def open_dump_text(dump_path: Path) -> TextIO:
    """Open a Wikidata dump as UTF-8 text. / Wikidata dump を UTF-8 テキストとして開く。"""

    suffixes = dump_path.suffixes
    if suffixes[-2:] == [".json", ".bz2"] or dump_path.suffix == ".bz2":
        return bz2.open(dump_path, "rt", encoding="utf-8")
    if suffixes[-2:] == [".json", ".gz"] or dump_path.suffix == ".gz":
        return gzip.open(dump_path, "rt", encoding="utf-8")
    return dump_path.open("r", encoding="utf-8")


def iter_dump_entities(dump_path: Path) -> Iterator[dict]:
    """Yield entity dictionaries from a Wikidata JSON dump. / Wikidata JSON dump から entity 辞書を順に返す。"""

    with open_dump_text(dump_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line in {"[", "]"}:
                continue
            if line.endswith(","):
                line = line[:-1]
            entity = json.loads(line)
            if isinstance(entity, dict) and entity.get("id"):
                yield entity


def extract_entities_from_dump(input_path: Path, dump_path: Path, output_dir: Path) -> int:
    """Extract only requested QIDs from a full Wikidata dump. / Wikidata 全量 dump から必要な QID だけを書き出す。"""

    target_qids = load_qids(input_path)
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump file does not exist: {dump_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    remaining = set(target_qids)
    for entity in iter_dump_entities(dump_path):
        qid = entity.get("id", "")
        if qid not in remaining:
            continue
        (output_dir / f"{qid}.json").write_text(
            json.dumps(entity, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        remaining.remove(qid)
        written += 1
        if not remaining:
            break
    if remaining:
        missing = ", ".join(sorted(remaining)[:20])
        raise ValueError(f"Some QIDs were not found in dump: {missing}")
    return written


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for dump extraction. / dump 抽出コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(
        description="Extract requested Wikidata entity JSON files from a downloaded dump."
    )
    parser.add_argument("--input", default=str(paths.qids_tsv))
    parser.add_argument(
        "--dump",
        default=str(paths.raw_wikidata_dir / "latest-all.json.bz2"),
        help="Path to a downloaded Wikidata JSON dump such as latest-all.json.bz2.",
    )
    parser.add_argument("--output-dir", default=str(paths.json_dir))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dump extraction command. / dump 抽出コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    extract_entities_from_dump(Path(args.input), Path(args.dump), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
