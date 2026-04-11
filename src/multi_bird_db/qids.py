from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import get_project_paths


def extract_qid(url: str) -> str:
    """Take the final QID part from a Wikidata entity URL. / Wikidata URL の末尾から QID を取り出す。"""

    return url.rstrip("/").rsplit("/", 1)[-1]


def iter_qids(input_path: Path) -> list[str]:
    """Read query.tsv and return unique QIDs in file order. / query.tsv を読み、重複しない QID を順番どおり返す。"""

    qids: list[str] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item_url = (row.get("item") or "").strip()
            if not item_url:
                continue
            qid = extract_qid(item_url)
            if qid not in seen:
                seen.add(qid)
                qids.append(qid)
    return qids


def write_qids(output_path: Path, qids: list[str]) -> None:
    """Write QIDs as a one-column TSV file. / QID 一覧を 1 列の TSV として保存する。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["qid"])
        writer.writerows([[qid] for qid in qids])


def extract_qids(input_path: Path, output_path: Path, root_qid: str = "Q5113") -> None:
    """Build a TSV list of Bird descendant QIDs. / Bird 配下 QID の TSV 一覧を作る。"""

    qids = iter_qids(input_path)
    if root_qid and root_qid not in qids:
        qids.insert(0, root_qid)
    write_qids(output_path, qids)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for QID extraction. / QID 抽出コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Extract Bird descendant QIDs from query.tsv.")
    parser.add_argument("--input", default=str(paths.query_tsv))
    parser.add_argument("--output", default=str(paths.qids_tsv))
    parser.add_argument("--root-qid", default="Q5113")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the QID extraction command. / QID 抽出コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    extract_qids(Path(args.input), Path(args.output), root_qid=args.root_qid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
