from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qwikidata.json_dump import WikidataJsonDump

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


def qid_to_json_path(output_dir: Path, qid: str) -> Path:
    """Return the hierarchical JSON path for one QID. / 1 件の QID に対応する階層化 JSON パスを返す。"""

    digits = qid[1:]
    first_digit = digits[0] if len(digits) >= 1 else "_"
    second_digit = digits[1] if len(digits) >= 2 else "_"
    return output_dir / first_digit / second_digit / f"{qid}.json"


def filter_existing_qids(qids: set[str], output_dir: Path) -> set[str]:
    """Skip QIDs that already have extracted JSON files. / 既存 JSON がある QID を抽出対象から除く。"""

    remaining = set(qids)
    for qid in qids:
        output_path = qid_to_json_path(output_dir, qid)
        if output_path.exists() and output_path.stat().st_size > 0:
            remaining.discard(qid)
    return remaining


def render_progress(scanned: int, written: int, target_total: int, remaining_count: int) -> None:
    """Render one in-place progress line. / 進捗を 1 行上書きで表示する。"""

    message = (
        f"\rDump scan: {scanned:,} entities scanned | "
        f"QIDs extracted: {written:,}/{target_total:,} | "
        f"remaining: {remaining_count:,}"
    )
    sys.stderr.write(message)
    sys.stderr.flush()


def extract_entities_from_dump(input_path: Path, dump_path: Path, output_dir: Path) -> int:
    """Extract only requested QIDs from a full Wikidata dump. / Wikidata 全量 dump から必要な QID だけを書き出す。"""

    target_qids = load_qids(input_path)
    if not dump_path.exists():
        raise FileNotFoundError(f"Dump file does not exist: {dump_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = filter_existing_qids(target_qids, output_dir)
    if not remaining:
        print("All target QIDs already exist. Nothing to extract.", file=sys.stderr)
        return 0
    written = 0
    scanned = 0
    target_total = len(remaining)
    last_progress_time = time.monotonic()
    progress_interval_seconds = 3.0
    print(
        f"Start extracting {target_total} QIDs from dump: {dump_path}",
        file=sys.stderr,
    )
    for entity in WikidataJsonDump(str(dump_path)):
        scanned += 1
        now = time.monotonic()
        if scanned % 100000 == 0:
            render_progress(scanned, written, target_total, len(remaining))
            last_progress_time = now
        elif now - last_progress_time >= progress_interval_seconds:
            render_progress(scanned, written, target_total, len(remaining))
            last_progress_time = now
        qid = entity.get("id", "")
        if qid not in remaining:
            continue
        output_path = qid_to_json_path(output_dir, qid)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(entity, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        remaining.remove(qid)
        written += 1
        render_progress(scanned, written, target_total, len(remaining))
        last_progress_time = time.monotonic()
        if not remaining:
            break
    if remaining:
        sys.stderr.write("\n")
        missing = ", ".join(sorted(remaining)[:20])
        raise ValueError(f"Some QIDs were not found in dump: {missing}")
    sys.stderr.write("\n")
    print(
        f"Completed extraction: scanned {scanned} entities, wrote {written} JSON files.",
        file=sys.stderr,
    )
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
        default=str(paths.wikidata_dump_file),
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
