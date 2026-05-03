from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
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


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load completed QIDs from a checkpoint file. / チェックポイントから完了済み QID を読む。"""

    if not checkpoint_path.exists():
        return set()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completed = payload.get("completed_qids", [])
    if not isinstance(completed, list):
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    return {str(qid).strip() for qid in completed if str(qid).strip()}


def save_checkpoint(checkpoint_path: Path, completed_qids: set[str]) -> None:
    """Persist completed QIDs atomically. / 完了済み QID を原子的に保存する。"""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed_qids": sorted(completed_qids),
    }
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(checkpoint_path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def render_progress(scanned: int, completed_count: int, target_total: int, remaining_count: int) -> None:
    """Render one in-place progress line. / 進捗を 1 行上書きで表示する。"""

    message = (
        f"\rDump scan: {scanned:,} entities scanned | "
        f"QIDs extracted: {completed_count:,}/{target_total:,} | "
        f"remaining: {remaining_count:,}"
    )
    sys.stderr.write(message)
    sys.stderr.flush()


def load_completed_outputs(target_qids: list[str], output_dir: Path) -> set[str]:
    """Return QIDs that already have extracted JSON files. / 既に抽出済みの QID を返す。"""

    completed_qids: set[str] = set()
    for qid in target_qids:
        if qid_to_json_path(output_dir, qid).exists():
            completed_qids.add(qid)
    return completed_qids


def extract_entities_from_dump(
    input_path: Path,
    dump_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
) -> int:
    """Materialize per-QID JSON files by scanning the dump directly.

    / dump を直接走査して、QID ごとの JSON を生成する。
    """

    if not dump_path.exists():
        raise FileNotFoundError(f"Dump file does not exist: {dump_path}")

    target_qids = load_qids(input_path)
    if not target_qids:
        return 0

    if checkpoint_path is None:
        checkpoint_path = get_project_paths().dump_extract_checkpoint

    completed_qids = load_checkpoint(checkpoint_path)
    completed_qids.update(load_completed_outputs(target_qids, output_dir))
    remaining_qids = set(target_qids)
    remaining_qids.difference_update(completed_qids)
    if not remaining_qids:
        print("All target QIDs already exist. Nothing to extract.", file=sys.stderr)
        return 0
    written = 0
    scanned = 0
    target_total = len(target_qids)
    last_progress_time = time.monotonic()
    progress_interval_seconds = 3.0
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint_path, completed_qids)
    print(
        f"Start extracting {len(remaining_qids):,} QIDs from dump: {dump_path}",
        file=sys.stderr,
    )

    for entity in WikidataJsonDump(str(dump_path)):
        scanned += 1
        now = time.monotonic()
        if scanned % 100000 == 0 or now - last_progress_time >= progress_interval_seconds:
            render_progress(scanned, len(completed_qids), target_total, len(remaining_qids))
            last_progress_time = now
        qid = str(entity.get("id", "")).strip()
        if qid not in remaining_qids:
            continue
        output_path = qid_to_json_path(output_dir, qid)
        if not output_path.exists():
            write_json_atomically(output_path, entity)
            written += 1
        completed_qids.add(qid)
        remaining_qids.discard(qid)
        save_checkpoint(checkpoint_path, completed_qids)
        render_progress(scanned, len(completed_qids), target_total, len(remaining_qids))
        last_progress_time = time.monotonic()
        if not remaining_qids:
            break

    if remaining_qids:
        sys.stderr.write("\n")
        missing_preview = ", ".join(sorted(remaining_qids)[:20])
        raise ValueError(f"Some QIDs were not found in dump: {missing_preview}")

    save_checkpoint(checkpoint_path, completed_qids)
    sys.stderr.write("\n")
    print(
        f"Completed extraction: scanned {scanned:,} entities, wrote {written:,} JSON files.",
        file=sys.stderr,
    )
    return written


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for dump extraction. / dump 抽出コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Extract target Wikidata entity JSON files from the dump.")
    parser.add_argument("--input", default=str(paths.qids_tsv))
    parser.add_argument("--dump", default=str(paths.wikidata_dump_file))
    parser.add_argument("--output-dir", default=str(paths.json_dir))
    parser.add_argument("--checkpoint", default=str(paths.dump_extract_checkpoint))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dump extraction command. / dump 抽出コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    extract_entities_from_dump(Path(args.input), Path(args.dump), Path(args.output_dir), Path(args.checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
