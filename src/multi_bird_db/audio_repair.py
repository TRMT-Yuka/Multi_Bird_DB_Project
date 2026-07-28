from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .audio_embeddings import (
    DEFAULT_EXTENSIONS,
    DEFAULT_MAX_SECONDS,
    DEFAULT_TARGET_SAMPLE_RATE,
    _finish_progress_line,
    _render_progress_line,
    _timestamp_utc,
    _write_json_atomic,
    load_audio_file,
)
from .audio_finetuning import DEFAULT_RECORDING_MAP_PATH, build_finetune_examples
from .config import get_project_paths
from .xeno_canto_audio import (
    DEFAULT_CLIP_SECONDS,
    copy_audio_file,
    fetch_bytes,
    make_audio_temp_dir,
    probe_audio_duration_seconds,
    recording_download_url,
    write_bytes_atomic,
    clip_audio_file,
)

DEFAULT_REPORT_PATH = get_project_paths().root / "data" / "external" / "models" / "audio" / "wav2vec2-finetuned" / "xeno_canto_audio_repair.tsv"
REPAIR_COLUMNS = [
    "qid",
    "recording_id",
    "audio_path",
    "relative_path",
    "status",
    "reason",
    "download_url",
    "before_size",
    "after_size",
]


@dataclass(frozen=True)
class AudioRepairRow:
    qid: str
    recording_id: str
    audio_path: str
    relative_path: str
    status: str
    reason: str
    download_url: str
    before_size: int
    after_size: int


def _write_repair_tsv(path: Path, rows: list[AudioRepairRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPAIR_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    temp_path.replace(path)


def _decode_check(path: Path, *, sample_rate: int, max_seconds: float) -> tuple[bool, str]:
    try:
        waveform, decoded_sample_rate = load_audio_file(path, target_sample_rate=sample_rate, max_seconds=max_seconds)
        if decoded_sample_rate != sample_rate:
            return False, f"unexpected sample_rate: {decoded_sample_rate}"
        if np.asarray(waveform).size == 0:
            return False, "decoded waveform is empty"
    except Exception as exc:  # noqa: BLE001 - invalid user data should be reported, not crash the audit.
        return False, str(exc).replace("\n", " ").strip()
    return True, ""


def _file_type_for_download(download_url: str, target_path: Path) -> str:
    suffix = Path(urlparse(download_url).path).suffix.lstrip(".").lower()
    if suffix in {"mp3", "wav"}:
        return suffix
    path_suffix = target_path.suffix.lstrip(".").lower()
    if path_suffix in {"mp3", "wav"}:
        return path_suffix
    return "mp3"


def _download_candidate(
    *,
    qid: str,
    recording_id: str,
    target_path: Path,
    download_url: str,
    clip_seconds: int,
) -> Path:
    file_type = _file_type_for_download(download_url, target_path)
    tmp_dir = make_audio_temp_dir(target_path.parents[1], qid, recording_id)
    tmp_input = tmp_dir / f"{recording_id}.{file_type}"
    candidate_path = tmp_dir / f"{recording_id}.candidate.{target_path.suffix.lstrip('.') or file_type}"
    payload = fetch_bytes(download_url)
    write_bytes_atomic(tmp_input, payload)
    duration = probe_audio_duration_seconds(tmp_input)
    if duration is not None and duration <= clip_seconds:
        copy_audio_file(tmp_input, candidate_path)
    else:
        clip_audio_file(tmp_input, candidate_path, file_type=file_type, clip_seconds=clip_seconds)
    return candidate_path


def repair_xeno_canto_audio(
    *,
    input_dir: Path,
    recording_map_path: Path,
    report_path: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
    check_seconds: float = min(DEFAULT_MAX_SECONDS, 1.0),
    clip_seconds: int = DEFAULT_CLIP_SECONDS,
    repair: bool = True,
) -> dict[str, Any]:
    examples = build_finetune_examples(input_dir=input_dir, recording_map_path=recording_map_path, extensions=extensions)
    if not examples:
        raise FileNotFoundError(f"No audio files were found under: {input_dir}")

    rows: list[AudioRepairRow] = []
    valid_count = 0
    invalid_count = 0
    repaired_count = 0
    failed_count = 0
    total = len(examples)

    for index, item in enumerate(examples, start=1):
        path = Path(item.audio_path)
        before_size = path.stat().st_size if path.exists() else 0
        ok, reason = _decode_check(path, sample_rate=sample_rate, max_seconds=check_seconds)
        download_url = item.download_url or recording_download_url(item.recording_id)
        status = "valid"
        after_size = before_size
        if ok:
            valid_count += 1
        else:
            invalid_count += 1
            status = "invalid"
            if repair:
                candidate_path: Path | None = None
                try:
                    candidate_path = _download_candidate(
                        qid=item.qid,
                        recording_id=item.recording_id,
                        target_path=path,
                        download_url=download_url,
                        clip_seconds=clip_seconds,
                    )
                    candidate_ok, candidate_reason = _decode_check(
                        candidate_path,
                        sample_rate=sample_rate,
                        max_seconds=check_seconds,
                    )
                    if not candidate_ok:
                        raise RuntimeError(candidate_reason)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(candidate_path), str(path))
                    after_size = path.stat().st_size
                    status = "repaired"
                    repaired_count += 1
                except Exception as exc:  # noqa: BLE001 - report every failed repair and continue.
                    status = "repair_failed"
                    failed_count += 1
                    reason = f"{reason} | repair failed: {str(exc).replace(chr(10), ' ').strip()}"
                finally:
                    if candidate_path is not None:
                        shutil.rmtree(candidate_path.parent, ignore_errors=True)
            else:
                failed_count += 1
        rows.append(
            AudioRepairRow(
                qid=item.qid,
                recording_id=item.recording_id,
                audio_path=str(path),
                relative_path=item.relative_path,
                status=status,
                reason=reason,
                download_url=download_url,
                before_size=before_size,
                after_size=after_size,
            )
        )
        _render_progress_line(
            f"repair audio {index}/{total} | valid {valid_count} | invalid {invalid_count} | "
            f"repaired {repaired_count} | failed {failed_count} | {item.relative_path}"
        )
    _finish_progress_line(
        f"repair audio done | files {total}/{total} | valid {valid_count} | invalid {invalid_count} | "
        f"repaired {repaired_count} | failed {failed_count}"
    )
    _write_repair_tsv(report_path, rows)
    summary = {
        "kind": "xeno_canto_audio_repair",
        "created_at_utc": _timestamp_utc(),
        "input_dir": str(input_dir),
        "recording_map_path": str(recording_map_path),
        "report_path": str(report_path),
        "audio_file_count": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "repaired_count": repaired_count,
        "failed_count": failed_count,
        "repair_enabled": repair,
        "sample_rate": sample_rate,
        "check_seconds": check_seconds,
        "clip_seconds": clip_seconds,
    }
    _write_json_atomic(report_path.with_suffix(".json"), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Validate and repair corrupted Xeno-canto audio files.")
    parser.add_argument("--input-dir", default=str(paths.xeno_canto_raw_dir))
    parser.add_argument("--recording-map", default=str(DEFAULT_RECORDING_MAP_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_TARGET_SAMPLE_RATE)
    parser.add_argument("--check-seconds", type=float, default=min(DEFAULT_MAX_SECONDS, 1.0))
    parser.add_argument("--clip-seconds", type=int, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--repair", action="store_true", default=True)
    parser.add_argument("--check-only", dest="repair", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    extensions = tuple(ext.strip() for ext in args.extensions.split(",") if ext.strip())
    summary = repair_xeno_canto_audio(
        input_dir=Path(args.input_dir),
        recording_map_path=Path(args.recording_map),
        report_path=Path(args.report),
        extensions=extensions,
        sample_rate=args.sample_rate,
        check_seconds=args.check_seconds,
        clip_seconds=args.clip_seconds,
        repair=args.repair,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
