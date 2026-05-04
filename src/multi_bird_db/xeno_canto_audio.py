from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import get_project_paths

USER_AGENT = "Multi_Bird_DB_Project/0.1 (research and educational use; contact: local-project)"
DEFAULT_SINCE_DATE = date(2025, 5, 1)
DEFAULT_LIMIT_PER_QID = 10
DEFAULT_MAX_PAGES = 20
DEFAULT_SLEEP_SECONDS = 0.25
MANIFEST_COLUMNS = [
    "audio_id",
    "qid",
    "xeno_canto_species_id",
    "ordinal",
    "recording_id",
    "uploaded",
    "file_type",
    "species_page_url",
    "recording_page_url",
    "download_url",
    "local_path",
]

RECORDING_ID_RE = re.compile(r"\bXC(\d{3,7})\b")
UPLOADED_RE = re.compile(r"Uploaded\s*\|\s*([0-9]{4}-[0-9]{2}-[0-9]{2})")
FILE_TYPE_RE = re.compile(r"File type\s*\|\s*([A-Za-z0-9]+)")


def load_ontology(ontology_path: Path) -> list[dict[str, Any]]:
    """Load ontology PKL rows. / ontology PKL の行一覧を読む。"""

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file does not exist: {ontology_path}")
    with ontology_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Ontology PKL must contain a list, got: {type(data).__name__}")
    return data


def extract_targets(ontology_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract qid-to-Xeno-canto species ID pairs. / qid と Xeno-canto species ID の組を抜き出す。"""

    rows: list[dict[str, str]] = []
    for row in ontology_rows:
        qid = str(row.get("qid") or row.get("id") or "").strip()
        xeno_canto_species_id = str(row.get("xeno_canto_species_id") or "").strip()
        if not qid or not xeno_canto_species_id:
            continue
        rows.append({"qid": qid, "xeno_canto_species_id": xeno_canto_species_id})
    return rows


def fetch_text(url: str) -> str:
    """Fetch one HTML page as UTF-8 text. / 1 件の HTML ページを UTF-8 テキストで読む。"""

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_bytes(url: str) -> bytes:
    """Fetch one binary response. / 1 件のバイナリ応答を読む。"""

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return response.read()


def species_page_url(species_id: str, view: int) -> str:
    """Build a Xeno-canto species listing URL. / Xeno-canto species 一覧 URL を作る。"""

    return f"https://xeno-canto.org/species/{species_id}?order=rec&pg={view}"


def recording_page_url(recording_id: str) -> str:
    """Build a Xeno-canto recording URL. / Xeno-canto 録音ページ URL を作る。"""

    return f"https://xeno-canto.org/{recording_id.removeprefix('XC')}"


def recording_download_url(recording_id: str) -> str:
    """Build a Xeno-canto download URL. / Xeno-canto 音声ダウンロード URL を作る。"""

    return f"{recording_page_url(recording_id)}/download"


def extract_recording_ids(species_html: str) -> list[str]:
    """Extract recording IDs from a species listing page. / species 一覧ページから recording ID を抜き出す。"""

    recording_ids: list[str] = []
    seen: set[str] = set()
    for match in RECORDING_ID_RE.finditer(species_html):
        recording_id = f"XC{match.group(1)}"
        if recording_id in seen:
            continue
        seen.add(recording_id)
        recording_ids.append(recording_id)
    return recording_ids


def parse_recording_page(recording_html: str) -> dict[str, str]:
    """Extract the upload date and file type from a recording page. / 録音ページから投稿日とファイル種別を読む。"""

    uploaded_match = UPLOADED_RE.search(recording_html)
    file_type_match = FILE_TYPE_RE.search(recording_html)
    uploaded = uploaded_match.group(1) if uploaded_match else ""
    file_type = (file_type_match.group(1) if file_type_match else "").strip().lower()
    return {"uploaded": uploaded, "file_type": file_type}


def write_text_atomic(output_path: Path, text: str) -> None:
    """Write one text file atomically. / テキストを原子的に書く。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def write_bytes_atomic(output_path: Path, payload: bytes) -> None:
    """Write one binary file atomically. / バイナリを原子的に書く。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
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


def write_json_atomic(output_path: Path, payload: Any) -> None:
    """Write JSON atomically. / JSON を原子的に書く。"""

    write_text_atomic(output_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def parse_date(value: str) -> date:
    """Parse an ISO formatted date. / ISO 形式の日付を読む。"""

    return date.fromisoformat(value)


def ensure_query_rows(ontology_path: Path) -> list[dict[str, str]]:
    """Load qids and species IDs from ontology PKL. / ontology PKL から qid と species ID を読む。"""

    return extract_targets(load_ontology(ontology_path))


def download_recording(recording_id: str, local_path: Path, download_bytes_fn: Callable[[str], bytes]) -> None:
    """Download one recording if the target file does not exist yet. / 1 件の録音を必要に応じて取得する。"""

    if local_path.exists() and local_path.stat().st_size > 0:
        return
    payload = download_bytes_fn(recording_download_url(recording_id))
    write_bytes_atomic(local_path, payload)


def fetch_qid_audio(
    qid: str,
    xeno_canto_species_id: str,
    output_dir: Path,
    since_date: date,
    limit_per_qid: int,
    max_pages: int,
    sleep_seconds: float,
    fetch_text_fn: Callable[[str], str],
    download_bytes_fn: Callable[[str], bytes],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Fetch one qid's audio recordings from Xeno-canto. / 1 QID 分の音声を Xeno-canto から取得する。"""

    qid_dir = output_dir / qid
    qid_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seen_recording_ids: set[str] = set()
    pages_scanned = 0
    for view in range(1, max_pages + 1):
        pages_scanned = view
        species_url = species_page_url(xeno_canto_species_id, view)
        try:
            species_html = fetch_text_fn(species_url)
        except (HTTPError, URLError) as exc:
            return rows, {
                "qid": qid,
                "xeno_canto_species_id": xeno_canto_species_id,
                "status": "species-page-error",
                "error": str(exc),
                "pages_scanned": pages_scanned,
            }
        recording_ids = extract_recording_ids(species_html)
        if not recording_ids:
            break
        new_ids_on_page = 0
        for recording_id in recording_ids:
            if recording_id in seen_recording_ids:
                continue
            seen_recording_ids.add(recording_id)
            new_ids_on_page += 1
            if len(rows) >= limit_per_qid:
                break
            recording_url = recording_page_url(recording_id)
            try:
                recording_html = fetch_text_fn(recording_url)
            except (HTTPError, URLError):
                continue
            recording_meta = parse_recording_page(recording_html)
            uploaded_text = recording_meta["uploaded"]
            file_type = recording_meta["file_type"] or "mp3"
            if not uploaded_text:
                continue
            try:
                uploaded_date = parse_date(uploaded_text)
            except ValueError:
                continue
            if uploaded_date < since_date:
                continue
            ordinal = len(rows)
            audio_id = f"{qid}_xeno_canto_{ordinal}"
            local_path = qid_dir / f"{recording_id}.{file_type}"
            try:
                download_recording(recording_id, local_path, download_bytes_fn)
            except (HTTPError, URLError):
                continue
            rows.append(
                {
                    "audio_id": audio_id,
                    "qid": qid,
                    "xeno_canto_species_id": xeno_canto_species_id,
                    "ordinal": str(ordinal),
                    "recording_id": recording_id,
                    "uploaded": uploaded_text,
                    "file_type": file_type,
                    "species_page_url": species_url,
                    "recording_page_url": recording_url,
                    "download_url": recording_download_url(recording_id),
                    "local_path": str(local_path),
                }
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if len(rows) >= limit_per_qid:
                break
        if len(rows) >= limit_per_qid:
            break
        if new_ids_on_page == 0:
            break
    return rows, {
        "qid": qid,
        "xeno_canto_species_id": xeno_canto_species_id,
        "status": "ok" if rows else "no-qualifying-recordings",
        "pages_scanned": pages_scanned,
        "selected_count": len(rows),
    }


def fetch_xeno_canto_audio(
    ontology_path: Path,
    output_dir: Path,
    since_date: date = DEFAULT_SINCE_DATE,
    limit_per_qid: int = DEFAULT_LIMIT_PER_QID,
    max_pages: int = DEFAULT_MAX_PAGES,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    fetch_text_fn: Callable[[str], str] = fetch_text,
    download_bytes_fn: Callable[[str], bytes] = fetch_bytes,
) -> dict[str, Any]:
    """Download raw Xeno-canto audio files into per-QID directories. / QID ごとのディレクトリへ Xeno-canto 音声を保存する。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = ensure_query_rows(ontology_path)
    manifest_rows: list[dict[str, str]] = []
    audio_ids: list[str] = []
    qids: list[str] = []
    status_rows: list[dict[str, Any]] = []
    failed_qids: list[str] = []
    insufficient_qids: list[dict[str, Any]] = []

    for target in targets:
        qid = target["qid"]
        xeno_canto_species_id = target["xeno_canto_species_id"]
        rows, status = fetch_qid_audio(
            qid=qid,
            xeno_canto_species_id=xeno_canto_species_id,
            output_dir=output_dir,
            since_date=since_date,
            limit_per_qid=limit_per_qid,
            max_pages=max_pages,
            sleep_seconds=sleep_seconds,
            fetch_text_fn=fetch_text_fn,
            download_bytes_fn=download_bytes_fn,
        )
        status_rows.append(status)
        if status["status"] != "ok":
            failed_qids.append(qid)
        if len(rows) < limit_per_qid:
            insufficient_qids.append(
                {
                    "qid": qid,
                    "xeno_canto_species_id": xeno_canto_species_id,
                    "selected_count": len(rows),
                }
            )
        for row in rows:
            manifest_rows.append(row)
            audio_ids.append(row["audio_id"])
            qids.append(row["qid"])

    manifest_path = output_dir / "audio_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_json_atomic(output_dir / "audio_ids.json", audio_ids)
    write_json_atomic(output_dir / "qids.json", qids)

    metadata = {
        "dataset": "xeno-canto",
        "since_date": since_date.isoformat(),
        "limit_per_qid": limit_per_qid,
        "max_pages": max_pages,
        "sleep_seconds": sleep_seconds,
        "recording_page_order": "rec",
        "manifest_path": str(manifest_path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(output_dir / "metadata.json", metadata)

    summary = {
        "dataset": "xeno-canto",
        "target_qid_count": len(targets),
        "completed_qid_count": len({row["qid"] for row in manifest_rows}),
        "recording_count": len(manifest_rows),
        "failed_qid_count": len(failed_qids),
        "insufficient_qid_count": len(insufficient_qids),
        "failed_qids": failed_qids[:50],
        "insufficient_qids": insufficient_qids[:50],
    }
    write_json_atomic(output_dir / "summary.json", summary)

    return {
        "metadata": metadata,
        "summary": summary,
        "status_rows": status_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser. / CLI パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Download Xeno-canto audio files for bird QIDs.")
    parser.add_argument("--input", default=str(paths.ontology_pkl))
    parser.add_argument("--output-dir", default=str(paths.xeno_canto_after_202505_dir))
    parser.add_argument("--since-date", default=DEFAULT_SINCE_DATE.isoformat())
    parser.add_argument("--limit-per-qid", type=int, default=DEFAULT_LIMIT_PER_QID)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the download command. / ダウンロードコマンドを実行する。"""

    args = build_parser().parse_args(argv)
    fetch_xeno_canto_audio(
        ontology_path=Path(args.input),
        output_dir=Path(args.output_dir),
        since_date=parse_date(args.since_date),
        limit_per_qid=args.limit_per_qid,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
