from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from .config import get_project_paths

USER_AGENT = "Multi_Bird_DB_Project/0.1 (research and educational use; contact: local-project)"
DEFAULT_SINCE_DATE = date(2025, 5, 1)
DEFAULT_LIMIT_PER_QID = 20
DEFAULT_MAX_PAGES = 1
DEFAULT_SLEEP_SECONDS = 0.25
DEFAULT_CLIP_SECONDS = 15
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_API_PER_PAGE = 20
DEFAULT_API_KEY = "demo"
DEFAULT_QUALITY = "A"
RECORDING_ID_RE = re.compile(r"\bXC(\d{3,7})\b")
UPLOADED_RE = re.compile(r"Uploaded\s*\|\s*([0-9]{4}-[0-9]{2}-[0-9]{2})")
FILE_TYPE_RE = re.compile(r"File type\s*\|\s*([A-Za-z0-9]+)")
SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _render_progress_line(message: str) -> None:
    """Render one in-place progress line to stderr. / stderr に進捗を 1 行表示する。"""

    sys.stderr.write(f"\r{message[:200]:<200}")
    sys.stderr.flush()


def _finish_progress_line(message: str | None = None) -> None:
    """Finish an in-place progress line. / 進捗行を確定する。"""

    if message:
        sys.stderr.write(f"\r{message[:200]:<200}\n")
    else:
        sys.stderr.write("\n")
    sys.stderr.flush()


def load_targets(input_path: Path) -> list[dict[str, str]]:
    """Load qid-to-Xeno-canto pairs from TSV or ontology PKL. / TSV か ontology PKL から対応表を読む。"""

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    leaf_qids = load_leaf_qids()
    suffix = input_path.suffix.lower()
    if suffix == ".tsv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows: list[dict[str, str]] = []
            for row in reader:
                qid = str(row.get("qid") or "").strip()
                xeno_canto_species_id = str(row.get("xeno_canto_species_id") or "").strip()
                if not qid or not xeno_canto_species_id or qid not in leaf_qids:
                    continue
                rows.append({"qid": qid, "xeno_canto_species_id": xeno_canto_species_id})
            return rows
    if suffix in {".pkl", ".pickle"}:
        with input_path.open("rb") as handle:
            data = pickle.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Ontology PKL must contain a list, got: {type(data).__name__}")
        rows: list[dict[str, str]] = []
        for row in data:
            qid = str(row.get("qid") or row.get("id") or "").strip()
            xeno_canto_species_id = str(row.get("xeno_canto_species_id") or "").strip()
            if not qid or not xeno_canto_species_id or qid not in leaf_qids:
                continue
            rows.append({"qid": qid, "xeno_canto_species_id": xeno_canto_species_id})
        return rows
    raise ValueError(f"Unsupported input format: {input_path}")


def fetch_text(url: str, timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS) -> str:
    """Fetch one UTF-8 text response. / 1 件の UTF-8 テキスト応答を読む。"""

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_bytes(url: str, timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS) -> bytes:
    """Fetch one binary response. / 1 件のバイナリ応答を読む。"""

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Fetch one JSON response. / 1 件の JSON 応答を読む。"""

    return json.loads(fetch_text(url, timeout=timeout))


def load_xeno_canto_api_key(api_key: str | None = None) -> str:
    """Resolve the API key from CLI arg, env var, or local env file. / CLI 引数・環境変数・ローカル env ファイルから API key を決める。"""

    if api_key and str(api_key).strip():
        return str(api_key).strip()
    env_key = os.environ.get("XENO_CANTO_API_KEY", "").strip()
    if env_key:
        return env_key
    paths = get_project_paths()
    env_file = paths.root / "xeno_canto_api_key.env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "XENO_CANTO_API_KEY":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    return DEFAULT_API_KEY


def load_leaf_qids(graph_path: Path | None = None) -> set[str]:
    """Load leaf QIDs from the taxonomy graph. / taxonomy graph から末端 QID を読む。"""

    paths = get_project_paths()
    path = graph_path or paths.taxonomy_graph_pkl
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy graph does not exist: {path}")
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not hasattr(graph, "out_degree"):
        raise ValueError(f"Taxonomy graph must be a graph object, got: {type(graph).__name__}")
    return {str(node) for node in graph.nodes if graph.out_degree(node) == 0}


def api_query_for_species_id(species_id: str, quality: str | None = DEFAULT_QUALITY) -> str:
    """Build a search query for the Xeno-canto API. / Xeno-canto API 用の検索語を作る。"""

    normalized = species_id.replace("-", " ").strip()
    parts = [f'sp:"{normalized}"']
    quality_value = str(quality or "").strip().upper()
    if quality_value:
        parts.append(f"q:{quality_value}")
    return " ".join(parts)


def api_recordings_url(
    species_id: str,
    api_key: str,
    page: int,
    per_page: int = DEFAULT_API_PER_PAGE,
    quality: str | None = DEFAULT_QUALITY,
) -> str:
    """Build a Xeno-canto API recordings URL. / Xeno-canto API の recordings URL を作る。"""

    params = {
        "query": api_query_for_species_id(species_id, quality=quality),
        "page": str(page),
        "per_page": str(per_page),
        "key": api_key,
    }
    return "https://xeno-canto.org/api/3/recordings?" + urlencode(params, quote_via=quote)


def recording_page_url(recording_id: str) -> str:
    """Build a Xeno-canto recording URL. / Xeno-canto 録音ページ URL を作る。"""

    return f"https://xeno-canto.org/{recording_id}"


def recording_download_url(recording_id: str) -> str:
    """Build a Xeno-canto download URL. / Xeno-canto 音声ダウンロード URL を作る。"""

    return f"https://xeno-canto.org/{recording_id}/download"


def safe_path_component(value: str) -> str:
    """Convert one identifier into a filesystem-safe path component. / 識別子をファイル名安全な成分にする。"""

    normalized = SAFE_PATH_COMPONENT_RE.sub("_", value.strip())
    normalized = normalized.strip("._")
    return normalized or "item"


def write_tsv_atomic(output_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write one TSV atomically. / TSV を原子的に書く。"""

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
            newline="",
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def fetch_xeno_canto_recording_jsons(
    input_path: Path,
    output_dir: Path,
    api_key: str | None = None,
    per_page: int = DEFAULT_API_PER_PAGE,
    max_pages: int = DEFAULT_MAX_PAGES,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    quality: str | None = DEFAULT_QUALITY,
    fetch_json_fn: Callable[[str], dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    """Fetch and save Xeno-canto API JSON responses. / Xeno-canto API の JSON 応答を保存する。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_xeno_canto_api_key(api_key)
    targets = load_targets(input_path)
    manifest_rows: list[dict[str, Any]] = []
    failed_qids: list[str] = []
    total_targets = len(targets)

    for target_index, target in enumerate(targets, start=1):
        qid = target["qid"]
        xeno_canto_species_id = target["xeno_canto_species_id"]
        qid_dir = output_dir / qid
        qid_dir.mkdir(parents=True, exist_ok=True)
        api_urls: list[str] = []
        api_paths: list[str] = []
        last_num_pages = 0
        status = "ok"
        error = ""
        pages_scanned = 0
        _render_progress_line(
            f"api json {target_index}/{total_targets} | {qid} | page 0/{max_pages} | recordings pending"
        )
        for page in range(1, max_pages + 1):
            api_url = api_recordings_url(
                xeno_canto_species_id,
                api_key=api_key,
                page=page,
                per_page=per_page,
                quality=quality,
            )
            try:
                payload = fetch_json_fn(api_url)
            except (HTTPError, URLError) as exc:
                status = "api-error"
                error = str(exc)
                failed_qids.append(qid)
                break
            api_path = qid_dir / f"page{page:03}.json"
            write_json_atomic(api_path, payload)
            api_urls.append(api_url)
            api_paths.append(str(api_path))
            pages_scanned = page
            recording_count = len(payload.get("recordings") or [])
            _render_progress_line(
                f"api json {target_index}/{total_targets} | {qid} | page {page}/{max_pages} | recordings {recording_count}"
            )
            try:
                last_num_pages = int(payload.get("numPages") or 0)
            except (TypeError, ValueError):
                last_num_pages = 0
            recordings = payload.get("recordings") or []
            if not recordings:
                break
            if last_num_pages and page >= last_num_pages:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        if status == "ok" and not api_paths:
            status = "no-pages"
        _finish_progress_line(f"api json {target_index}/{total_targets} | {qid} | done ({status})")
        manifest_rows.append(
            {
                "qid": qid,
                "xeno_canto_species_id": xeno_canto_species_id,
                "status": status,
                "error": error,
                "pages_scanned": pages_scanned,
                "api_urls": api_urls,
                "api_paths": api_paths,
            }
        )

    manifest_path = output_dir / "api_recordings_manifest.json"
    summary_path = output_dir / "api_recordings_summary.json"
    write_json_atomic(manifest_path, manifest_rows)
    summary = {
        "dataset": "xeno-canto",
        "target_qid_count": len(targets),
        "saved_qid_count": sum(1 for row in manifest_rows if row["status"] == "ok"),
        "failed_qid_count": len(failed_qids),
        "failed_qids": failed_qids[:50],
        "api_key_provided": api_key != DEFAULT_API_KEY,
        "per_page": per_page,
        "max_pages": max_pages,
        "quality": quality,
    }
    write_json_atomic(summary_path, summary)
    return {"manifest_path": str(manifest_path), "summary": summary, "manifest_rows": manifest_rows}


def scan_api_recording_pages(input_dir: Path) -> list[dict[str, Any]]:
    """Collect saved API page JSON paths from a directory tree. / 保存済み API JSON のディレクトリを走査する。"""

    if not input_dir.exists():
        raise FileNotFoundError(f"API recordings directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"API recordings input must be a directory, got: {input_dir}")

    rows: list[dict[str, Any]] = []
    for qid_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        qid = qid_dir.name.strip()
        if not qid:
            continue
        api_paths = sorted(str(path) for path in qid_dir.glob("page*.json"))
        rows.append(
            {
                "qid": qid,
                "api_paths": api_paths,
            }
        )
    return rows


def build_xeno_canto_recording_map_from_api(
    input_path: Path,
    output_json: Path,
) -> dict[str, Any]:
    """Extract recording IDs and download URLs from saved API JSON pages. / API JSON から recording ID と download URL を抜き出す。"""

    if input_path.is_dir():
        manifest = scan_api_recording_pages(input_path)
        targets_by_qid = {row["qid"]: row for row in load_targets(get_project_paths().xeno_canto_ids_tsv)}
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"API input does not exist: {input_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, list):
            raise ValueError(f"API manifest must be a list, got: {type(manifest).__name__}")
        targets_by_qid = {}

    recording_map_rows: list[dict[str, Any]] = []
    for item in manifest:
        qid = str(item.get("qid") or "").strip()
        xeno_canto_species_id = str(item.get("xeno_canto_species_id") or "").strip()
        api_paths = [str(path) for path in item.get("api_paths") or []]
        if not xeno_canto_species_id:
            target_row = targets_by_qid.get(qid, {})
            xeno_canto_species_id = str(target_row.get("xeno_canto_species_id") or "").strip()
        recording_ids: list[str] = []
        download_urls: list[str] = []
        seen: set[str] = set()
        for api_path in api_paths:
            page_path = Path(api_path)
            if not page_path.exists():
                continue
            payload = json.loads(page_path.read_text(encoding="utf-8"))
            for recording in payload.get("recordings") or []:
                recording_id = str(recording.get("id") or "").strip()
                if not recording_id or recording_id in seen:
                    continue
                seen.add(recording_id)
                recording_ids.append(recording_id)
                download_url = str(recording.get("file") or "").strip()
                if not download_url:
                    download_url = recording_download_url(recording_id)
                download_urls.append(download_url)
        recording_map_rows.append(
            {
                "qid": qid,
                "xeno_canto_species_id": xeno_canto_species_id,
                "recording_count": len(recording_ids),
                "recording_ids": recording_ids,
                "download_urls": download_urls,
                "api_paths": api_paths,
            }
        )

    write_json_atomic(output_json, recording_map_rows)
    summary = {
        "dataset": "xeno-canto",
        "species_count": len(recording_map_rows),
        "recording_count": sum(int(row["recording_count"]) for row in recording_map_rows),
        "output_json": str(output_json),
    }
    return {"summary": summary, "recording_map_rows": recording_map_rows}


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


def make_audio_temp_dir(output_dir: Path, qid: str, recording_id: str) -> Path:
    """Create one project-local temporary directory for audio work. / 音声処理用のプロジェクト内一時ディレクトリを作る。"""

    temp_root = output_dir.parent.parent / "temp" / "xeno-canto"
    temp_dir = temp_root / qid / recording_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def scan_existing_audio_files(output_dir: Path) -> list[dict[str, Any]]:
    """Collect already existing audio files under the output root. / 出力先に既にある音声ファイルを集める。"""

    existing_rows: list[dict[str, Any]] = []
    if not output_dir.exists():
        return existing_rows
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".tmp"):
            continue
        if path.name in {"existing_audio_manifest.json"}:
            continue
        try:
            rel_path = path.relative_to(output_dir)
        except ValueError:
            continue
        if len(rel_path.parts) < 2:
            continue
        qid = rel_path.parts[0]
        recording_id = path.stem
        existing_rows.append(
            {
                "qid": qid,
                "recording_id": recording_id,
                "path": str(rel_path),
                "size": path.stat().st_size,
            }
        )
    return existing_rows


def parse_date(value: str) -> date:
    """Parse an ISO formatted date. / ISO 形式の日付を読む。"""

    return date.fromisoformat(value)


def _ffmpeg_codec_args(file_type: str) -> list[str]:
    normalized = file_type.strip().lower()
    if normalized == "mp3":
        return ["-c:a", "libmp3lame", "-q:a", "2"]
    if normalized == "wav":
        return ["-c:a", "pcm_s16le"]
    return ["-c:a", "copy"]


def probe_audio_duration_seconds(input_path: Path) -> float | None:
    """Return audio duration in seconds if ffprobe is available. / ffprobe があれば音声長を秒で返す。"""

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    value = completed.stdout.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def clip_audio_file(input_path: Path, output_path: Path, file_type: str, clip_seconds: int) -> None:
    """Trim the first N seconds using ffmpeg. / ffmpeg で先頭 N 秒を切り出す。"""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to clip Xeno-canto audio.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-t",
        str(clip_seconds),
        "-vn",
        *(_ffmpeg_codec_args(file_type)),
        str(output_path),
    ]
    subprocess.run(command, check=True)


def copy_audio_file(input_path: Path, output_path: Path) -> None:
    """Copy one audio file atomically. / 音声をそのままコピーする。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def download_recording(
    recording_id: str,
    local_path: Path,
    file_type: str,
    clip_seconds: int,
    download_bytes_fn: Callable[[str], bytes],
    clip_audio_fn: Callable[[Path, Path, str, int], None] = clip_audio_file,
) -> None:
    """Download and clip one recording if the target file does not exist yet. / 1 件の録音を取得して切り出す。"""

    if local_path.exists() and local_path.stat().st_size > 0:
        return
    payload = download_bytes_fn(recording_download_url(recording_id))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / f"{recording_id}.{file_type or 'mp3'}"
        write_bytes_atomic(tmp_input, payload)
        duration = probe_audio_duration_seconds(tmp_input)
        if duration is not None and duration <= clip_seconds:
            copy_audio_file(tmp_input, local_path)
            return
        clip_audio_fn(tmp_input, local_path, file_type=file_type or "mp3", clip_seconds=clip_seconds)


def load_recording_map(input_path: Path) -> list[dict[str, Any]]:
    """Load a recording map from JSON or TSV. / recording map を JSON か TSV から読む。"""

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Recording map JSON must contain a list, got: {type(data).__name__}")
        rows: list[dict[str, Any]] = []
        for item in data:
            qid = str(item.get("qid") or "").strip()
            xeno_canto_species_id = str(item.get("xeno_canto_species_id") or "").strip()
            recording_ids = [str(recording_id).strip() for recording_id in (item.get("recording_ids") or []) if str(recording_id).strip()]
            download_urls = [str(download_url).strip() for download_url in (item.get("download_urls") or []) if str(download_url).strip()]
            if not qid or not xeno_canto_species_id or not recording_ids:
                continue
            rows.append(
                {
                    "qid": qid,
                    "xeno_canto_species_id": xeno_canto_species_id,
                    "recording_ids": recording_ids,
                    "download_urls": download_urls,
                    "species_page_paths": [str(path) for path in (item.get("species_page_paths") or [])],
                }
            )
        return rows
    if suffix == ".tsv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            grouped: dict[tuple[str, str], list[str]] = {}
            for row in reader:
                qid = str(row.get("qid") or "").strip()
                xeno_canto_species_id = str(row.get("xeno_canto_species_id") or "").strip()
                recording_id = str(row.get("recording_id") or "").strip()
                if not qid or not xeno_canto_species_id or not recording_id:
                    continue
                grouped.setdefault((qid, xeno_canto_species_id), []).append(recording_id)
        return [
            {
                "qid": qid,
                "xeno_canto_species_id": xeno_species_id,
                "recording_ids": recording_ids,
                "download_urls": [],
                "species_page_paths": [],
            }
            for (qid, xeno_species_id), recording_ids in grouped.items()
        ]
    raise ValueError(f"Unsupported input format: {input_path}")


def fetch_audio_from_recording_map(
    input_path: Path,
    output_dir: Path,
    limit_per_qid: int = DEFAULT_LIMIT_PER_QID,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    clip_seconds: int = DEFAULT_CLIP_SECONDS,
    max_workers: int = 3,
    fetch_text_fn: Callable[[str], str] = fetch_text,
    download_bytes_fn: Callable[[str], bytes] = fetch_bytes,
    clip_audio_fn: Callable[[Path, Path, str, int], None] = clip_audio_file,
) -> dict[str, Any]:
    """Download and clip audio using an extracted recording map. / 抽出済み recording map を使って音声を保存する。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_audio_manifest_path = output_dir / "existing_audio_manifest.json"
    existing_audio_rows = scan_existing_audio_files(output_dir)
    write_json_atomic(
        existing_audio_manifest_path,
        {
            "dataset": "xeno-canto",
            "output_dir": str(output_dir),
            "item_count": len(existing_audio_rows),
            "items": existing_audio_rows,
        },
    )
    existing_audio_paths = {row["path"] for row in existing_audio_rows}
    targets = load_recording_map(input_path)
    status_rows: list[dict[str, Any]] = []
    failed_qids: list[str] = []
    total_targets = len(targets)
    existing_audio_lock = Lock()

    for target_index, target in enumerate(targets, start=1):
        qid = target["qid"]
        xeno_canto_species_id = target["xeno_canto_species_id"]
        recording_ids = target["recording_ids"][:limit_per_qid]
        download_urls = target.get("download_urls") or []
        qid_dir = output_dir / qid
        qid_dir.mkdir(parents=True, exist_ok=True)
        downloaded_count = 0
        _render_progress_line(
            f"audio {target_index}/{total_targets} | {qid} | 0/{len(recording_ids)} downloaded"
        )
        task_specs: list[tuple[int, str, str, Path, str]] = []
        for ordinal, recording_id in enumerate(recording_ids):
            download_url = str(download_urls[ordinal]) if ordinal < len(download_urls) and str(download_urls[ordinal]).strip() else recording_download_url(recording_id)
            suffix = Path(urlparse(download_url).path).suffix.lstrip(".").lower()
            file_type = suffix if suffix in {"mp3", "wav"} else "mp3"
            local_path = qid_dir / f"{recording_id}.{file_type}"
            task_specs.append((ordinal, recording_id, download_url, local_path, file_type))

        def _download_one(spec: tuple[int, str, str, Path, str]) -> bool:
            _, recording_id, download_url, local_path, file_type = spec
            try:
                relative_path = str(local_path.relative_to(output_dir))
                with existing_audio_lock:
                    if relative_path in existing_audio_paths:
                        return True
                    if local_path.exists() and local_path.stat().st_size > 0:
                        existing_audio_paths.add(relative_path)
                        return True
                payload = download_bytes_fn(download_url)
                tmp_dir = make_audio_temp_dir(output_dir, qid, recording_id)
                tmp_input = tmp_dir / f"{recording_id}.{file_type or 'mp3'}"
                try:
                    write_bytes_atomic(tmp_input, payload)
                    duration = probe_audio_duration_seconds(tmp_input)
                    if duration is not None and duration <= clip_seconds:
                        copy_audio_file(tmp_input, local_path)
                    else:
                        clip_audio_fn(tmp_input, local_path, file_type=file_type or "mp3", clip_seconds=clip_seconds)
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                with existing_audio_lock:
                    existing_audio_paths.add(relative_path)
                return True
            except (HTTPError, URLError):
                return False

        if task_specs:
            with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
                future_map = {executor.submit(_download_one, spec): spec for spec in task_specs}
                for future in as_completed(future_map):
                    if future.result():
                        downloaded_count += 1
                        _render_progress_line(
                            f"audio {target_index}/{total_targets} | {qid} | {downloaded_count}/{len(recording_ids)} downloaded"
                        )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
        if downloaded_count == 0:
            failed_qids.append(qid)
        _finish_progress_line(
            f"audio {target_index}/{total_targets} | {qid} | done ({downloaded_count}/{len(recording_ids)} downloaded)"
        )
        status_rows.append(
            {
                "qid": qid,
                "xeno_canto_species_id": xeno_canto_species_id,
                "status": "ok" if downloaded_count else "no-recordings",
                "selected_count": downloaded_count,
            }
        )
    return {
        "target_qid_count": len(targets),
        "downloaded_qid_count": sum(1 for row in status_rows if row["status"] == "ok"),
        "failed_qid_count": len(failed_qids),
        "failed_qids": failed_qids[:50],
        "status_rows": status_rows,
        "existing_audio_manifest_path": str(existing_audio_manifest_path),
    }


def build_api_recordings_parser() -> argparse.ArgumentParser:
    """Create the API-recording fetch CLI parser. / API recording 取得用 CLI パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Fetch Xeno-canto API JSON responses for each species.")
    parser.add_argument("--input", default=str(paths.xeno_canto_ids_tsv))
    parser.add_argument("--output-dir", default=str(paths.xeno_canto_interim_dir / "api_recordings"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--per-page", type=int, default=DEFAULT_API_PER_PAGE)
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    return parser


def build_recording_map_parser() -> argparse.ArgumentParser:
    """Create the recording-map extraction CLI parser. / recording map 抽出用 CLI パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Extract Xeno-canto recording IDs from saved API JSON files.")
    parser.add_argument("--input", default=str(paths.xeno_canto_interim_dir / "api_recordings"))
    parser.add_argument("--output-json", default=str(paths.xeno_canto_recording_map_json))
    return parser


def build_audio_parser() -> argparse.ArgumentParser:
    """Create the audio download CLI parser. / 音声取得用 CLI パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Download Xeno-canto audio files from an extracted recording map.")
    parser.add_argument("--input", default=str(paths.xeno_canto_recording_map_json))
    parser.add_argument("--output-dir", default=str(paths.xeno_canto_raw_dir))
    parser.add_argument("--limit-per-qid", type=int, default=DEFAULT_LIMIT_PER_QID)
    parser.add_argument("--clip-seconds", type=int, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Backward-compatible alias for the audio parser. / 音声取得用パーサの互換エイリアス。"""

    return build_audio_parser()


def main(argv: list[str] | None = None) -> int:
    """Run the audio download command. / 音声取得コマンドを実行する。"""

    args = build_audio_parser().parse_args(argv)
    fetch_audio_from_recording_map(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        limit_per_qid=args.limit_per_qid,
        clip_seconds=args.clip_seconds,
        max_workers=args.max_workers,
        sleep_seconds=args.sleep_seconds,
    )
    return 0


def main_api_recordings(argv: list[str] | None = None) -> int:
    """Run the API-recording fetch command. / API recording 取得コマンドを実行する。"""

    args = build_api_recordings_parser().parse_args(argv)
    fetch_xeno_canto_recording_jsons(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        api_key=load_xeno_canto_api_key(args.api_key),
        per_page=args.per_page,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep_seconds,
        quality=args.quality,
    )
    return 0


def main_recording_map(argv: list[str] | None = None) -> int:
    """Run the recording-map extraction command. / recording map 抽出コマンドを実行する。"""

    args = build_recording_map_parser().parse_args(argv)
    build_xeno_canto_recording_map_from_api(
        input_path=Path(args.input),
        output_json=Path(args.output_json),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
