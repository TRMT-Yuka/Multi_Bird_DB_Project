from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import get_project_paths
from .ontology import load_entity

MEDIAWIKI_EXPORT_NS = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
MANIFEST_COLUMNS = ["qid", "language", "title", "xml_url", "xml_output_path", "text_output_path"]
USER_AGENT = (
    "Multi_Bird_DB_Project/0.1 "
    "(research and educational use; contact: local-project)"
)


def get_sitelink_title(entity: dict[str, Any], site_key: str) -> str:
    """Return the linked Wikipedia title for one site. / あるサイトに対応する Wikipedia 記事タイトルを返す。"""

    return entity.get("sitelinks", {}).get(site_key, {}).get("title", "")


def build_export_url(language_code: str, title: str) -> str:
    """Build a Special:Export URL for one Wikipedia title. / Wikipedia 記事タイトルから Special:Export の URL を作る。"""

    encoded_title = quote(title.replace(" ", "_"), safe="")
    return (
        f"https://{language_code}.wikipedia.org/w/index.php"
        f"?title=Special:Export&pages={encoded_title}&history=0&action=submit"
    )


def validate_json_dir(json_dir: Path) -> list[Path]:
    """Return JSON input files, or fail when the directory is unusable. / 入力 JSON 一覧を返し、使えない場合は失敗する。"""

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")
    json_paths = sorted(json_dir.rglob("Q*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found under: {json_dir}")
    return json_paths


def ensure_directories(*directories: Path) -> None:
    """Create all directories that will receive generated files. / 出力先ディレクトリをまとめて作る。"""

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def build_manifest_row(
    qid: str,
    title: str,
    language_code: str,
    xml_output_dir: Path,
    text_output_dir: Path,
) -> dict[str, str]:
    """Describe one Wikipedia article and where its outputs should go. / 1 件の Wikipedia 記事について XML とテキストの保存先を表す。"""

    return {
        "qid": qid,
        "language": language_code,
        "title": title,
        "xml_url": build_export_url(language_code, title),
        "xml_output_path": str(xml_output_dir / f"{qid}.xml"),
        "text_output_path": str(text_output_dir / f"{qid}.txt"),
    }


def build_wikipedia_manifest(
    json_dir: Path,
    output_path: Path,
    xml_en_dir: Path,
    xml_ja_dir: Path,
    text_en_dir: Path,
    text_ja_dir: Path,
) -> None:
    """Create a manifest that lists XML and text destinations for Wikipedia pages. / Wikipedia 記事の XML とテキスト保存先をまとめた manifest を作る。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_directories(xml_en_dir, xml_ja_dir, text_en_dir, text_ja_dir)

    rows: list[dict[str, str]] = []
    for json_path in validate_json_dir(json_dir):
        entity = load_entity(json_path)
        if not (qid := (entity.get("qid") or entity.get("id") or "")):
            continue
        if en_title := get_sitelink_title(entity, "enwiki"):
            rows.append(build_manifest_row(qid, en_title, "en", xml_en_dir, text_en_dir))
        if ja_title := get_sitelink_title(entity, "jawiki"):
            rows.append(build_manifest_row(qid, ja_title, "ja", xml_ja_dir, text_ja_dir))

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def iter_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Read the manifest TSV as a list of rows. / manifest TSV を行の一覧として読む。"""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def download_url(url: str) -> str:
    """Download one text response using a project-specific User-Agent. / プロジェクト用 User-Agent を付けて 1 件のテキスト応答を取得する。"""

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return response.read().decode("utf-8")


def fetch_wikipedia_xml(manifest_path: Path) -> None:
    """Download XML export files listed in the manifest. / manifest にある Wikipedia XML を取得する。"""

    for row in iter_manifest_rows(manifest_path):
        xml_url = (row.get("xml_url") or "").strip()
        xml_output_path = Path((row.get("xml_output_path") or "").strip())
        if not xml_url:
            continue
        xml_output_path.parent.mkdir(parents=True, exist_ok=True)
        xml_output_path.write_text(download_url(xml_url), encoding="utf-8")


def read_latest_wikitext(xml_path: Path) -> str:
    """Read the newest revision text from one Wikipedia export XML file. / 1 件の Wikipedia XML から最新 revision の wikitext を読む。"""

    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    revisions = root.findall(".//mw:page/mw:revision", MEDIAWIKI_EXPORT_NS)
    if not revisions:
        return ""
    text_node = revisions[-1].find("mw:text", MEDIAWIKI_EXPORT_NS)
    return "" if text_node is None or text_node.text is None else text_node.text


def remove_nested_markup(text: str, start_token: str, end_token: str) -> str:
    """Remove simple nested blocks such as templates or tables. / テンプレートや表のような単純な入れ子記法を取り除く。"""

    while start_token in text and end_token in text:
        start_index = text.find(start_token)
        end_index = text.find(end_token, start_index + len(start_token))
        if end_index == -1:
            break
        text = text[:start_index] + text[end_index + len(end_token) :]
    return text


def convert_wikitext_to_plain_text(wikitext: str) -> str:
    """Convert rough wikitext into a simpler plain-text form. / wikitext を簡単なプレーンテキストへ整形する。"""

    text = wikitext
    substitutions = [
        (r"<!--.*?-->", " ", re.DOTALL),
        (r"<ref[^>]*>.*?</ref>", " ", re.DOTALL),
        (r"<[^>]+>", " ", 0),
        (r"\[\[(?:File|Image|ファイル|画像):[^\]]+\]\]", " ", re.IGNORECASE),
        (r"\[\[(?:Category|カテゴリ):[^\]]+\]\]", " ", re.IGNORECASE),
        (r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", 0),
        (r"\[\[([^\]]+)\]\]", r"\1", 0),
        (r"\[https?://[^\s\]]+\s([^\]]+)\]", r"\1", 0),
        (r"https?://\S+", " ", 0),
        (r"^=+\s*(.*?)\s*=+$", r"\1", re.MULTILINE),
        (r"^\*.*$", " ", re.MULTILINE),
        (r"^#.*$", " ", re.MULTILINE),
        (r"^[;:].*$", " ", re.MULTILINE),
        (r"\n{2,}", "\n", 0),
        (r"[ \t]+", " ", 0),
    ]
    text = remove_nested_markup(text, "{{", "}}")
    text = remove_nested_markup(text, "{|", "|}")
    for pattern, replacement, flags in substitutions:
        text = re.sub(pattern, replacement, text, flags=flags)
    return text.replace("'''", "").replace("''", "").strip()


def split_into_sentences(text: str) -> list[str]:
    """Split plain text into sentence-like lines and drop short fragments. / プレーンテキストを文らしい単位へ分割し、短すぎる断片を除く。"""

    chunks = re.split(r"(?<=[。．！？.!?])\s+|\n+", text.replace("\r\n", "\n"))
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 20]


def extract_wikipedia_text(manifest_path: Path) -> None:
    """Read saved XML files and write extracted text files. / 保存済み XML を読み、抽出テキストを書き出す。"""

    for row in iter_manifest_rows(manifest_path):
        xml_output_path = Path((row.get("xml_output_path") or "").strip())
        text_output_path = Path((row.get("text_output_path") or "").strip())
        if not xml_output_path.exists():
            continue
        text_output_path.parent.mkdir(parents=True, exist_ok=True)
        plain_text = convert_wikitext_to_plain_text(read_latest_wikitext(xml_output_path))
        text_output_path.write_text("\n".join(split_into_sentences(plain_text)) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Wikipedia XML and text processing. / Wikipedia XML 取得とテキスト抽出用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(
        description="Build a Wikipedia XML manifest, fetch XML, and extract text."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "build-wikipedia-manifest",
        help="Build a TSV manifest for English and Japanese Wikipedia XML export files.",
    )
    manifest_parser.add_argument("--json-dir", default=str(paths.json_dir))
    manifest_parser.add_argument("--output", default=str(paths.wikipedia_manifest_tsv))
    manifest_parser.add_argument("--xml-en-dir", default=str(paths.wikipedia_xml_en_dir))
    manifest_parser.add_argument("--xml-ja-dir", default=str(paths.wikipedia_xml_ja_dir))
    manifest_parser.add_argument("--text-en-dir", default=str(paths.wikipedia_text_en_dir))
    manifest_parser.add_argument("--text-ja-dir", default=str(paths.wikipedia_text_ja_dir))

    fetch_parser = subparsers.add_parser(
        "fetch-wikipedia-xml",
        help="Fetch English and Japanese Wikipedia article XML files from a manifest TSV.",
    )
    fetch_parser.add_argument("--manifest", default=str(paths.wikipedia_manifest_tsv))

    extract_parser = subparsers.add_parser(
        "extract-wikipedia-text",
        help="Extract plain text sentences from saved Wikipedia XML files.",
    )
    extract_parser.add_argument("--manifest", default=str(paths.wikipedia_manifest_tsv))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one Wikipedia-related CLI subcommand. / Wikipedia 関連の CLI サブコマンドを 1 つ実行する。"""

    args = build_parser().parse_args(argv)
    if args.command == "build-wikipedia-manifest":
        build_wikipedia_manifest(
            json_dir=Path(args.json_dir),
            output_path=Path(args.output),
            xml_en_dir=Path(args.xml_en_dir),
            xml_ja_dir=Path(args.xml_ja_dir),
            text_en_dir=Path(args.text_en_dir),
            text_ja_dir=Path(args.text_ja_dir),
        )
        return 0
    if args.command == "fetch-wikipedia-xml":
        fetch_wikipedia_xml(Path(args.manifest))
        return 0
    if args.command == "extract-wikipedia-text":
        extract_wikipedia_text(Path(args.manifest))
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
