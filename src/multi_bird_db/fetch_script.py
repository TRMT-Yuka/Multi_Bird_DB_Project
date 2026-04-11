from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import get_project_paths


def load_qids(input_path: Path) -> list[str]:
    """Read a one-column TSV file of QIDs. / 1 列の QID TSV を読む。"""

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        return [
            qid
            for row in csv.DictReader(handle, delimiter="\t")
            if (qid := (row.get("qid") or "").strip())
        ]


def build_fetch_line(qid: str, json_dir: Path) -> str:
    """Return one curl command that downloads a Wikidata entity JSON file. / 1 件分の Wikidata JSON 取得コマンドを返す。"""

    url = f"https://www.wikidata.org/w/rest.php/wikibase/v1/entities/items/{qid}"
    return (
        "curl --fail --silent --show-error --location --retry 3 --retry-delay 2 "
        "--header \"Accept: application/json\" "
        "--header \"Content-Type: application/json\" "
        "--header \"Authorization: Bearer ${WIKIDATA_ACCESS_TOKEN}\" "
        "--header \"User-Agent: ${WIKIDATA_USER_AGENT}\" "
        f"\"{url}\" -o \"{json_dir.as_posix()}/{qid}.json\""
    )


def generate_fetch_script(input_path: Path, output_path: Path, json_dir: Path) -> None:
    """Create a shell script that downloads all requested entity JSON files. / 必要な Wikidata JSON をまとめて取得するシェルスクリプトを作る。"""

    paths = get_project_paths()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'PROJECT_ROOT="{paths.root.as_posix()}"',
        'if [ -f "$PROJECT_ROOT/.env" ]; then',
        "  # Load repo-local credentials for direct script execution.",
        '  set -a; . "$PROJECT_ROOT/.env"; set +a',
        "fi",
        'if [ -z "${WIKIDATA_ACCESS_TOKEN:-}" ]; then',
        '  echo "WIKIDATA_ACCESS_TOKEN is not set." >&2',
        "  exit 1",
        "fi",
        'if [ -z "${WIKIDATA_USER_AGENT:-}" ]; then',
        '  echo "WIKIDATA_USER_AGENT is not set." >&2',
        "  exit 1",
        "fi",
        f'mkdir -p "{json_dir.as_posix()}"',
        *[build_fetch_line(qid, json_dir) for qid in load_qids(input_path)],
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(0o755)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for fetch script generation. / 取得スクリプト生成コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(
        description="Generate a shell script that downloads Wikidata entity JSON."
    )
    parser.add_argument("--input", default=str(paths.qids_tsv))
    parser.add_argument("--output", default=str(paths.fetch_script))
    parser.add_argument("--json-dir", default=str(paths.json_dir))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fetch script generation command. / 取得スクリプト生成コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    generate_fetch_script(Path(args.input), Path(args.output), Path(args.json_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
