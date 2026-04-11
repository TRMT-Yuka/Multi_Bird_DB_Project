from __future__ import annotations

import argparse
from collections.abc import Callable

from .env_loader import load_project_env
from . import dump_extract, fetch_script, ontology, qids, wikipedia_articles


def add_arguments(parser: argparse.ArgumentParser, defaults: argparse.Namespace, names: list[str]) -> None:
    """Copy a small set of default arguments into one subcommand parser. / 既定引数の一部をサブコマンド用パーサへ写す。"""

    for name in names:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=getattr(defaults, name))


def namespace_to_args(args: argparse.Namespace, names: list[str]) -> list[str]:
    """Convert parsed values back into a flat argument list. / 解析済み引数をフラットな CLI 引数列へ戻す。"""

    flat_args: list[str] = []
    for name in names:
        flat_args.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    return flat_args


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser for the whole project. / プロジェクト全体のトップレベル CLI パーサを作る。"""

    parser = argparse.ArgumentParser(description="Multi Bird DB utility CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_arguments(
        subparsers.add_parser("extract-qids", help="Extract QIDs from query.tsv."),
        qids.build_parser().parse_args([]),
        ["input", "output", "root_qid"],
    )
    add_arguments(
        subparsers.add_parser("generate-fetch", help="Generate a shell script to fetch Wikidata JSON."),
        fetch_script.build_parser().parse_args([]),
        ["input", "output", "json_dir"],
    )
    add_arguments(
        subparsers.add_parser(
            "extract-dump-json",
            help="Extract requested Wikidata entity JSON files from a downloaded dump.",
        ),
        dump_extract.build_parser().parse_args([]),
        ["input", "dump", "output_dir"],
    )
    add_arguments(
        subparsers.add_parser("build-ontology", help="Build ontology TSV from downloaded JSON."),
        ontology.build_parser().parse_args([]),
        ["json_dir", "output", "root_qid"],
    )
    add_arguments(
        subparsers.add_parser(
            "build-wikipedia-manifest",
            help="Build a TSV manifest for related English and Japanese Wikipedia articles.",
        ),
        wikipedia_articles.build_parser().parse_args(["build-wikipedia-manifest"]),
        ["json_dir", "output", "xml_en_dir", "xml_ja_dir", "text_en_dir", "text_ja_dir"],
    )
    add_arguments(
        subparsers.add_parser(
            "fetch-wikipedia-xml",
            help="Fetch English and Japanese Wikipedia article XML files.",
        ),
        wikipedia_articles.build_parser().parse_args(["fetch-wikipedia-xml"]),
        ["manifest"],
    )
    add_arguments(
        subparsers.add_parser(
            "extract-wikipedia-text",
            help="Extract plain text sentences from saved Wikipedia XML files.",
        ),
        wikipedia_articles.build_parser().parse_args(["extract-wikipedia-text"]),
        ["manifest"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one CLI command to the matching module. / CLI コマンドを対応するモジュールへ振り分ける。"""

    load_project_env()
    args = build_parser().parse_args(argv)
    handlers: dict[str, tuple[Callable[[list[str] | None], int], list[str], list[str]]] = {
        "extract-qids": (qids.main, [], ["input", "output", "root_qid"]),
        "generate-fetch": (fetch_script.main, [], ["input", "output", "json_dir"]),
        "extract-dump-json": (dump_extract.main, [], ["input", "dump", "output_dir"]),
        "build-ontology": (ontology.main, [], ["json_dir", "output", "root_qid"]),
        "build-wikipedia-manifest": (
            wikipedia_articles.main,
            ["build-wikipedia-manifest"],
            ["json_dir", "output", "xml_en_dir", "xml_ja_dir", "text_en_dir", "text_ja_dir"],
        ),
        "fetch-wikipedia-xml": (wikipedia_articles.main, ["fetch-wikipedia-xml"], ["manifest"]),
        "extract-wikipedia-text": (wikipedia_articles.main, ["extract-wikipedia-text"], ["manifest"]),
    }
    handler, prefix, names = handlers[args.command]
    return handler(prefix + namespace_to_args(args, names))


if __name__ == "__main__":
    raise SystemExit(main())
