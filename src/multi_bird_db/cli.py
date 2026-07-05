from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class CommandSpec:
    """Describe one top-level CLI command. / トップレベル CLI コマンドの定義。"""

    help: str
    module: str | None
    handler_name: str
    argv_prefix: tuple[str, ...] = ()


COMMAND_SPECS: dict[str, CommandSpec] = {
    "extract-qids": CommandSpec(
        help="Extract QIDs from query.tsv.",
        module="multi_bird_db.qids",
        handler_name="main",
    ),
    "extract-dump-json": CommandSpec(
        help="Materialize requested Wikidata entity JSON files by scanning the dump directly.",
        module="multi_bird_db.dump_extract",
        handler_name="main",
    ),
    "build-ontology": CommandSpec(
        help="Build ontology PKL from downloaded JSON.",
        module="multi_bird_db.ontology",
        handler_name="main",
    ),
    "extract-xeno-canto-ids": CommandSpec(
        help="Extract qid-to-Xeno-canto species ID pairs from ontology PKL.",
        module="multi_bird_db.xeno_canto_ids",
        handler_name="main",
    ),
    "fetch-xeno-canto-recording-json": CommandSpec(
        help="Fetch and save Xeno-canto API JSON responses per species.",
        module="multi_bird_db.xeno_canto_audio",
        handler_name="main_api_recordings",
    ),
    "extract-xeno-canto-recording-ids": CommandSpec(
        help="Extract recording IDs from saved Xeno-canto API JSON files.",
        module="multi_bird_db.xeno_canto_audio",
        handler_name="main_recording_map",
    ),
    "fetch-xeno-canto-audio": CommandSpec(
        help="Download Xeno-canto audio files into per-QID raw directories.",
        module="multi_bird_db.xeno_canto_audio",
        handler_name="main",
    ),
    "build-audio-embeddings": CommandSpec(
        help="Build wav2vec2-based embeddings from a directory tree of audio files.",
        module="multi_bird_db.audio_embeddings",
        handler_name="main",
    ),
    "download-audio-models": CommandSpec(
        help="Download and cache audio embedding model assets.",
        module="multi_bird_db.audio_embeddings",
        handler_name="main_download",
    ),
    "finetune-wav2vec2-crossval": CommandSpec(
        help="Fine-tune wav2vec2 audio classifiers across cross-validation folds.",
        module="multi_bird_db.audio_finetuning",
        handler_name="main",
    ),
    "build-graph": CommandSpec(
        help="Build taxonomy graph PKL from ontology PKL.",
        module="multi_bird_db.graph",
        handler_name="main",
    ),
    "build-sqlite": CommandSpec(
        help="Build a lightweight SQLite DB from ontology PKL.",
        module="multi_bird_db.sqlite_store",
        handler_name="main",
    ),
    "build-embeddings": CommandSpec(
        help="Build graph embeddings from a taxonomy graph PKL.",
        module="multi_bird_db.embeddings",
        handler_name="main",
    ),
    "build-language-surface-manifest": CommandSpec(
        help="Build per-language surface_id-to-text manifests from bird ontology PKL.",
        module="multi_bird_db.language_embeddings",
        handler_name="main",
    ),
    "build-language-embeddings": CommandSpec(
        help="Build BERT-based language embeddings from bird ontology PKL.",
        module="multi_bird_db.language_embeddings",
        handler_name="main_embeddings",
    ),
    "check-gpu": CommandSpec(
        help="Print a small CUDA / torch environment report.",
        module=None,
        handler_name="print_cuda_report",
    ),
    "serve-graph": CommandSpec(
        help="Serve an interactive Dash Cytoscape viewer for the taxonomy graph.",
        module="multi_bird_db.graph_dash",
        handler_name="main",
    ),
    "evaluate-graph-embeddings": CommandSpec(
        help="Evaluate graph embeddings with clustering metrics and write a report.",
        module="multi_bird_db.graph_evaluation",
        handler_name="main",
    ),
    "inspect-multimodal-sources": CommandSpec(
        help="Inspect multimodal source embedding directories and validate row-aligned files.",
        module="multi_bird_db.multimodal.cli",
        handler_name="main",
    ),
    "run-multimodal-baseline": CommandSpec(
        help="Run the initial multimodal taxon classification baseline.",
        module="multi_bird_db.multimodal.cli",
        handler_name="main",
        argv_prefix=("run-baseline",),
    ),
    "build-wikipedia-manifest": CommandSpec(
        help="Build a TSV manifest for related English and Japanese Wikipedia articles.",
        module="multi_bird_db.wikipedia_articles",
        handler_name="main",
        argv_prefix=("build-wikipedia-manifest",),
    ),
    "fetch-wikipedia-xml": CommandSpec(
        help="Fetch English and Japanese Wikipedia article XML files.",
        module="multi_bird_db.wikipedia_articles",
        handler_name="main",
        argv_prefix=("fetch-wikipedia-xml",),
    ),
    "extract-wikipedia-text": CommandSpec(
        help="Extract plain text sentences from saved Wikipedia XML files.",
        module="multi_bird_db.wikipedia_articles",
        handler_name="main",
        argv_prefix=("extract-wikipedia-text",),
    ),
}


def print_cuda_report(argv: list[str] | None = None) -> int:
    """Print a small CUDA environment report. / CUDA 環境の簡易レポートを出す。"""

    if argv and any(flag in {"-h", "--help"} for flag in argv):
        parser = argparse.ArgumentParser(description="Print a small CUDA / torch environment report.")
        parser.parse_args(argv)

    language_embeddings = import_module("multi_bird_db.language_embeddings")
    print(json.dumps(language_embeddings.probe_cuda(), ensure_ascii=False, indent=2))
    return 0


def _load_handler(spec: CommandSpec) -> Callable[[list[str] | None], int]:
    """Resolve a command handler only when requested. / 必要になった時点でハンドラを解決する。"""

    if spec.module is None:
        return globals()[spec.handler_name]
    module = import_module(spec.module)
    handler: Any = getattr(module, spec.handler_name)
    return handler


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser for the whole project. / プロジェクト全体のトップレベル CLI パーサを作る。"""

    parser = argparse.ArgumentParser(description="Multi Bird DB utility CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, spec in COMMAND_SPECS.items():
        subparsers.add_parser(command, help=spec.help, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the selected subcommand with lazy imports. / 遅延 import で対象サブコマンドへ委譲する。"""

    args, remaining = build_parser().parse_known_args(argv)
    spec = COMMAND_SPECS[args.command]
    handler = _load_handler(spec)
    forwarded_args = list(spec.argv_prefix) + list(remaining)
    return handler(forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
