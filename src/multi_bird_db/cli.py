from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from . import (
    audio_embeddings,
    dump_extract,
    embeddings,
    graph,
    graph_dash,
    graph_evaluation,
    language_embeddings,
    ontology,
    qids,
    sqlite_store,
    wikipedia_articles,
    xeno_canto_audio,
    xeno_canto_ids,
)


def add_arguments(parser: argparse.ArgumentParser, defaults: argparse.Namespace, names: list[str]) -> None:
    """Copy a small set of default arguments into one subcommand parser. / 既定引数の一部をサブコマンド用パーサへ写す。"""

    for name in names:
        default = getattr(defaults, name)
        if isinstance(default, bool):
            parser.add_argument(f"--{name.replace('_', '-')}", dest=name, action="store_true", default=default)
        else:
            parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=default)


def namespace_to_args(args: argparse.Namespace, names: list[str]) -> list[str]:
    """Convert parsed values back into a flat argument list. / 解析済み引数をフラットな CLI 引数列へ戻す。"""

    flat_args: list[str] = []
    for name in names:
        value = getattr(args, name)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                flat_args.append(f"--{name.replace('_', '-')}")
            continue
        flat_args.extend([f"--{name.replace('_', '-')}", str(value)])
    return flat_args


def print_cuda_report(_: list[str] | None = None) -> int:
    """Print a small CUDA environment report. / CUDA 環境の簡易レポートを出す。"""

    print(json.dumps(language_embeddings.probe_cuda(), ensure_ascii=False, indent=2))
    return 0


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
        subparsers.add_parser(
            "extract-dump-json",
            help="Materialize requested Wikidata entity JSON files by scanning the dump directly.",
        ),
        dump_extract.build_parser().parse_args([]),
        ["input", "dump", "output_dir"],
    )
    add_arguments(
        subparsers.add_parser("build-ontology", help="Build ontology PKL from downloaded JSON."),
        ontology.build_parser().parse_args([]),
        ["json_dir", "output", "root_qid"],
    )
    add_arguments(
        subparsers.add_parser(
            "extract-xeno-canto-ids",
            help="Extract qid-to-Xeno-canto species ID pairs from ontology PKL.",
        ),
        xeno_canto_ids.build_parser().parse_args([]),
        ["input", "output"],
    )
    add_arguments(
        subparsers.add_parser(
            "fetch-xeno-canto-recording-json",
            help="Fetch and save Xeno-canto API JSON responses per species.",
        ),
        xeno_canto_audio.build_api_recordings_parser().parse_args([]),
        ["input", "output_dir", "api_key", "per_page", "max_pages", "sleep_seconds"],
    )
    add_arguments(
        subparsers.add_parser(
            "extract-xeno-canto-recording-ids",
            help="Extract recording IDs from saved Xeno-canto API JSON files.",
        ),
        xeno_canto_audio.build_recording_map_parser().parse_args([]),
        ["input", "output_json"],
    )
    add_arguments(
        subparsers.add_parser(
            "fetch-xeno-canto-audio",
            help="Download Xeno-canto audio files into per-QID raw directories.",
        ),
        xeno_canto_audio.build_audio_parser().parse_args([]),
        ["input", "output_dir", "limit_per_qid", "clip_seconds", "sleep_seconds"],
    )
    add_arguments(
        subparsers.add_parser(
            "build-audio-embeddings",
            help="Build wav2vec2-based embeddings from a directory tree of audio files.",
        ),
        audio_embeddings.build_parser().parse_args([]),
        [
            "backend",
            "input_dir",
            "output_dir",
            "model_name",
            "device",
            "batch_size",
            "max_seconds",
            "target_sample_rate",
            "extensions",
            "cache_dir",
        ],
    )
    add_arguments(
        subparsers.add_parser("build-graph", help="Build taxonomy graph PKL from ontology PKL."),
        graph.build_parser().parse_args([]),
        ["input", "output", "root_qid"],
    )
    add_arguments(
        subparsers.add_parser(
            "build-sqlite",
            help="Build a lightweight SQLite DB from ontology PKL.",
        ),
        sqlite_store.build_parser().parse_args([]),
        ["input", "output", "root_qid"],
    )
    add_arguments(
        subparsers.add_parser(
            "build-embeddings",
            help="Build graph embeddings from a taxonomy graph PKL.",
        ),
        embeddings.build_parser().parse_args([]),
        [
            "input",
            "output_dir",
            "algorithm",
            "dim",
            "proj_dim",
            "seed",
            "initial_features",
            "walk_length",
            "num_walks",
            "window_size",
            "negative_samples",
            "transe_negative_samples",
            "epochs",
            "learning_rate",
            "p",
            "q",
            "undirected",
            "layers",
            "residual",
            "tau",
            "drop_edge_rate_1",
            "drop_edge_rate_2",
            "drop_feature_rate_1",
            "drop_feature_rate_2",
            "batch_size",
            "grace_encoder",
            "device",
            "weight_decay",
            "p_norm",
            "graphsage_num_neighbors_1",
            "graphsage_num_neighbors_2",
            "root_qid",
        ],
    )
    add_arguments(
        subparsers.add_parser(
            "build-language-surface-manifest",
            help="Build per-language surface_id-to-text manifests from bird ontology PKL.",
        ),
        language_embeddings.build_parser().parse_args([]),
        ["input", "output_dir"],
    )
    add_arguments(
        subparsers.add_parser(
            "build-language-embeddings",
            help="Build BERT-based language embeddings from bird ontology PKL.",
        ),
        language_embeddings.build_embeddings_parser().parse_args([]),
        ["input", "output_dir", "english_model", "japanese_model", "batch_size", "max_length", "device"],
    )
    subparsers.add_parser(
        "check-gpu",
        help="Print a small CUDA / torch environment report.",
    )
    add_arguments(
        subparsers.add_parser(
            "serve-graph",
            help="Serve an interactive Dash Cytoscape viewer for the taxonomy graph.",
        ),
        graph_dash.build_parser().parse_args([]),
        ["input", "root_qid", "max_depth", "max_nodes", "host", "port", "debug"],
    )
    add_arguments(
        subparsers.add_parser(
            "evaluate-graph-embeddings",
            help="Evaluate graph embeddings with clustering metrics and write a report.",
        ),
        graph_evaluation.build_parser().parse_args([]),
        ["graph_input", "embeddings_root", "output_root", "label_field", "methods", "seed", "silhouette_sample_size"],
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

    args = build_parser().parse_args(argv)
    handlers: dict[str, tuple[Callable[[list[str] | None], int], list[str], list[str]]] = {
        "extract-qids": (qids.main, [], ["input", "output", "root_qid"]),
        "extract-dump-json": (dump_extract.main, [], ["input", "dump", "output_dir"]),
        "build-ontology": (ontology.main, [], ["json_dir", "output", "root_qid"]),
        "extract-xeno-canto-ids": (xeno_canto_ids.main, [], ["input", "output"]),
        "fetch-xeno-canto-recording-json": (
            xeno_canto_audio.main_api_recordings,
            [],
            ["input", "output_dir", "api_key", "per_page", "max_pages", "sleep_seconds"],
        ),
        "extract-xeno-canto-recording-ids": (
            xeno_canto_audio.main_recording_map,
            [],
            ["input", "output_json"],
        ),
        "fetch-xeno-canto-audio": (
            xeno_canto_audio.main,
            [],
            ["input", "output_dir", "limit_per_qid", "clip_seconds", "sleep_seconds"],
        ),
        "build-audio-embeddings": (
                audio_embeddings.main,
                [],
                [
                    "backend",
                    "input_dir",
                    "output_dir",
                    "model_name",
                "device",
                "batch_size",
                "max_seconds",
                "target_sample_rate",
                "extensions",
                "cache_dir",
            ],
        ),
        "build-graph": (graph.main, [], ["input", "output", "root_qid"]),
        "build-sqlite": (sqlite_store.main, [], ["input", "output", "root_qid"]),
        "build-embeddings": (
            embeddings.main,
            [],
            [
                "input",
                "output_dir",
                "algorithm",
                "dim",
                "proj_dim",
                "seed",
                "initial_features",
                "walk_length",
                "num_walks",
                "window_size",
                "negative_samples",
                "transe_negative_samples",
                "epochs",
                "learning_rate",
                "p",
                "q",
                "undirected",
                "layers",
                "residual",
                "tau",
                "drop_edge_rate_1",
                "drop_edge_rate_2",
                "drop_feature_rate_1",
                "drop_feature_rate_2",
                "batch_size",
                "grace_encoder",
                "device",
                "weight_decay",
                "p_norm",
                "graphsage_num_neighbors_1",
                "graphsage_num_neighbors_2",
                "root_qid",
            ],
        ),
        "build-language-surface-manifest": (
            language_embeddings.main,
            [],
            ["input", "output_dir"],
        ),
        "build-language-embeddings": (
            language_embeddings.main_embeddings,
            [],
            ["input", "output_dir", "english_model", "japanese_model", "batch_size", "max_length", "device"],
        ),
        "check-gpu": (print_cuda_report, [], []),
        "serve-graph": (
            graph_dash.main,
            [],
            ["input", "root_qid", "max_depth", "max_nodes", "host", "port", "debug"],
        ),
        "evaluate-graph-embeddings": (
            graph_evaluation.main,
            [],
            ["graph_input", "embeddings_root", "output_root", "label_field", "methods", "seed", "silhouette_sample_size"],
        ),
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
