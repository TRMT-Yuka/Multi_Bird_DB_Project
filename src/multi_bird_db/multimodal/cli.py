from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loaders import (
    DEFAULT_GRAPH_MODEL,
    load_audio_embedding_run,
    load_graph_embedding_run,
    load_language_embedding_run,
    resolve_default_source_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Multimodal experiment utilities.')
    subparsers = parser.add_subparsers(dest='subcommand', required=False)

    inspect_parser = subparsers.add_parser('inspect', help='Inspect multimodal embedding source directories.')
    inspect_parser.add_argument('--graph-embedding-dir', type=Path, default=None)
    inspect_parser.add_argument(
        '--graph-embedding-method',
        choices=['all', 'node2vec', 'gcn', 'grace', 'graphsage', 'transe'],
        default='all',
        help='Graph method to inspect; all checks every available graph embedding method.',
    )
    inspect_parser.add_argument('--audio-embedding-dir', type=Path, default=None)
    inspect_parser.add_argument('--language-embedding-dir', type=Path, default=None)

    return parser


def _load_runs(args: argparse.Namespace):
    config = resolve_default_source_config(
        graph_embedding_dir=args.graph_embedding_dir,
        graph_embedding_method=args.graph_embedding_method,
        audio_embedding_dir=args.audio_embedding_dir,
        language_embedding_dir=args.language_embedding_dir,
    )
    graph_run = load_graph_embedding_run(config.graph_embedding_dir)
    audio_run = load_audio_embedding_run(config.audio_embedding_dir)
    language_run = load_language_embedding_run(config.language_embedding_dir)
    return config, graph_run, audio_run, language_run


def _run_inspect(args: argparse.Namespace) -> int:
    methods = ['node2vec', 'gcn', 'grace', 'graphsage', 'transe'] if args.graph_embedding_method == 'all' else [args.graph_embedding_method]
    graph_runs = {}
    audio_run = None
    language_run = None
    graph_errors = {}
    for method in methods:
        method_args_data = vars(args).copy()
        method_args_data['graph_embedding_method'] = method
        method_args = argparse.Namespace(**method_args_data)
        try:
            config, graph_run, current_audio_run, current_language_run = _load_runs(method_args)
        except FileNotFoundError as exc:
            if args.graph_embedding_method != 'all':
                raise
            graph_errors[method] = str(exc)
            continue
        graph_runs[method] = (config, graph_run)
        audio_run = current_audio_run
        language_run = current_language_run

    if not graph_runs:
        raise FileNotFoundError('; '.join(f'{method}: {error}' for method, error in graph_errors.items()))
    assert audio_run is not None
    assert language_run is not None
    graph_summaries = {
        method: {
            'graph_embedding_dir': str(config.graph_embedding_dir),
            'graph_qid_count': len(graph_run.qids),
            'graph_dim': int(graph_run.embeddings.shape[1]),
            'algorithm': graph_run.metadata.get('algorithm'),
            'feature_mode': graph_run.metadata.get('parameters', {}).get('feature_mode'),
        }
        for method, (config, graph_run) in graph_runs.items()
    }
    first_config = next(iter(graph_runs.values()))[0]
    payload = {
        'graph_embedding_method': args.graph_embedding_method,
        'graph_runs': graph_summaries,
        'audio_embedding_dir': str(first_config.audio_embedding_dir),
        'language_embedding_dir': str(first_config.language_embedding_dir),
        'audio_row_count': len(audio_run.rows),
        'language_row_count': len(language_run.rows),
        'audio_dim': int(audio_run.embeddings.shape[1]),
        'language_dim': int(language_run.embeddings.shape[1]),
    }
    for method, error in graph_errors.items():
        payload['graph_runs'][method] = {'error': error}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subcommand = args.subcommand or 'inspect'
    if subcommand == 'inspect':
        return _run_inspect(args)
    raise ValueError(f'Unsupported multimodal subcommand: {subcommand}')


if __name__ == '__main__':
    raise SystemExit(main())
