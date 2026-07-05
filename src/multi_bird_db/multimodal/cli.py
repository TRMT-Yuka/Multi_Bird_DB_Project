from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_project_paths
from ..embeddings import load_graph
from .classifiers import SoftmaxClassifierConfig, fit_softmax_classifier, predict_labels
from .evaluate import (
    evaluate_predictions,
    labels_from_rows,
    select_rows_by_qids,
    write_json,
    write_metrics_tsv,
    write_split_predictions_tsv,
)
from .expanders import build_multimodal_feature_matrix
from .labels import assign_labels_for_qids
from .loaders import (
    load_audio_embedding_run,
    load_graph_embedding_run,
    load_language_embedding_run,
    resolve_default_source_config,
)
from .splits import split_qids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Multimodal experiment utilities.')
    subparsers = parser.add_subparsers(dest='subcommand', required=False)

    inspect_parser = subparsers.add_parser('inspect', help='Inspect multimodal embedding source directories.')
    inspect_parser.add_argument('--graph-embedding-dir', type=Path, default=None)
    inspect_parser.add_argument('--audio-embedding-dir', type=Path, default=None)
    inspect_parser.add_argument('--language-embedding-dir', type=Path, default=None)

    run_parser = subparsers.add_parser('run-baseline', help='Run the initial multimodal taxon classification baseline.')
    run_parser.add_argument('--graph-embedding-dir', type=Path, default=None)
    run_parser.add_argument('--audio-embedding-dir', type=Path, default=None)
    run_parser.add_argument('--language-embedding-dir', type=Path, default=None)
    run_parser.add_argument('--taxonomy-graph', type=Path, default=None)
    run_parser.add_argument('--target-rank', default='family')
    run_parser.add_argument('--modalities', default='graph,audio,language')
    run_parser.add_argument('--validation-fraction', type=float, default=0.1)
    run_parser.add_argument('--test-fraction', type=float, default=0.2)
    run_parser.add_argument('--seed', type=int, default=42)
    run_parser.add_argument('--learning-rate', type=float, default=0.05)
    run_parser.add_argument('--num-epochs', type=int, default=200)
    run_parser.add_argument('--l2-weight', type=float, default=1e-4)
    run_parser.add_argument('--output-dir', type=Path, default=None)
    return parser


def _parse_modalities(modalities: str) -> tuple[bool, bool, bool]:
    items = {item.strip().lower() for item in modalities.split(',') if item.strip()}
    unknown = items.difference({'graph', 'audio', 'language'})
    if unknown:
        unknown_list = ', '.join(sorted(unknown))
        raise ValueError(f'Unknown modalities: {unknown_list}')
    return ('graph' in items, 'audio' in items, 'language' in items)


def _load_runs(args: argparse.Namespace):
    config = resolve_default_source_config(
        graph_embedding_dir=args.graph_embedding_dir,
        audio_embedding_dir=args.audio_embedding_dir,
        language_embedding_dir=args.language_embedding_dir,
    )
    graph_run = load_graph_embedding_run(config.graph_embedding_dir)
    audio_run = load_audio_embedding_run(config.audio_embedding_dir)
    language_run = load_language_embedding_run(config.language_embedding_dir)
    return config, graph_run, audio_run, language_run


def _run_inspect(args: argparse.Namespace) -> int:
    config, graph_run, audio_run, language_run = _load_runs(args)
    payload = {
        'graph_embedding_dir': str(config.graph_embedding_dir),
        'audio_embedding_dir': str(config.audio_embedding_dir),
        'language_embedding_dir': str(config.language_embedding_dir),
        'graph_qid_count': len(graph_run.qids),
        'audio_row_count': len(audio_run.rows),
        'language_row_count': len(language_run.rows),
        'graph_dim': int(graph_run.embeddings.shape[1]),
        'audio_dim': int(audio_run.embeddings.shape[1]),
        'language_dim': int(language_run.embeddings.shape[1]),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return get_project_paths().root / 'data' / 'external' / 'experiments' / 'multimodal_taxon_classification' / timestamp


def _run_baseline(args: argparse.Namespace) -> int:
    include_graph, include_audio, include_language = _parse_modalities(args.modalities)
    config, graph_run, audio_run, language_run = _load_runs(args)
    project_paths = get_project_paths()
    taxonomy_graph_path = args.taxonomy_graph or project_paths.taxonomy_graph_pkl
    taxonomy_graph = load_graph(taxonomy_graph_path)

    candidate_qids = sorted(set(graph_run.qids) | set(audio_run.qids) | set(language_run.qids))
    assignments = assign_labels_for_qids(taxonomy_graph, candidate_qids, args.target_rank)
    feature_result = build_multimodal_feature_matrix(
        graph_run=graph_run,
        audio_run=audio_run,
        language_run=language_run,
        assignments=assignments,
        include_graph=include_graph,
        include_audio=include_audio,
        include_language=include_language,
    )

    labeled_qids = sorted({assignment.qid for assignment in assignments})
    split = split_qids(
        labeled_qids,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    train_matrix = select_rows_by_qids(feature_result.matrix, set(split.train_qids))
    validation_matrix = select_rows_by_qids(feature_result.matrix, set(split.validation_qids))
    test_matrix = select_rows_by_qids(feature_result.matrix, set(split.test_qids))
    if train_matrix.embeddings.shape[0] == 0:
        raise RuntimeError('Training split is empty after QID-level filtering.')

    classifier = fit_softmax_classifier(
        train_matrix.embeddings,
        labels_from_rows(train_matrix.rows),
        validation_features=validation_matrix.embeddings if validation_matrix.rows else None,
        validation_labels=labels_from_rows(validation_matrix.rows) if validation_matrix.rows else None,
        config=SoftmaxClassifierConfig(
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            l2_weight=args.l2_weight,
            seed=args.seed,
        ),
    )

    evaluations = []
    for split_name, split_matrix in (
        ('train', train_matrix),
        ('validation', validation_matrix),
        ('test', test_matrix),
    ):
        if not split_matrix.rows:
            continue
        predicted = predict_labels(classifier, split_matrix.embeddings)
        evaluations.append(
            evaluate_predictions(
                split_name=split_name,
                rows=split_matrix.rows,
                predicted_labels=predicted,
                classes=classifier.classes,
            )
        )

    output_dir = args.output_dir or _default_output_dir()
    write_metrics_tsv(output_dir / 'metrics.tsv', evaluations)
    write_split_predictions_tsv(output_dir / 'predictions.tsv', evaluations)
    write_json(
        output_dir / 'config.json',
        {
            'graph_embedding_dir': str(config.graph_embedding_dir),
            'audio_embedding_dir': str(config.audio_embedding_dir),
            'language_embedding_dir': str(config.language_embedding_dir),
            'taxonomy_graph': str(taxonomy_graph_path),
            'target_rank': args.target_rank,
            'modalities': args.modalities,
            'validation_fraction': args.validation_fraction,
            'test_fraction': args.test_fraction,
            'seed': args.seed,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'l2_weight': args.l2_weight,
        },
    )
    write_json(
        output_dir / 'dataset_summary.json',
        {
            'row_count': len(feature_result.matrix.rows),
            'embedding_dim': int(feature_result.matrix.embeddings.shape[1]),
            'feature_slices': {key: [start, stop] for key, (start, stop) in feature_result.feature_slices.items()},
            'class_count': len(classifier.classes),
            'classes': classifier.classes,
            'train_rows': len(train_matrix.rows),
            'validation_rows': len(validation_matrix.rows),
            'test_rows': len(test_matrix.rows),
            'train_qids': len(split.train_qids),
            'validation_qids': len(split.validation_qids),
            'test_qids': len(split.test_qids),
        },
    )
    write_json(output_dir / 'training_trace.json', {'training_trace': classifier.training_trace})
    print(
        json.dumps(
            {
                'output_dir': str(output_dir),
                'class_count': len(classifier.classes),
                'row_count': len(feature_result.matrix.rows),
                'train_rows': len(train_matrix.rows),
                'validation_rows': len(validation_matrix.rows),
                'test_rows': len(test_matrix.rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subcommand = args.subcommand or 'inspect'
    if subcommand == 'inspect':
        return _run_inspect(args)
    if subcommand == 'run-baseline':
        return _run_baseline(args)
    raise ValueError(f'Unsupported multimodal subcommand: {subcommand}')


if __name__ == '__main__':
    raise SystemExit(main())
