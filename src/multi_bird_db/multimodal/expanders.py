from __future__ import annotations

from collections import defaultdict
from itertools import product

from .features import FeatureAssemblyResult, build_feature_matrix, concatenate_sample_vectors
from .types import (
    AudioEmbeddingRun,
    AudioEmbeddingRow,
    GraphEmbeddingRun,
    LanguageEmbeddingRun,
    LanguageSurfaceRow,
    MultimodalSampleRow,
)
from .labels import TaxonLabelAssignment


def _group_audio_rows_by_qid(audio_run: AudioEmbeddingRun) -> dict[str, list[AudioEmbeddingRow]]:
    grouped: dict[str, list[AudioEmbeddingRow]] = defaultdict(list)
    for row in audio_run.rows:
        grouped[row.qid].append(row)
    return dict(grouped)


def _group_language_rows_by_qid(language_run: LanguageEmbeddingRun) -> dict[str, list[LanguageSurfaceRow]]:
    grouped: dict[str, list[LanguageSurfaceRow]] = defaultdict(list)
    for row in language_run.rows:
        grouped[row.qid].append(row)
    return dict(grouped)


def _assignment_map(assignments: list[TaxonLabelAssignment]) -> dict[str, TaxonLabelAssignment]:
    return {assignment.qid: assignment for assignment in assignments}


def _pattern_name(*, include_graph: bool, include_audio: bool, include_language: bool) -> str:
    names: list[str] = []
    if include_graph:
        names.append('graph')
    if include_audio:
        names.append('audio')
    if include_language:
        names.append('language')
    return '+'.join(names)


def build_multimodal_feature_matrix(
    *,
    graph_run: GraphEmbeddingRun,
    audio_run: AudioEmbeddingRun,
    language_run: LanguageEmbeddingRun,
    assignments: list[TaxonLabelAssignment],
    include_graph: bool,
    include_audio: bool,
    include_language: bool,
) -> FeatureAssemblyResult:
    if not any((include_graph, include_audio, include_language)):
        raise ValueError('At least one modality must be enabled.')

    assignment_by_qid = _assignment_map(assignments)
    audio_rows_by_qid = _group_audio_rows_by_qid(audio_run)
    language_rows_by_qid = _group_language_rows_by_qid(language_run)
    graph_index_by_qid = {qid: index for index, qid in enumerate(graph_run.qids)}
    modality_pattern = _pattern_name(
        include_graph=include_graph,
        include_audio=include_audio,
        include_language=include_language,
    )

    rows_and_vectors: list[tuple[MultimodalSampleRow, object]] = []
    first_feature_slices: dict[str, tuple[int, int]] | None = None
    sample_counter = 0

    for qid in sorted(assignment_by_qid):
        if include_graph and qid not in graph_index_by_qid:
            continue
        graph_indices = [graph_index_by_qid[qid]] if include_graph else [None]
        audio_candidates = audio_rows_by_qid.get(qid, []) if include_audio else [None]
        language_candidates = language_rows_by_qid.get(qid, []) if include_language else [None]
        if include_audio and not audio_candidates:
            continue
        if include_language and not language_candidates:
            continue

        assignment = assignment_by_qid[qid]
        for graph_index, audio_row, language_row in product(graph_indices, audio_candidates, language_candidates):
            graph_vector = graph_run.embeddings[graph_index] if graph_index is not None else None
            audio_vector = audio_run.embeddings[audio_row.embedding_index] if audio_row is not None else None
            language_vector = language_run.embeddings[language_row.embedding_index] if language_row is not None else None
            vector, feature_slices = concatenate_sample_vectors(
                graph_vector=graph_vector,
                audio_vector=audio_vector,
                language_vector=language_vector,
            )
            if first_feature_slices is None:
                first_feature_slices = dict(feature_slices)
            rows_and_vectors.append(
                (
                    MultimodalSampleRow(
                        sample_id=f'{qid}__{sample_counter:06d}',
                        qid=qid,
                        graph_embedding_index=graph_index,
                        audio_embedding_index=None if audio_row is None else audio_row.embedding_index,
                        language_embedding_index=None if language_row is None else language_row.embedding_index,
                        modality_pattern=modality_pattern,
                        target_rank=assignment.target_rank,
                        target_label=assignment.label_qid,
                    ),
                    vector,
                )
            )
            sample_counter += 1

    result = build_feature_matrix(rows_and_vectors, metadata={"modality_pattern": modality_pattern})
    return FeatureAssemblyResult(matrix=result.matrix, feature_slices=first_feature_slices or {})
