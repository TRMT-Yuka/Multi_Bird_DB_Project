from __future__ import annotations

"""Multimodal experiment utilities isolated from legacy pipelines."""

from .classifiers import (
    SoftmaxClassifierConfig,
    SoftmaxClassifierModel,
    fit_softmax_classifier,
    predict_label_indices,
    predict_labels,
    predict_probabilities,
)
from .evaluate import SplitEvaluation, evaluate_predictions, labels_from_rows, select_rows_by_qids
from .expanders import build_multimodal_feature_matrix
from .features import FeatureAssemblyResult, build_feature_matrix, concatenate_sample_vectors
from .labels import TaxonLabelAssignment, assign_labels_for_qids, assign_upper_taxon_label, iter_ancestor_chain
from .splits import QidSplit, qid_split_lookup, split_qids
from .loaders import (
    find_latest_audio_embedding_dir,
    find_latest_graph_embedding_dir,
    find_latest_language_embedding_dir,
    load_audio_embedding_run,
    load_graph_embedding_run,
    load_language_embedding_run,
    resolve_default_source_config,
)
from .types import (
    AudioEmbeddingRun,
    AudioEmbeddingRow,
    EmbeddingSourceConfig,
    MultimodalFeatureMatrix,
    MultimodalSampleRow,
    GraphEmbeddingRun,
    LanguageEmbeddingRun,
    LanguageSurfaceRow,
)

__all__ = [
    'AudioEmbeddingRun',
    'AudioEmbeddingRow',
    'EmbeddingSourceConfig',
    'QidSplit',
    'SoftmaxClassifierConfig',
    'SoftmaxClassifierModel',
    'SplitEvaluation',
    'TaxonLabelAssignment',
    'MultimodalFeatureMatrix',
    'MultimodalSampleRow',
    'FeatureAssemblyResult',
    'GraphEmbeddingRun',
    'LanguageEmbeddingRun',
    'LanguageSurfaceRow',
    'assign_labels_for_qids',
    'assign_upper_taxon_label',
    'build_multimodal_feature_matrix',
    'build_feature_matrix',
    'concatenate_sample_vectors',
    'evaluate_predictions',
    'fit_softmax_classifier',
    'iter_ancestor_chain',
    'find_latest_audio_embedding_dir',
    'find_latest_graph_embedding_dir',
    'find_latest_language_embedding_dir',
    'labels_from_rows',
    'load_audio_embedding_run',
    'load_graph_embedding_run',
    'load_language_embedding_run',
    'predict_label_indices',
    'predict_labels',
    'predict_probabilities',
    'qid_split_lookup',
    'resolve_default_source_config',
    'select_rows_by_qids',
    'split_qids',
]
