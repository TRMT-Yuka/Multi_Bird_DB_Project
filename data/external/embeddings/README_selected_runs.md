# Selected Embedding Runs

This directory can contain multiple timestamped embedding runs.
Downstream experiments should not blindly use every run under `data/external/embeddings`.
Instead, use [selected_runs.json](selected_runs.json) as the canonical run list.

## Current Selection

### Graph

- `gcn`: `data/external/embeddings/graph/gcn/05231851`
- `grace`: `data/external/embeddings/graph/grace/05231854`
- `graphsage`: `data/external/embeddings/graph/graphsage/07030546`
- `node2vec`: `data/external/embeddings/graph/node2vec/06280918`
- `transe`: `data/external/embeddings/graph/transe/05231858`

### Language

- `en`: `data/external/embeddings/language/en`
- `ja`: `data/external/embeddings/language/ja`

### Audio

- `wav2vec2_base`: `data/external/embeddings/audio/wav2vec2/facebook_wav2vec2-base-960h/06300702`

## Pending

The following audio embedding runs are visible or planned but not yet selected.
Move them into `runs.audio` only after the intended final run is complete and `embeddings.npy`, `qids.json`, `audio_manifest.tsv`, and `metadata.json` exist.

- `wav2vec2_finetuned_model_0`: `data/external/embeddings/audio/wav2vec2-finetuned/wav2vec2/wav2vec2-model_0/07300813`
- `wav2vec2_finetuned_model_1`: `data/external/embeddings/audio/wav2vec2-finetuned/wav2vec2/wav2vec2-model_1/07300858`

- `birdnet_2_in_progress`: `data/external/embeddings/audio/birdnet/birdnet_2-acoustic-2.4-pb`
