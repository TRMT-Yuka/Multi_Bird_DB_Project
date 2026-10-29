#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE_SCRIPT="${ROOT_DIR}/experiments/exp3_taxon_probe.py"

cd "${ROOT_DIR}"

graphs=(gcn grace graphsage node2vec transe)
languages=(en ja)
audios=(birdnet_acoustic_2_4_pb_07300911 perch_07300906 wav2vec2_base)

for graph in "${graphs[@]}"; do
  for language in "${languages[@]}"; do
    python3 "${PROBE_SCRIPT}" \
      --top-k 10 \
      --target-ranks family \
      --modalities GL \
      --graph-runs "${graph}" \
      --language-runs "${language}"
  done

  for audio in "${audios[@]}"; do
    python3 "${PROBE_SCRIPT}" \
      --top-k 10 \
      --target-ranks family \
      --modalities GA \
      --graph-runs "${graph}" \
      --audio-runs "${audio}"
  done

  for language in "${languages[@]}"; do
    for audio in "${audios[@]}"; do
      python3 "${PROBE_SCRIPT}" \
        --top-k 10 \
        --target-ranks family \
        --modalities GLA \
        --graph-runs "${graph}" \
        --language-runs "${language}" \
        --audio-runs "${audio}"
    done
  done
done
