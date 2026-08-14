#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

# family / single modality
# already generated
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities G --graph-runs gcn
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities G --graph-runs grace
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities G --graph-runs graphsage
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities G --graph-runs node2vec
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities G --graph-runs transe
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities L --language-runs en
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities L --language-runs ja
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities A --audio-runs birdnet_acoustic_2_4_pb_07300911
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities A --audio-runs perch_07300906
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities A --audio-runs wav2vec2_base

# family / GL
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs gcn --language-runs en
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs gcn --language-runs ja
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs grace --language-runs en
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs grace --language-runs ja
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs graphsage --language-runs en
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs graphsage --language-runs ja
# already generated
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs node2vec --language-runs en
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs node2vec --language-runs ja
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs transe --language-runs en
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GL --graph-runs transe --language-runs ja

# family / GA
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs gcn --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs gcn --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs gcn --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs grace --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs grace --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs grace --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs graphsage --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs graphsage --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs graphsage --audio-runs wav2vec2_base
# already generated
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs node2vec --audio-runs birdnet_acoustic_2_4_pb_07300911
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs node2vec --audio-runs perch_07300906
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs node2vec --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs transe --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs transe --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GA --graph-runs transe --audio-runs wav2vec2_base

# family / LA
# already generated
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities LA --language-runs en --audio-runs birdnet_acoustic_2_4_pb_07300911
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities LA --language-runs en --audio-runs perch_07300906
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities LA --language-runs en --audio-runs wav2vec2_base
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities LA --language-runs ja --audio-runs birdnet_acoustic_2_4_pb_07300911
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities LA --language-runs ja --audio-runs perch_07300906
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities LA --language-runs ja --audio-runs wav2vec2_base

# family / GLA
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs gcn --language-runs en --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs gcn --language-runs en --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs gcn --language-runs en --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs gcn --language-runs ja --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs gcn --language-runs ja --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs gcn --language-runs ja --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs grace --language-runs en --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs grace --language-runs en --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs grace --language-runs en --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs grace --language-runs ja --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs grace --language-runs ja --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs grace --language-runs ja --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs graphsage --language-runs en --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs graphsage --language-runs en --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs graphsage --language-runs en --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs graphsage --language-runs ja --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs graphsage --language-runs ja --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs graphsage --language-runs ja --audio-runs wav2vec2_base
# already generated
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs node2vec --language-runs en --audio-runs birdnet_acoustic_2_4_pb_07300911
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs node2vec --language-runs en --audio-runs perch_07300906
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs node2vec --language-runs en --audio-runs wav2vec2_base
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs node2vec --language-runs ja --audio-runs birdnet_acoustic_2_4_pb_07300911
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs node2vec --language-runs ja --audio-runs perch_07300906
# python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs node2vec --language-runs ja --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs transe --language-runs en --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs transe --language-runs en --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs transe --language-runs en --audio-runs wav2vec2_base
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs transe --language-runs ja --audio-runs birdnet_acoustic_2_4_pb_07300911
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs transe --language-runs ja --audio-runs perch_07300906
python3 experiments/root_short_reproduction/exp3_taxon_probe.py --top-k 10 --target-ranks family --modalities GLA --graph-runs transe --language-runs ja --audio-runs wav2vec2_base
