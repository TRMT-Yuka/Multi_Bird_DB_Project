PYTHON ?= python3
PYTHONPATH := src
EXTRACT_DUMP_JSON_ARGS ?=
EMBEDDING_ALGORITHM ?= node2vec

.PHONY: extract-qids extract-dump-json download-wikidata-dump build-ontology extract-xeno-canto-ids fetch-xeno-canto-recording-json fetch-xeno-canto-species-pages extract-xeno-canto-recording-ids fetch-xeno-canto-audio download-audio-models build-audio-gpu-image run-audio-gpu-shell check-audio-gpu-tensorflow build-audio-embeddings-wav2vec2 build-audio-embeddings-birdnet build-audio-embeddings-birdnet-gpu build-audio-embeddings-perch build-audio-embeddings-perch-gpu build-graph build-sqlite build-embeddings build-node2vec-embeddings build-gcn-embeddings build-grace-embeddings build-graphsage-embeddings build-transe-embeddings evaluate-graph-embeddings build-language-surface-manifest build-language-embeddings check-gpu serve-graph build-wikipedia-manifest fetch-wikipedia-xml extract-wikipedia-text verify

extract-qids:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-qids

extract-dump-json:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-dump-json $(EXTRACT_DUMP_JSON_ARGS)

download-wikidata-dump:
	bash scripts/download_wikidata_dump.sh

build-ontology:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-ontology

extract-xeno-canto-ids:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-xeno-canto-ids

fetch-xeno-canto-recording-json:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli fetch-xeno-canto-recording-json

fetch-xeno-canto-species-pages:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli fetch-xeno-canto-species-pages

extract-xeno-canto-recording-ids:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-xeno-canto-recording-ids

fetch-xeno-canto-audio:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli fetch-xeno-canto-audio

download-audio-models:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli download-audio-models

build-audio-gpu-image:
	bash scripts/run_audio_gpu_container.sh --build-only

run-audio-gpu-shell:
	bash scripts/run_audio_gpu_container.sh

check-audio-gpu-tensorflow:
	bash scripts/run_audio_gpu_container.sh python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

build-audio-embeddings-wav2vec2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-audio-embeddings --backend wav2vec2

build-audio-embeddings-birdnet:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-audio-embeddings --backend birdnet

build-audio-embeddings-birdnet-gpu:
	bash scripts/run_audio_gpu_container.sh python3 -m multi_bird_db.cli build-audio-embeddings --backend birdnet --device cuda

build-audio-embeddings-perch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-audio-embeddings --backend perch

build-audio-embeddings-perch-gpu:
	bash scripts/run_audio_gpu_container.sh python3 -m multi_bird_db.cli build-audio-embeddings --backend perch --device cuda

build-graph:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-graph

build-sqlite:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-sqlite

build-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm $(EMBEDDING_ALGORITHM)

build-node2vec-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm node2vec

build-gcn-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm gcn --layers 1 --epochs 300 --learning-rate 0.01 --negative-samples 20 --weight-decay 0

build-grace-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm grace

build-graphsage-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm graphsage --device cuda --dim 128 --epochs 200 --negative-samples 1 --graphsage-num-neighbors-1 8 --graphsage-num-neighbors-2 4

build-transe-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm transe

evaluate-graph-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli evaluate-graph-embeddings

build-language-surface-manifest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-language-surface-manifest

build-language-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-language-embeddings

check-gpu:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli check-gpu

serve-graph:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli serve-graph

build-wikipedia-manifest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-wikipedia-manifest

fetch-wikipedia-xml:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli fetch-wikipedia-xml

extract-wikipedia-text:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-wikipedia-text

verify:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m py_compile src/multi_bird_db/*.py
