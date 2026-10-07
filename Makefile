PYTHON ?= python3
PYTHONPATH := src
EXTRACT_DUMP_JSON_ARGS ?=
EMBEDDING_ALGORITHM ?= node2vec
INITIAL_FEATURES ?= embedding
OUTPUT_DIM ?= 128
HIDDEN_DIM ?= 32
INSPECT_GRAPH_EMBEDDING_METHOD ?= all
EXP3_TARGET_RANKS ?= family,order
EXP3_ARGS ?=
TAXON_RANKS ?= order,family
TAXON_GRAPH_INPUT ?=
TAXON_ONTOLOGY_INPUT ?=
TAXON_OUTPUT_DIR ?=

.PHONY: extract-qids extract-dump-json download-wikidata-dump build-ontology extract-xeno-canto-ids fetch-xeno-canto-recording-json fetch-xeno-canto-species-pages extract-xeno-canto-recording-ids fetch-xeno-canto-audio download-audio-models check-birdnet-gpu build-audio-embeddings-wav2vec2 build-audio-embeddings-wav2vec2-finetuned finetune-wav2vec2-crossval repair-xeno-canto-audio build-audio-embeddings-birdnet build-audio-embeddings-birdnet-gpu build-audio-embeddings-perch build-graph build-taxon-labels build-sqlite build-embeddings build-node2vec-embeddings build-gcn-embeddings build-grace-embeddings build-graphsage-embeddings build-transe-embeddings inspect-multimodal-sources run-exp1-simmatrix run-exp1-simmatrix-ordered run-exp2-integrated-graph run-exp3-sub1-audio run-exp3-search run-exp3-taxon-probe report-qid-coverage collect-xeno-canto-num-recordings build-language-surface-manifest build-language-embeddings check-gpu serve-graph build-wikipedia-manifest fetch-wikipedia-xml extract-wikipedia-text verify

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

check-birdnet-gpu:
	conda run -n birdnet --no-capture-output env TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 BIRDNET_APP_DATA=$(CURDIR)/temp/birdnet_appdata PYTHONPATH=$(PYTHONPATH) python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

build-audio-embeddings-wav2vec2:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-audio-embeddings --backend wav2vec2

build-audio-embeddings-wav2vec2-finetuned:
	for fold in 0 1 2 3 4; do \
		PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-audio-embeddings \
			--backend wav2vec2 \
			--model-name data/external/models/audio/wav2vec2-finetuned/wav2vec2-model_$$fold \
			--model-label wav2vec2-model_$$fold \
			--output-dir data/external/embeddings/audio/wav2vec2-finetuned \
			--device cuda \
			--resume-existing; \
	done

finetune-wav2vec2-crossval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli finetune-wav2vec2-crossval --device cuda

repair-xeno-canto-audio:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli repair-xeno-canto-audio

build-audio-embeddings-birdnet:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-audio-embeddings --backend birdnet

build-audio-embeddings-birdnet-gpu:
	mkdir -p temp/birdnet_appdata
	conda run -n birdnet --no-capture-output env TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0 BIRDNET_APP_DATA=$(CURDIR)/temp/birdnet_appdata PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-audio-embeddings --backend birdnet --device cuda

build-audio-embeddings-perch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-audio-embeddings --backend perch

build-graph:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-graph

build-taxon-labels:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-taxon-labels --ranks $(TAXON_RANKS) $(if $(TAXON_GRAPH_INPUT),--graph-input $(TAXON_GRAPH_INPUT),) $(if $(TAXON_ONTOLOGY_INPUT),--ontology-input $(TAXON_ONTOLOGY_INPUT),) $(if $(TAXON_OUTPUT_DIR),--output-dir $(TAXON_OUTPUT_DIR),)

build-sqlite:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-sqlite

build-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm $(EMBEDDING_ALGORITHM)

build-node2vec-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm node2vec

build-gcn-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm gcn --hidden-dim $(HIDDEN_DIM) --layers 2 --epochs 200 --learning-rate 0.01 --negative-samples 20 --weight-decay 1e-5 --output-dim $(OUTPUT_DIM) --initial-features $(INITIAL_FEATURES)

build-grace-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm grace --hidden-dim $(HIDDEN_DIM) --output-dim $(OUTPUT_DIM) --initial-features $(INITIAL_FEATURES)

build-graphsage-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm graphsage --device cuda --hidden-dim $(HIDDEN_DIM) --output-dim $(OUTPUT_DIM) --epochs 200 --negative-samples 1 --graphsage-num-neighbors-1 25 --graphsage-num-neighbors-2 10 --weight-decay 1e-5 --initial-features $(INITIAL_FEATURES)

build-transe-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm transe

inspect-multimodal-sources:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli inspect-multimodal-sources --graph-embedding-method $(INSPECT_GRAPH_EMBEDDING_METHOD)

run-exp1-simmatrix:
	$(PYTHON) experiments/root_short_reproduction/exp1_similarity_matrices.py

run-exp1-simmatrix-ordered:
	$(PYTHON) experiments/root_short_reproduction/exp1_similarity_matrices_ordered.py

run-exp2-integrated-graph:
	$(PYTHON) experiments/root_short_reproduction/exp2_integrated_embedding_graphs.py

run-exp3-sub1-audio:
	$(PYTHON) experiments/root_short_reproduction/exp3_sub1_audio_pretraining.py

run-exp3-search:
	$(PYTHON) experiments/root_short_reproduction/exp3_taxon_probe.py --target-ranks $(EXP3_TARGET_RANKS) $(EXP3_ARGS)

run-exp3-taxon-probe: run-exp3-search

report-qid-coverage:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/root_short_reproduction/exp0_qid_coverage.py

collect-xeno-canto-num-recordings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/root_short_reproduction/exp1_collect_xeno_canto_num_recordings.py

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
