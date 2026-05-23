PYTHON ?= python3
PYTHONPATH := src
EXTRACT_DUMP_JSON_ARGS ?=
EMBEDDING_ALGORITHM ?= node2vec

.PHONY: extract-qids extract-dump-json download-wikidata-dump build-ontology extract-xeno-canto-ids fetch-xeno-canto-audio build-graph build-sqlite build-embeddings build-node2vec-embeddings build-gcn-embeddings build-grace-embeddings build-graphsage-embeddings build-transe-embeddings evaluate-graph-embeddings build-language-surface-manifest build-language-embeddings check-gpu serve-graph build-wikipedia-manifest fetch-wikipedia-xml extract-wikipedia-text verify

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

fetch-xeno-canto-audio:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli fetch-xeno-canto-audio

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
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings --algorithm graphsage

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
