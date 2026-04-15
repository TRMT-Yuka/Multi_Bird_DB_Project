PYTHON ?= python3
PYTHONPATH := src
EXTRACT_DUMP_JSON_ARGS ?=

.PHONY: extract-qids extract-dump-json download-wikidata-dump build-ontology build-graph build-sqlite build-embeddings serve-graph build-wikipedia-manifest fetch-wikipedia-xml extract-wikipedia-text verify

extract-qids:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-qids

extract-dump-json:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-dump-json $(EXTRACT_DUMP_JSON_ARGS)

download-wikidata-dump:
	bash scripts/download_wikidata_dump.sh

build-ontology:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-ontology

build-graph:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-graph

build-sqlite:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-sqlite

build-embeddings:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-embeddings

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
