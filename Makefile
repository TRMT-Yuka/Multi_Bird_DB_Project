PYTHON ?= python3
PYTHONPATH := src

.PHONY: bootstrap extract-qids generate-fetch build-ontology build-wikipedia-manifest fetch-wikipedia-xml extract-wikipedia-text verify

bootstrap: extract-qids generate-fetch

extract-qids:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-qids

generate-fetch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli generate-fetch

build-ontology:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-ontology

build-wikipedia-manifest:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli build-wikipedia-manifest

fetch-wikipedia-xml:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli fetch-wikipedia-xml

extract-wikipedia-text:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m multi_bird_db.cli extract-wikipedia-text

verify:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m py_compile src/multi_bird_db/*.py
