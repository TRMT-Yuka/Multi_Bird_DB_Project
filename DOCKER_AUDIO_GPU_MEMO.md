# Docker Audio GPU Memo

This file tracks temporary, container-local adjustments that should later be reflected in `Dockerfile.audio-gpu`.

## Current status

- GPU audio container starts successfully via `make run-audio-gpu-shell`.
- Running BirdNET inside the container currently fails at CLI startup because `multi_bird_db.cli` imports `dump_extract.py`, which requires `qwikidata`.

## Pending Dockerfile updates

- Add `qwikidata` to the GPU audio image environment.
  - Reason: `python3 -m multi_bird_db.cli build-audio-embeddings --backend birdnet --device cuda`
    imports the top-level CLI module, which imports `dump_extract.py`, which imports `qwikidata`.

## If we make container-local fixes before updating Dockerfile

Record them here in order, for example:

- `python3 -m pip install qwikidata`
- any additional `pip install ...`
- any additional `apt-get install ...`

## Recorded temporary fixes

- Observed error:
  - `ModuleNotFoundError: No module named 'qwikidata'`
- Cause:
  - `python3 -m multi_bird_db.cli build-audio-embeddings --backend birdnet --device cuda`
    imports the top-level CLI module, which imports `dump_extract.py`, which imports `qwikidata`.
- Temporary in-container fix to apply:
  - `python3 -m pip install qwikidata`
- Follow-up:
  - Reflect this dependency in `Dockerfile.audio-gpu` after the workflow is stabilized.

## Final cleanup plan

Once the experiment setup is stable:

1. Reflect all required package additions in `Dockerfile.audio-gpu`.
2. Rebuild with `make build-audio-gpu-image`.
3. Re-test BirdNET / Perch in a fresh container.
4. Remove any no-longer-needed ad hoc notes from this memo.
