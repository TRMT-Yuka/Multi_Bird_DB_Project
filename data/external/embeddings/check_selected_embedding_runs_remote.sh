#!/usr/bin/env bash
set -u

ROOT="${1:-/home/teramoto/data5_teramoto/Multi_Bird_DB_Project}"
SELECTED_RUNS="$ROOT/data/external/embeddings/selected_runs.json"

if [ ! -f "$SELECTED_RUNS" ]; then
  echo "MISSING selected_runs.json: $SELECTED_RUNS"
  exit 1
fi

python3 - "$ROOT" "$SELECTED_RUNS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
selected_runs = Path(sys.argv[2])
payload = json.loads(selected_runs.read_text(encoding="utf-8"))
runs = payload.get("runs", {})

missing_runs = []
print("SELECTED_EMBEDDING_RUN_CHECK")
print(f"root={root}")
print(f"selected_runs={selected_runs}")
print()

for modality, modality_runs in runs.items():
    for run_name, rel_path in modality_runs.items():
        run_path = root / rel_path
        required = ["embeddings.npy", "qids.json"]
        if modality == "audio":
            required.append("audio_manifest.tsv")
        missing = [name for name in required if not (run_path / name).is_file()]
        status = "OK" if run_path.is_dir() and not missing else "MISSING"
        print(f"{status}\t{modality}:{run_name}\t{rel_path}")
        if missing:
            print(f"  missing_files={','.join(missing)}")
            missing_runs.append((modality, run_name, rel_path, missing))

print()
print("MISSING_RUNS")
if not missing_runs:
    print("none")
else:
    for modality, run_name, rel_path, missing in missing_runs:
        print(f"{modality}:{run_name}\t{rel_path}\tmissing={','.join(missing)}")
PY
