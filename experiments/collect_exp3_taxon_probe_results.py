from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "experiments" / "exp3" / "taxon_probe"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp3" / "taxon_probe_collected"
DEFAULT_FILENAMES = ("metrics.tsv", "metadata.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect EXP3 taxon-probe result files into one flat directory."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Also copy predictions.tsv files (approximately 783 MB currently).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_files(source_dir: Path, include_predictions: bool) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    filenames = set(DEFAULT_FILENAMES)
    if include_predictions:
        filenames.add("predictions.tsv")
    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.name in filenames
    )
    if not files:
        raise FileNotFoundError(f"No result files found below: {source_dir}")
    return files


def flat_name(source_dir: Path, source_path: Path) -> str:
    relative_path = source_path.relative_to(source_dir)
    if len(relative_path.parts) == 1:
        return f"ROOT__{relative_path.name}"
    return "__".join(relative_path.parts)


def collect(
    source_dir: Path, output_dir: Path, include_predictions: bool
) -> list[dict[str, object]]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be the source or its descendant")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )

    source_files = discover_files(source_dir, include_predictions)
    destination_names = [flat_name(source_dir, path) for path in source_files]
    if len(destination_names) != len(set(destination_names)):
        raise ValueError("Flattened filenames would collide")

    output_dir.mkdir(parents=True)
    manifest_rows: list[dict[str, object]] = []
    try:
        for source_path, destination_name in zip(
            source_files, destination_names, strict=True
        ):
            destination_path = output_dir / destination_name
            shutil.copy2(source_path, destination_path)
            relative_path = source_path.relative_to(source_dir)
            parts = relative_path.parts
            manifest_rows.append(
                {
                    "target_rank": parts[0] if len(parts) >= 4 else "ROOT",
                    "modality": parts[1] if len(parts) >= 4 else "ROOT",
                    "run_name": parts[2] if len(parts) >= 4 else "ROOT",
                    "file_type": source_path.name,
                    "source_relative_path": relative_path.as_posix(),
                    "collected_filename": destination_name,
                    "size_bytes": destination_path.stat().st_size,
                    "sha256": sha256_file(destination_path),
                }
            )

        manifest_path = output_dir / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
    except Exception:
        shutil.rmtree(output_dir)
        raise
    return manifest_rows


def main() -> None:
    args = parse_args()
    rows = collect(args.source_dir, args.output_dir, args.include_predictions)
    counts: dict[str, int] = {}
    for row in rows:
        file_type = str(row["file_type"])
        counts[file_type] = counts.get(file_type, 0) + 1
    print(f"Collected {len(rows)} files from {args.source_dir.resolve()}")
    for file_type, count in sorted(counts.items()):
        print(f"  {file_type}: {count}")
    print(f"Output: {args.output_dir.resolve()}")
    print(f"Manifest: {args.output_dir.resolve() / 'manifest.csv'}")


if __name__ == "__main__":
    main()
