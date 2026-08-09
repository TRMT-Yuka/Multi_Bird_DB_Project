from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from .config import get_project_paths
from .embeddings import load_graph
from .graph import build_taxonomy_graph
from .multimodal.labels import assign_upper_taxon_label


DEFAULT_RANKS = ("order", "family")


@dataclass(frozen=True, slots=True)
class TaxonLabelRow:
    qid: str
    labels: dict[str, dict[str, Any] | None]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _normalize_ranks(ranks: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for rank in ranks:
        item = str(rank).strip().lower().replace("_", " ")
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    if not normalized:
        raise ValueError("At least one rank must be specified.")
    return normalized


def load_cached_taxon_labels(tsv_path: Path, ranks: list[str]) -> dict[str, dict[str, dict[str, Any] | None]]:
    normalized_ranks = _normalize_ranks(ranks)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Taxon label cache does not exist: {tsv_path}")

    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    cache: dict[str, dict[str, dict[str, Any] | None]] = {}
    for row in rows:
        qid = str(row.get("qid") or "").strip()
        if not qid:
            continue
        cache[qid] = {}
        for rank in normalized_ranks:
            label_qid = str(row.get(f"{rank}_qid") or "").strip()
            label_name = str(row.get(f"{rank}_label") or row.get(f"{rank}_name") or "").strip()
            distance_raw = str(row.get(f"{rank}_distance") or "").strip()
            if not label_qid and not label_name and not distance_raw:
                cache[qid][rank] = None
                continue
            try:
                distance = int(distance_raw)
            except ValueError:
                distance = None
            cache[qid][rank] = {
                "label_qid": label_qid,
                "label_name": label_name,
                "distance_to_label": distance,
            }
    return cache


def _load_taxonomy_graph(*, graph_input: Path, ontology_input: Path | None, root_qid: str) -> nx.DiGraph:
    if graph_input.exists():
        return load_graph(graph_input)
    if ontology_input is None or not ontology_input.exists():
        raise FileNotFoundError(f"Neither graph nor ontology exists: graph={graph_input}, ontology={ontology_input}")
    with ontology_input.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Ontology PKL must contain a list, got: {type(rows).__name__}")
    return build_taxonomy_graph(rows, root_qid=root_qid)


def _resolve_rank_labels(graph: nx.DiGraph, qid: str, ranks: list[str]) -> dict[str, dict[str, Any] | None]:
    resolved: dict[str, dict[str, Any] | None] = {}
    for rank in ranks:
        assignment = assign_upper_taxon_label(graph, qid, rank)
        if assignment is None:
            resolved[rank] = None
            continue
        resolved[rank] = {
            "label_qid": assignment.label_qid,
            "label_name": assignment.label_name,
            "distance_to_label": assignment.distance_to_label,
        }
    return resolved


def build_taxon_labels(
    *,
    graph_input: Path,
    ontology_input: Path | None,
    output_dir: Path,
    ranks: list[str],
    root_qid: str,
) -> dict[str, Any]:
    graph = _load_taxonomy_graph(graph_input=graph_input, ontology_input=ontology_input, root_qid=root_qid)
    normalized_ranks = _normalize_ranks(ranks)
    qids = sorted(str(qid) for qid in graph.nodes())
    rows: list[TaxonLabelRow] = []
    complete_count = 0
    partial_count = 0

    for qid in qids:
        labels = _resolve_rank_labels(graph, qid, normalized_ranks)
        rows.append(TaxonLabelRow(qid=qid, labels=labels))
        if all(labels.get(rank) for rank in normalized_ranks):
            complete_count += 1
        else:
            partial_count += 1

    tsv_rows: list[dict[str, object]] = []
    json_payload: dict[str, Any] = {}
    for row in rows:
        tsv_row: dict[str, object] = {"qid": row.qid}
        json_payload[row.qid] = {}
        missing_ranks: list[str] = []
        for rank in normalized_ranks:
            label = row.labels.get(rank)
            if label is None:
                tsv_row[f"{rank}_qid"] = ""
                tsv_row[f"{rank}_label"] = ""
                tsv_row[f"{rank}_distance"] = ""
                missing_ranks.append(rank)
                json_payload[row.qid][rank] = None
                continue
            tsv_row[f"{rank}_qid"] = label["label_qid"]
            tsv_row[f"{rank}_label"] = label["label_name"]
            tsv_row[f"{rank}_distance"] = label["distance_to_label"]
            json_payload[row.qid][rank] = label
        tsv_row["complete_rank_count"] = sum(1 for rank in normalized_ranks if row.labels.get(rank))
        tsv_row["missing_ranks"] = ",".join(missing_ranks)
        tsv_rows.append(tsv_row)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    columns = ["qid"]
    for rank in normalized_ranks:
        columns.extend([f"{rank}_qid", f"{rank}_label", f"{rank}_distance"])
    columns.extend(["complete_rank_count", "missing_ranks"])

    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "qid_taxon_labels.tsv"
    json_path = output_dir / "qid_taxon_labels.json"
    meta_path = output_dir / "qid_taxon_labels.meta.json"

    _write_tsv(tsv_path, tsv_rows, columns)
    _write_json(json_path, json_payload)
    _write_json(
        meta_path,
        {
            "graph_input": str(graph_input),
            "ontology_input": str(ontology_input) if ontology_input is not None else "",
            "ranks": normalized_ranks,
            "root_qid": root_qid,
            "row_count": len(rows),
            "complete_row_count": complete_count,
            "partial_row_count": partial_count,
            "generated_from": "graph" if graph_input.exists() else "ontology",
            "generated_at": generated_at,
            "output_files": {
                "tsv": str(tsv_path),
                "json": str(json_path),
            },
        },
    )
    return {
        "tsv": tsv_path,
        "json": json_path,
        "meta": meta_path,
        "row_count": len(rows),
        "complete_row_count": complete_count,
        "partial_row_count": partial_count,
    }


def build_parser() -> argparse.ArgumentParser:
    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build cached QID to upper-taxon label tables.")
    parser.add_argument("--graph-input", type=Path, default=paths.taxonomy_graph_pkl)
    parser.add_argument("--ontology-input", type=Path, default=paths.ontology_pkl)
    parser.add_argument("--output-dir", type=Path, default=paths.taxonomy_labels_dir)
    parser.add_argument("--ranks", default="order,family", help="Comma-separated upper-taxon ranks to export.")
    parser.add_argument("--root-qid", default="Q5113")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    graph_input = Path(args.graph_input)
    ontology_input = Path(args.ontology_input) if args.ontology_input else None
    output_dir = Path(args.output_dir)
    ranks = [item.strip() for item in str(args.ranks).split(",") if item.strip()]
    result = build_taxon_labels(
        graph_input=graph_input,
        ontology_input=ontology_input,
        output_dir=output_dir,
        ranks=ranks,
        root_qid=str(args.root_qid),
    )
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
