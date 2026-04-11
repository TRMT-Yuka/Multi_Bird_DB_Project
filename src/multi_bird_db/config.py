from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Store the main paths used across this project. / このプロジェクトで使う主要なパスをまとめる。"""

    root: Path
    raw_wikidata_dir: Path
    interim_wikidata_dir: Path
    processed_dir: Path
    documents_dir: Path
    wikipedia_dir: Path
    wikipedia_xml_dir: Path
    wikipedia_xml_en_dir: Path
    wikipedia_xml_ja_dir: Path
    wikipedia_text_dir: Path
    wikipedia_text_en_dir: Path
    wikipedia_text_ja_dir: Path
    query_tsv: Path
    qids_tsv: Path
    fetch_script: Path
    json_dir: Path
    ontology_tsv: Path
    wikipedia_manifest_tsv: Path


def get_project_paths() -> ProjectPaths:
    """Return all standard paths from the project root. / プロジェクトルート基準の標準パスを返す。"""

    root = Path(__file__).resolve().parents[2]
    raw_wikidata_dir = root / "data" / "raw" / "wikidata"
    interim_wikidata_dir = root / "data" / "interim" / "wikidata"
    processed_dir = root / "data" / "processed"
    documents_dir = root / "data" / "external" / "documents"
    wikipedia_dir = documents_dir / "wikipedia"
    wikipedia_xml_dir = wikipedia_dir / "xml"
    wikipedia_text_dir = wikipedia_dir / "text"
    return ProjectPaths(
        root=root,
        raw_wikidata_dir=raw_wikidata_dir,
        interim_wikidata_dir=interim_wikidata_dir,
        processed_dir=processed_dir,
        documents_dir=documents_dir,
        wikipedia_dir=wikipedia_dir,
        wikipedia_xml_dir=wikipedia_xml_dir,
        wikipedia_xml_en_dir=wikipedia_xml_dir / "en",
        wikipedia_xml_ja_dir=wikipedia_xml_dir / "ja",
        wikipedia_text_dir=wikipedia_text_dir,
        wikipedia_text_en_dir=wikipedia_text_dir / "en",
        wikipedia_text_ja_dir=wikipedia_text_dir / "ja",
        query_tsv=raw_wikidata_dir / "query.tsv",
        qids_tsv=interim_wikidata_dir / "bird_qids.tsv",
        fetch_script=interim_wikidata_dir / "fetch_entities.sh",
        json_dir=interim_wikidata_dir / "json",
        ontology_tsv=processed_dir / "bird_ontology.tsv",
        wikipedia_manifest_tsv=processed_dir / "wikipedia_article_manifest.tsv",
    )
