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
    wikidata_dump_dir: Path
    wikidata_dump_file: Path
    qids_tsv: Path
    xeno_canto_ids_tsv: Path
    json_dir: Path
    dump_extract_checkpoint: Path
    ontology_pkl: Path
    graph_dir: Path
    taxonomy_graph_pkl: Path
    taxonomy_graph_html_dir: Path
    embeddings_dir: Path
    graph_embeddings_dir: Path
    raw_audio_dir: Path
    xeno_canto_raw_dir: Path
    xeno_canto_after_202505_dir: Path
    audio_dir: Path
    xeno_canto_audio_dir: Path
    sqlite_dir: Path
    taxonomy_sqlite_path: Path
    wikipedia_manifest_tsv: Path


def get_project_paths() -> ProjectPaths:
    """Return all standard paths from the project root. / プロジェクトルート基準の標準パスを返す。"""

    root = Path(__file__).resolve().parents[2]
    raw_wikidata_dir = root / "data" / "raw" / "wikidata"
    wikidata_dump_dir = raw_wikidata_dir / "dumps"
    interim_wikidata_dir = root / "data" / "interim" / "wikidata"
    processed_dir = root / "data" / "processed"
    embeddings_dir = root / "data" / "external" / "embeddings"
    raw_audio_dir = root / "data" / "raw" / "audio"
    xeno_canto_raw_dir = root / "data" / "raw" / "xeno-canto"
    audio_dir = root / "data" / "external" / "audio"
    graph_dir = processed_dir / "graph"
    documents_dir = root / "data" / "external" / "documents"
    sqlite_dir = root / "data" / "external" / "sqlite"
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
        wikidata_dump_dir=wikidata_dump_dir,
        wikidata_dump_file=wikidata_dump_dir / "latest-all.json.bz2",
        qids_tsv=interim_wikidata_dir / "bird_qids.tsv",
        xeno_canto_ids_tsv=interim_wikidata_dir / "bird_xeno_canto_ids.tsv",
        json_dir=interim_wikidata_dir / "json",
        dump_extract_checkpoint=interim_wikidata_dir / "local_dump_extract_checkpoint.json",
        ontology_pkl=processed_dir / "bird_ontology.pkl",
        graph_dir=graph_dir,
        taxonomy_graph_pkl=graph_dir / "bird_taxonomy_graph.pkl",
        taxonomy_graph_html_dir=graph_dir / "dash",
        embeddings_dir=embeddings_dir,
        graph_embeddings_dir=embeddings_dir / "graph",
        raw_audio_dir=raw_audio_dir,
        xeno_canto_raw_dir=xeno_canto_raw_dir,
        xeno_canto_after_202505_dir=xeno_canto_raw_dir / "after_202505",
        audio_dir=audio_dir,
        xeno_canto_audio_dir=audio_dir / "xeno-canto",
        sqlite_dir=sqlite_dir,
        taxonomy_sqlite_path=sqlite_dir / "taxonomy" / "bird_taxonomy.sqlite",
        wikipedia_manifest_tsv=processed_dir / "wikipedia_article_manifest.tsv",
    )
