from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config import get_project_paths

ONTOLOGY_COLUMNS = [
    "id",
    "entity_url",
    "en_name",
    "ja_name",
    "en_aliases",
    "ja_aliases",
    "img_names",
    "xeno_canto_species_id",
    "taxon_name",
    "taxon_rank",
    "taxon_rank_name",
    "taxon_rank_ja_name",
    "parent_taxon",
    "parent_taxon_name",
    "parent_taxon_ja_name",
    "enwiki_title",
    "enwiki_url",
    "jawiki_title",
    "jawiki_url",
    "path",
]


def load_entity(json_path: Path) -> dict[str, Any]:
    """Load one Wikidata JSON file and return the entity data inside it. / 1 件の Wikidata JSON を読み、内部の entity データを返す。"""

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("id") and payload.get("type"):
        return payload
    entities = payload.get("entities", {})
    if not entities:
        raise ValueError(f"No entities found in {json_path}")
    return next(iter(entities.values()))


def get_nested_value(mapping: dict[str, Any], *keys: str) -> Any:
    """Safely read a nested value from a dictionary. / 入れ子辞書から安全に値を読む。"""

    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return current


def get_label(entity: dict[str, Any], lang: str) -> str:
    """Return one label in the requested language. / 指定言語のラベルを返す。"""

    value = entity.get("labels", {}).get(lang, "")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("value", "")
    return ""


def get_aliases(entity: dict[str, Any], lang: str) -> str:
    """Return aliases as a JSON string so they fit in one TSV cell. / 別名を TSV 1 セルに入る JSON 文字列で返す。"""

    aliases = entity.get("aliases", {}).get(lang, [])
    normalized = [
        item if isinstance(item, str) else item.get("value", "")
        for item in aliases
        if (isinstance(item, str) and item) or (isinstance(item, dict) and item.get("value"))
    ]
    return json.dumps(normalized, ensure_ascii=False)


def get_claims(entity: dict[str, Any], prop: str) -> list[dict[str, Any]]:
    """Return all claim objects for one Wikidata property. / あるプロパティの claim 一覧を返す。"""

    if prop in entity.get("claims", {}):
        return entity.get("claims", {}).get(prop, [])
    return entity.get("statements", {}).get(prop, [])


def get_first_claim_value(entity: dict[str, Any], prop: str) -> Any:
    """Return the first claim value for one property. / あるプロパティの最初の値を返す。"""

    for claim in get_claims(entity, prop):
        if "mainsnak" in claim:
            value = get_nested_value(claim, "mainsnak", "datavalue", "value")
        else:
            value_field = claim.get("value", {})
            if not isinstance(value_field, dict) or value_field.get("type") != "value":
                continue
            value = value_field.get("content", "")
        if value not in ("", None):
            return value
    return ""


def get_claim_entity_id(entity: dict[str, Any], prop: str) -> str:
    """Return the first linked entity ID for one property. / あるプロパティの最初の参照先 QID を返す。"""

    value = get_first_claim_value(entity, prop)
    if not isinstance(value, dict):
        return ""
    return value.get("id", "") or get_nested_value(value, "content", "id")


def get_claim_string(entity: dict[str, Any], prop: str) -> str:
    """Return the first string claim for one property. / あるプロパティの最初の文字列値を返す。"""

    value = get_first_claim_value(entity, prop)
    return value if isinstance(value, str) else ""


def get_sitelink_title(entity: dict[str, Any], site_key: str) -> str:
    """Return the linked Wikipedia title for one site. / あるサイトに対応する Wikipedia 記事タイトルを返す。"""

    sitelink = entity.get("sitelinks", {}).get(site_key, {})
    return sitelink.get("title", "") if isinstance(sitelink, dict) else ""


def build_wikipedia_url(language_code: str, title: str) -> str:
    """Convert a Wikipedia title into a readable page URL. / Wikipedia 記事タイトルから閲覧用 URL を作る。"""

    if not title:
        return ""
    return f"https://{language_code}.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='_()')}"


def get_image_names(entity: dict[str, Any]) -> str:
    """Return image file names as a JSON string. / 画像ファイル名一覧を JSON 文字列で返す。"""

    images: list[str] = []
    for claim in get_claims(entity, "P18"):
        if "mainsnak" in claim:
            value = get_nested_value(claim, "mainsnak", "datavalue", "value")
        else:
            value_field = claim.get("value", {})
            value = value_field.get("content", "") if isinstance(value_field, dict) else ""
        if isinstance(value, str):
            images.append(value)
    return json.dumps(images, ensure_ascii=False)


def compute_path(qid: str, parent_map: dict[str, str], root_qid: str) -> str:
    """Build a path from the root taxon to the current entity. / root taxon から現在の entity までのパスを作る。"""

    trail, seen, current = [qid], {qid}, qid
    while current != root_qid:
        parent = parent_map.get(current, "")
        if not parent or parent in seen:
            break
        trail.append(parent)
        seen.add(parent)
        current = parent
    return "/" + "/".join(reversed(trail))


def validate_json_dir(json_dir: Path) -> list[Path]:
    """Return input JSON files, or fail when the directory is unusable. / 入力 JSON 一覧を返し、使えない場合は失敗する。"""

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")
    json_paths = sorted(json_dir.glob("Q*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found under: {json_dir}")
    return json_paths


def collect_entities(json_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all entity JSON files into a QID-keyed dictionary. / すべての entity JSON を QID キーの辞書へまとめる。"""

    entities: dict[str, dict[str, Any]] = {}
    for json_path in validate_json_dir(json_dir):
        entity = load_entity(json_path)
        if qid := entity.get("id", ""):
            entities[qid] = entity
    return entities


def build_parent_map(entities_by_qid: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Map each entity QID to its parent taxon QID. / 各 QID を親 taxon の QID に対応付ける。"""

    return {qid: get_claim_entity_id(entity, "P171") for qid, entity in entities_by_qid.items()}


def build_row(
    qid: str,
    entity: dict[str, Any],
    entities_by_qid: dict[str, dict[str, Any]],
    parent_map: dict[str, str],
    root_qid: str,
) -> dict[str, str]:
    """Build one row for bird_ontology.tsv. / bird_ontology.tsv の 1 行分を作る。"""

    rank_qid = get_claim_entity_id(entity, "P105")
    parent_qid = parent_map.get(qid, "")
    rank_entity = entities_by_qid.get(rank_qid, {})
    parent_entity = entities_by_qid.get(parent_qid, {})
    enwiki_title = get_sitelink_title(entity, "enwiki")
    jawiki_title = get_sitelink_title(entity, "jawiki")
    return {
        "id": qid,
        "entity_url": f"https://www.wikidata.org/entity/{qid}",
        "en_name": get_label(entity, "en"),
        "ja_name": get_label(entity, "ja"),
        "en_aliases": get_aliases(entity, "en"),
        "ja_aliases": get_aliases(entity, "ja"),
        "img_names": get_image_names(entity),
        "xeno_canto_species_id": get_claim_string(entity, "P2426"),
        "taxon_name": get_claim_string(entity, "P225"),
        "taxon_rank": rank_qid,
        "taxon_rank_name": get_label(rank_entity, "en"),
        "taxon_rank_ja_name": get_label(rank_entity, "ja"),
        "parent_taxon": parent_qid,
        "parent_taxon_name": get_label(parent_entity, "en"),
        "parent_taxon_ja_name": get_label(parent_entity, "ja"),
        "enwiki_title": enwiki_title,
        "enwiki_url": build_wikipedia_url("en", enwiki_title),
        "jawiki_title": jawiki_title,
        "jawiki_url": build_wikipedia_url("ja", jawiki_title),
        "path": compute_path(qid, parent_map, root_qid),
    }


def build_ontology(json_dir: Path, output_path: Path, root_qid: str = "Q5113") -> None:
    """Create bird_ontology.tsv from downloaded Wikidata JSON files. / 取得済み Wikidata JSON から bird_ontology.tsv を作る。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    entities_by_qid = collect_entities(json_dir)
    parent_map = build_parent_map(entities_by_qid)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=ONTOLOGY_COLUMNS)
        writer.writeheader()
        for qid in sorted(entities_by_qid):
            writer.writerow(build_row(qid, entities_by_qid[qid], entities_by_qid, parent_map, root_qid))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for ontology generation. / ontology 生成コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build Bird ontology TSV from Wikidata JSON.")
    parser.add_argument("--json-dir", default=str(paths.json_dir))
    parser.add_argument("--output", default=str(paths.ontology_tsv))
    parser.add_argument("--root-qid", default="Q5113")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ontology generation command. / ontology 生成コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    build_ontology(Path(args.json_dir), Path(args.output), root_qid=args.root_qid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
