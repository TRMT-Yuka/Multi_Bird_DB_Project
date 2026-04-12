from __future__ import annotations

import pickle
from pathlib import Path

from multi_bird_db.config import get_project_paths
from multi_bird_db.graph import build_taxonomy_graph


def make_row(
    qid: str,
    en_name: str,
    taxon_name: str,
    taxon_rank_name: str,
    parent_taxon: str = "",
    ja_name: str = "",
) -> dict[str, str]:
    return {
        "id": qid,
        "en_name": en_name,
        "ja_name": ja_name,
        "taxon_name": taxon_name,
        "taxon_rank": f"{qid}-rank",
        "taxon_rank_name": taxon_rank_name,
        "parent_taxon": parent_taxon,
        "entity_url": f"https://www.wikidata.org/wiki/{qid}",
        "enwiki_url": f"https://en.wikipedia.org/wiki/{en_name.replace(' ', '_')}",
        "jawiki_url": "",
    }


def build_dummy_rows() -> list[dict[str, str]]:
    root = "QD1000"
    return [
        make_row(root, "Aves", "Aves", "class"),
        make_row("QD1100", "Passeriformes", "Passeriformes", "order", root),
        make_row("QD1110", "Corvidae", "Corvidae", "family", "QD1100"),
        make_row("QD1111", "Corvus", "Corvus", "genus", "QD1110"),
        make_row("QD1112", "Common raven", "Corvus corax", "species", "QD1111"),
        make_row("QD1113", "Carrion crow", "Corvus corone", "species", "QD1111"),
        make_row("QD1120", "Paridae", "Paridae", "family", "QD1100"),
        make_row("QD1121", "Parus", "Parus", "genus", "QD1120"),
        make_row("QD1122", "Great tit", "Parus major", "species", "QD1121"),
        make_row("QD1200", "Accipitriformes", "Accipitriformes", "order", root),
        make_row("QD1210", "Accipitridae", "Accipitridae", "family", "QD1200"),
        make_row("QD1211", "Haliaeetus", "Haliaeetus", "genus", "QD1210"),
        make_row("QD1212", "Bald eagle", "Haliaeetus leucocephalus", "species", "QD1211"),
        make_row("QD1213", "White-tailed eagle", "Haliaeetus albicilla", "species", "QD1211"),
        make_row("QD1300", "Sphenisciformes", "Sphenisciformes", "order", root),
        make_row("QD1310", "Spheniscidae", "Spheniscidae", "family", "QD1300"),
        make_row("QD1311", "Aptenodytes", "Aptenodytes", "genus", "QD1310"),
        make_row("QD1312", "Emperor penguin", "Aptenodytes forsteri", "species", "QD1311"),
        make_row("QD1313", "King penguin", "Aptenodytes patagonicus", "species", "QD1311"),
        make_row("QD1400", "Neoaves clade", "Neoaves", "other", root),
        make_row("QD1410", "Experimental lineage", "Linea experimentalis", "other", "QD1400"),
    ]


def main() -> int:
    paths = get_project_paths()
    output_path = paths.graph_dir / "bird_taxonomy_graph_dummy.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = build_taxonomy_graph(build_dummy_rows(), root_qid="QD1000")
    with output_path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(output_path)
    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
