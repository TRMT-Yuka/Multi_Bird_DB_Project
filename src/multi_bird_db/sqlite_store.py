from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import get_project_paths


def load_ontology(ontology_path: Path) -> list[dict[str, Any]]:
    """Load ontology PKL rows. / ontology PKL の行一覧を読む。"""

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file does not exist: {ontology_path}")
    with ontology_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Ontology PKL must contain a list, got: {type(data).__name__}")
    return data


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nodes (
            qid TEXT PRIMARY KEY,
            entity_url TEXT NOT NULL,
            en_name TEXT NOT NULL,
            ja_name TEXT NOT NULL,
            en_aliases_json TEXT NOT NULL,
            ja_aliases_json TEXT NOT NULL,
            img_names_json TEXT NOT NULL,
            xeno_canto_species_id TEXT NOT NULL,
            taxon_name TEXT NOT NULL,
            taxon_rank TEXT NOT NULL,
            taxon_rank_name TEXT NOT NULL,
            taxon_rank_ja_name TEXT NOT NULL,
            parent_taxon TEXT NOT NULL,
            parent_taxon_name TEXT NOT NULL,
            parent_taxon_ja_name TEXT NOT NULL,
            enwiki_title TEXT NOT NULL,
            enwiki_url TEXT NOT NULL,
            jawiki_title TEXT NOT NULL,
            jawiki_url TEXT NOT NULL,
            path TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS edges (
            parent_qid TEXT NOT NULL,
            child_qid TEXT NOT NULL,
            relation TEXT NOT NULL,
            PRIMARY KEY (parent_qid, child_qid),
            FOREIGN KEY (parent_qid) REFERENCES nodes(qid) ON DELETE CASCADE,
            FOREIGN KEY (child_qid) REFERENCES nodes(qid) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_parent_taxon ON nodes(parent_taxon);
        CREATE INDEX IF NOT EXISTS idx_nodes_taxon_rank_name ON nodes(taxon_rank_name);
        CREATE INDEX IF NOT EXISTS idx_nodes_taxon_name ON nodes(taxon_name);
        CREATE INDEX IF NOT EXISTS idx_edges_parent_qid ON edges(parent_qid);
        CREATE INDEX IF NOT EXISTS idx_edges_child_qid ON edges(child_qid);
        """
    )


def _insert_metadata(conn: sqlite3.Connection, entries: dict[str, str]) -> None:
    conn.executemany("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", entries.items())


def _insert_nodes(conn: sqlite3.Connection, ontology_rows: Iterable[dict[str, Any]]) -> int:
    rows = []
    for row in ontology_rows:
        qid = str(row.get("qid") or row.get("id") or "").strip()
        if not qid:
            continue
        rows.append(
            (
                qid,
                str(row.get("entity_url") or "").strip(),
                str(row.get("en_name") or "").strip(),
                str(row.get("ja_name") or "").strip(),
                _json_text(row.get("en_aliases") or []),
                _json_text(row.get("ja_aliases") or []),
                _json_text(row.get("img_names") or []),
                str(row.get("xeno_canto_species_id") or "").strip(),
                str(row.get("taxon_name") or "").strip(),
                str(row.get("taxon_rank") or "").strip(),
                str(row.get("taxon_rank_name") or "").strip(),
                str(row.get("taxon_rank_ja_name") or "").strip(),
                str(row.get("parent_taxon") or "").strip(),
                str(row.get("parent_taxon_name") or "").strip(),
                str(row.get("parent_taxon_ja_name") or "").strip(),
                str(row.get("enwiki_title") or "").strip(),
                str(row.get("enwiki_url") or "").strip(),
                str(row.get("jawiki_title") or "").strip(),
                str(row.get("jawiki_url") or "").strip(),
                str(row.get("path") or "").strip(),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO nodes(
            qid, entity_url, en_name, ja_name, en_aliases_json, ja_aliases_json, img_names_json,
            xeno_canto_species_id, taxon_name, taxon_rank, taxon_rank_name, taxon_rank_ja_name,
            parent_taxon, parent_taxon_name, parent_taxon_ja_name, enwiki_title, enwiki_url,
            jawiki_title, jawiki_url, path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _insert_edges(conn: sqlite3.Connection, ontology_rows: Iterable[dict[str, Any]]) -> int:
    rows = []
    for row in ontology_rows:
        child_qid = str(row.get("qid") or row.get("id") or "").strip()
        parent_qid = str(row.get("parent_taxon") or "").strip()
        if not child_qid or not parent_qid:
            continue
        rows.append((parent_qid, child_qid, "parent_taxon"))
    conn.executemany(
        "INSERT OR REPLACE INTO edges(parent_qid, child_qid, relation) VALUES (?, ?, ?)",
        rows,
    )
    return len(rows)


def build_taxonomy_sqlite(ontology_rows: list[dict[str, Any]], db_path: Path, root_qid: str = "Q5113") -> None:
    """Create a small SQLite store from ontology rows. / ontology 行から小さな SQLite ストアを作る。"""

    with _connect(db_path) as conn:
        _initialize_schema(conn)
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM nodes")
        inserted_nodes = _insert_nodes(conn, ontology_rows)
        inserted_edges = _insert_edges(conn, ontology_rows)
        _insert_metadata(
            conn,
            {
                "format_version": "1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "root_qid": root_qid,
                "node_count": str(inserted_nodes),
                "edge_count": str(inserted_edges),
            },
        )
        conn.commit()


@dataclass(slots=True)
class TaxonomySQLiteStore:
    """Read-only helper for qid lookups in the taxonomy SQLite DB. / taxonomy SQLite DB の参照ヘルパー。"""

    connection: sqlite3.Connection

    @classmethod
    def open(cls, db_path: Path) -> "TaxonomySQLiteStore":
        conn = _connect(db_path)
        return cls(connection=conn)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TaxonomySQLiteStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_node(self, qid: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM nodes WHERE qid = ?", (qid,)).fetchone()
        return dict(row) if row is not None else None

    def get_children(self, parent_qid: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT n.*
            FROM edges e
            JOIN nodes n ON n.qid = e.child_qid
            WHERE e.parent_qid = ?
            ORDER BY n.qid
            """,
            (parent_qid,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_metadata(self) -> dict[str, str]:
        rows = self.connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def get_descendants(self, qid: str, max_depth: int | None = None) -> list[dict[str, Any]]:
        depth_filter = "" if max_depth is None else "WHERE tree.depth < ?"
        params: tuple[Any, ...] = (qid,) if max_depth is None else (qid, max_depth)
        rows = self.connection.execute(
            f"""
            WITH RECURSIVE tree(qid, depth) AS (
                SELECT ? AS qid, 0 AS depth
                UNION ALL
                SELECT e.child_qid, tree.depth + 1
                FROM tree
                JOIN edges e ON e.parent_qid = tree.qid
                {depth_filter}
            )
            SELECT n.*, tree.depth
            FROM tree
            JOIN nodes n ON n.qid = tree.qid
            ORDER BY tree.depth, n.qid
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for SQLite export. / SQLite 出力コマンド用の引数パーサを作る。"""

    paths = get_project_paths()
    parser = argparse.ArgumentParser(description="Build a taxonomy SQLite DB from ontology PKL.")
    parser.add_argument("--input", default=str(paths.ontology_pkl))
    parser.add_argument("--output", default=str(paths.taxonomy_sqlite_path))
    parser.add_argument("--root-qid", default="Q5113")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the SQLite export command. / SQLite 出力コマンドを実行する。"""

    args = build_parser().parse_args(argv)
    build_taxonomy_sqlite(load_ontology(Path(args.input)), Path(args.output), root_qid=args.root_qid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
