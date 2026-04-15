from __future__ import annotations

import bz2
import json
import tempfile
import unittest
from pathlib import Path

from multi_bird_db.dump_extract import extract_entities_from_dump, qid_to_json_path


def write_dump(path: Path, entities: list[dict]) -> None:
    """Write a minimal Wikidata dump file for tests. / テスト用の最小 Wikidata dump を書く。"""

    with bz2.open(path, mode="wt", encoding="utf-8") as handle:
        handle.write("[\n")
        handle.write(",\n".join(json.dumps(entity, ensure_ascii=False) for entity in entities))
        handle.write("\n]\n")


class DumpExtractTests(unittest.TestCase):
    def test_extract_entities_from_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "qids.tsv"
            dump_path = tmp_path / "latest-all.json.bz2"
            output_dir = tmp_path / "json"
            checkpoint_path = tmp_path / "dump_extract_checkpoint.json"

            qids = ["Q123", "Q456", "Q789"]
            input_path.write_text("qid\n" + "\n".join(qids) + "\n", encoding="utf-8")
            entities = [{"id": qid, "type": "item", "labels": {"en": {"value": qid}}} for qid in qids]
            write_dump(dump_path, entities)

            q456_path = qid_to_json_path(output_dir, "Q456")
            q456_path.parent.mkdir(parents=True, exist_ok=True)
            q456_path.write_text(json.dumps({"id": "Q456", "cached": True}), encoding="utf-8")

            written = extract_entities_from_dump(input_path, dump_path, output_dir, checkpoint_path)

            self.assertEqual(written, 2)
            self.assertEqual(json.loads(qid_to_json_path(output_dir, "Q123").read_text(encoding="utf-8"))["id"], "Q123")
            self.assertEqual(json.loads(qid_to_json_path(output_dir, "Q789").read_text(encoding="utf-8"))["id"], "Q789")
            self.assertEqual(json.loads(q456_path.read_text(encoding="utf-8"))["cached"], True)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(checkpoint["completed_qids"]), qids)


if __name__ == "__main__":
    unittest.main()
