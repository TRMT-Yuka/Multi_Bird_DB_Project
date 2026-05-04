from __future__ import annotations

import csv
import pickle
import tempfile
import unittest
from pathlib import Path

from multi_bird_db.xeno_canto_ids import extract_xeno_canto_ids


class XenoCantoIdsTest(unittest.TestCase):
    def test_extracts_qid_and_xeno_canto_id_pairs(self) -> None:
        rows = [
            {"qid": "Q1", "xeno_canto_species_id": "XC1"},
            {"id": "Q2", "xeno_canto_species_id": "XC2"},
            {"qid": "Q3", "xeno_canto_species_id": ""},
            {"qid": "", "xeno_canto_species_id": "XC4"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ontology_path = tmp_path / "bird_ontology.pkl"
            output_path = tmp_path / "bird_xeno_canto_ids.tsv"
            with ontology_path.open("wb") as handle:
                pickle.dump(rows, handle)

            extract_xeno_canto_ids(ontology_path, output_path)

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                result = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(
            result,
            [
                {"qid": "Q1", "xeno_canto_species_id": "XC1"},
                {"qid": "Q2", "xeno_canto_species_id": "XC2"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
