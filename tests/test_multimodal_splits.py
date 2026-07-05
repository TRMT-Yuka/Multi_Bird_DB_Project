from __future__ import annotations

import unittest

from multi_bird_db.multimodal.splits import qid_split_lookup, split_qids


class MultimodalSplitTests(unittest.TestCase):
    def test_split_qids_is_deterministic_for_same_seed(self) -> None:
        qids = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
        left = split_qids(qids, validation_fraction=0.2, test_fraction=0.2, seed=7)
        right = split_qids(qids, validation_fraction=0.2, test_fraction=0.2, seed=7)
        self.assertEqual(left, right)

    def test_qid_split_lookup_marks_all_memberships(self) -> None:
        split = split_qids(['Q1', 'Q2', 'Q3', 'Q4'], validation_fraction=0.25, test_fraction=0.25, seed=1)
        lookup = qid_split_lookup(split)
        self.assertEqual(set(lookup.values()), {'train', 'validation', 'test'})
        self.assertEqual(set(lookup.keys()), {'Q1', 'Q2', 'Q3', 'Q4'})
