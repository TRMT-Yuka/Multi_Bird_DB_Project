from __future__ import annotations

import unittest

import networkx as nx

from multi_bird_db.multimodal.labels import assign_upper_taxon_label, iter_ancestor_chain


class MultimodalLabelTests(unittest.TestCase):
    def test_iter_ancestor_chain_follows_parent_taxon(self) -> None:
        graph = nx.DiGraph()
        graph.add_node('QFAMILY', parent_taxon='', taxon_rank_name='family', label_en='Family A')
        graph.add_node('QGENUS', parent_taxon='QFAMILY', taxon_rank_name='genus', label_en='Genus A')
        graph.add_node('QSPECIES', parent_taxon='QGENUS', taxon_rank_name='species', label_en='Species A')

        self.assertEqual(iter_ancestor_chain(graph, 'QSPECIES'), ['QGENUS', 'QFAMILY'])

    def test_assign_upper_taxon_label_returns_matching_ancestor(self) -> None:
        graph = nx.DiGraph()
        graph.add_node('QFAMILY', parent_taxon='', taxon_rank_name='family', label_en='Family A')
        graph.add_node('QGENUS', parent_taxon='QFAMILY', taxon_rank_name='genus', label_en='Genus A')
        graph.add_node('QSPECIES', parent_taxon='QGENUS', taxon_rank_name='species', label_en='Species A')

        assignment = assign_upper_taxon_label(graph, 'QSPECIES', 'family')

        assert assignment is not None
        self.assertEqual(assignment.label_qid, 'QFAMILY')
        self.assertEqual(assignment.label_name, 'Family A')
        self.assertEqual(assignment.distance_to_label, 2)
