# Analysis Notebooks

This directory stores ad hoc analysis scripts used for one-off checks and reproducible inspection of the project data.

Current scripts:

- `qid_leaf_analysis.py`
  - Counts leaf QIDs in the taxonomy graph.
  - Counts leaf QIDs that have `xeno_canto_species_id` in `bird_ontology.pkl`.


  <!-- 中身:

  - 末端 QID 数の計算
  - 末端のうち xeno_canto_species_id を持つ件数の計算

  実行確認:

  - leaf_qid_count      40173
  - leaf_qid_with_xeno_canto_species_id_count   5573 -->