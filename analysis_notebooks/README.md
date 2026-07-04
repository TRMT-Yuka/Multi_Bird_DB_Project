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

- `taxon_rank_label_assignment_distribution.py`
  - Treats each node at a given `taxon_rank` as a label.
  - Counts how many lower nodes each label would cover.
  - Writes detail / summary CSV files.


CSV columns:

- `taxon_rank_label_assignment_detail.csv`
  - `label_rank_qid`: そのラベル自身の taxon rank の QID
  - `label_rank_name`: そのラベル自身の taxon rank 名
  - `label_qid`: ラベルとして使うノードの QID
  - `label_name`: ラベルとして使うノード名
  - `assigned_node_count`: そのラベルの下位にあるノード数
  - `assigned_leaf_node_count`: そのラベルの下位にある葉ノード数
  - `strict_descendant_count`: 下位ノード数（`assigned_node_count` と同じ）
  - `leaf_descendant_count`: 下位の葉ノード数（`assigned_leaf_node_count` と同じ）
  - `is_leaf_label`: そのラベル自身が葉ノードなら `1`、そうでなければ `0`

- `taxon_rank_label_assignment_summary.csv`
  - `label_rank_qid`: 集計対象 rank の QID
  - `label_rank_name`: 集計対象 rank 名
  - `label_count`: その rank に属するラベル数
  - `assigned_node_count_mean`: 1 ラベルあたり下位ノード数の平均
  - `assigned_node_count_median`: 1 ラベルあたり下位ノード数の中央値
  - `assigned_node_count_max`: 1 ラベルあたり下位ノード数の最大値
  - `assigned_node_count_min`: 1 ラベルあたり下位ノード数の最小値
  - `assigned_leaf_node_count_mean`: 1 ラベルあたり下位葉ノード数の平均
  - `assigned_leaf_node_count_median`: 1 ラベルあたり下位葉ノード数の中央値
  - `assigned_leaf_node_count_max`: 1 ラベルあたり下位葉ノード数の最大値
  - `zero_assigned_label_count`: 下位ノードを 1 つも持たないラベル数

Label examples by `label_rank_name`:

以下は今回の出力からそのまま拾った具体例です。網羅ではありません。日本語訳は、一般的な和名があるものは和名を、そうでないものは便宜的な訳や学名のカタカナ転写を付けています。

- `class`
  - `bird` : 鳥類

- `subclass`
  - `Neornithes` : 現生鳥類
  - `Neognathae` : 新顎類
  - `Passerae` : スズメ類

- `infraclass`
  - `Palaeognathae` : 古顎類
  - `Odontotormae` : オドントルマエ

- `superorder`
  - `Neoaves` : 新鳥類
  - `Aequornithes` : 水鳥類
  - `Psittacimorphae` : オウム類

- `order`
  - `passerines` : スズメ目
  - `Apodiformes` : アマツバメ目
  - `Piciformes` : キツツキ目

- `suborder`
  - `songbirds` : 鳴禽類
  - `Tyranni` : タイラン亜目
  - `Lari` : カモメ亜目

- `infraorder`
  - `Meliphagida` : メリファギダ群
  - `Climacterida` : クリマクテリダ群
  - `Orthonychida` : オルトニキダ群

- `parvorder`
  - `Passerida` : スズメ小目
  - `Tyrannida` : タイラン小目
  - `Corvida` : カラス小目

- `superfamily`
  - `Passeroidea` : スズメ上科
  - `Muscicapoidea` : ヒタキ上科
  - `Sylvioidea` : ムシクイ上科

- `family`
  - `Muscicapidae` : ヒタキ科
  - `Tyrannidae` : タイランチョウ科
  - `Thraupidae` : フウキンチョウ科

- `subfamily`
  - `Trochilinae` : ハチドリ亜科
  - `Carduelinae` : アトリ亜科
  - `Picinae` : キツツキ亜科

- `tribe`
  - `swiftlet` : アナツバメ類
  - `Copsychini` : シキチョウ族
  - `Anserini` : ハクチョウ族

- `genus`
  - `Turdus` : ツグミ属
  - `Zosterops` : メジロ属
  - `Cisticola` : セッカ属

- `subgenus`
  - `hobby` : チゴハヤブサ類
  - `Chloris` : カワラヒワ類
  - `Anthus` : タヒバリ類

- `species`
  - `Turdus poliocephalus` : ムナフジツグミ
  - `Horned Lark` : ミミヒバリ
  - `Bananaquit` : バナナクイト

- `subspecies`
  - `Tyto alba punctatissima` : メンフクロウの `punctatissima` 亜種
  - `Fulica atra pontica` : オオバンの `pontica` 亜種
  - `Pampusana beccarii beccarii` : `Pampusana beccarii` の基亜種

- `form`
  - `Cepphus grylle f. mandtii` : ウミバトの `mandtii` 型
  - `Chen caerulescens f. atlantica` : ハクガンの `atlantica` 型
  - `Ardea alba f. modesta` : ダイサギの `modesta` 型

- `ichnogenus`
  - `Archaeornithipus` : アルカエオルニティプス（足跡化石属）

- `unknown`
  - `Q3239179` : 名称未解決 QID の例
  - `Q135105005` : 名称未解決 QID の例
  - `Q104864059` : 名称未解決 QID の例

