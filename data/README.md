# Data Layout

このディレクトリはプロジェクト内でデータを扱うための作業領域です。既存のデータ運用ルールに従い、プロジェクト内で完結する配置にしています。

実行手順そのものは [README_wikidata_pred.md](../README_wikidata_pred.md) を参照してください。このファイルでは主に、`data/` 配下に何が置かれるかを整理します。

## ディレクトリ構成

```text
data/                                     # このプロジェクトで扱う全データのルート
├── raw/                                  # 外部から取得した未加工データ
│   ├── wikidata/                         # Wikidata の入力領域
│   │   ├── query.tsv                     # Bird 配下の entity URL 一覧
│   │   └── dumps/                        # Wikidata JSON dump の保存先
│   └── xeno-canto/                       # Xeno-canto の生音声保存先
├── interim/                              # 処理途中で生成される中間データ
│   ├── wikidata/                         # Wikidata の処理作業領域
│   │   ├── bird_qids.tsv                 # QID 抽出結果
│   │   └── json/                         # QID ごとの JSON 応答
│   └── xeno-canto/                       # Xeno-canto の中間生成物
│       ├── api_recordings/               # API JSON の保存先
│       ├── api_recordings_manifest.json  # API JSON 保存結果
│       └── recording_map.json            # species_id -> recording_id, download_url の対応表
├── processed/                            # 下流利用向けに整形済みの成果物
│   ├── bird_ontology.pkl                 # 鳥類 ontology の構造化成果物
│   ├── graph/                            # graph モダリティの成果物
│   │   ├── bird_taxonomy_graph.pkl       # NetworkX taxonomy graph の構造化成果物
│   │   └── dash/                         # Dash Cytoscape viewer 用の作業領域
│   └── wikipedia_article_manifest.tsv    # Wikipedia 取得対象一覧
└── external/                             # 文書・音声・埋め込みの保管先
    ├── documents/                        # 文書コーパス
    ├── audio/                            # 音声ファイルや音声メタデータ
    ├── sqlite/                           # 軽量な内部参照用 SQLite DB
    └── embeddings/                       # グラフ・テキスト・画像埋め込み
```

## 主なファイル

- `raw/wikidata/query.tsv`
  - 手動で取得して配置する Bird 配下 entity の入力 TSV です
- `raw/wikidata/dumps/latest-all.json.bz2`
  - Wikidata 全量 JSON dump です
- `raw/xeno-canto/<qid>/`
  - Xeno-canto から取得した生音声ファイルです
- `interim/wikidata/bird_qids.tsv`
  - `query.tsv` から抽出した QID 一覧です
- `interim/wikidata/bird_xeno_canto_ids.tsv`
  - ontology から抜き出した `qid` と `xeno_canto_species_id` の対応表です
- `interim/xeno-canto/api_recordings/`
  - Xeno-canto API の JSON 応答を保存する領域です
- `interim/xeno-canto/api_recordings_manifest.json`
  - 保存した API 応答の一覧です
- `interim/xeno-canto/recording_map.json`
  - `xeno_canto_species_id` と `recording_id` / `download_url` の対応表です
- `interim/wikidata/json/Qxxxx.json`
  - dump から切り出した個別 entity JSON です
  - 実際には `interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json` の階層で保存します
- `processed/bird_ontology.pkl`
  - Python でそのまま扱うための構造化成果物です
- `processed/graph/bird_taxonomy_graph.pkl`
  - ontology から生成した `networkx.DiGraph` です
- `processed/graph/dash/`
  - Dash Cytoscape viewer 用の graph モダリティ作業領域です
- `processed/wikipedia_article_manifest.tsv`
  - Wikipedia XML とテキストの保存先を管理する一覧です
- `external/embeddings/graph/node2vec/<MMDDhhmm>/`
  - graph の node2vec 埋め込みです
- `external/embeddings/graph/gcn/<MMDDhhmm>/`
  - graph の GCN 埋め込みです
- `external/embeddings/graph/grace/<MMDDhhmm>/`
  - graph の GRACE 埋め込みです
- `external/embeddings/graph/graphsage/<MMDDhhmm>/`
  - graph の GraphSAGE 埋め込みです
- `external/embeddings/graph/transe/<MMDDhhmm>/`
  - graph の TransE 埋め込みです
- `external/embeddings/audio/<model>/<MMDDhhmm>/`
  - 音声ファイルから作る wav2vec2 系の埋め込みです
- `external/embeddings/graph/evaluation/`
  - graph 埋め込みのクラスタリング評価結果・レポートです
- `external/sqlite/taxonomy/bird_taxonomy.sqlite`
  - taxonomy graph と ontology を引くための軽量 SQLite DB です

生成順やコマンドは [README_wikidata_pred.md](../README_wikidata_pred.md) の「実行」「出力」を参照してください。

## 列説明

### `processed/bird_ontology.pkl`

列定義は [README_wikidata_pred.md](../README_wikidata_pred.md) にもあります。ここではデータ配置の観点で再掲します。

- `qid`
  - Wikidata の QID です
- `entity_url`
  - Wikidata エンティティの URL です
- `en_name`
  - 英語ラベルです
- `ja_name`
  - 日本語ラベルです
- `en_aliases`
  - 英語の別名一覧です
- `ja_aliases`
  - 日本語の別名一覧です
- `img_names`
  - Wikidata に登録されている画像ファイル名一覧です
- `xeno_canto_species_id`
  - Xeno-canto における species ID です
- `taxon_name`
  - 学名です
- `taxon_rank`
  - taxon rank の QID です
- `taxon_rank_name`
  - taxon rank の英語名です
- `taxon_rank_ja_name`
  - taxon rank の日本語名です
- `parent_taxon`
  - 1 つ上の分類群の QID です
- `parent_taxon_name`
  - 1 つ上の分類群の英語名です
- `parent_taxon_ja_name`
  - 1 つ上の分類群の日本語名です
- `enwiki_title`
  - 対応する英語版 Wikipedia 記事タイトルです
- `enwiki_url`
  - 対応する英語版 Wikipedia 記事 URL です
- `jawiki_title`
  - 対応する日本語版 Wikipedia 記事タイトルです
- `jawiki_url`
  - 対応する日本語版 Wikipedia 記事 URL です
- `path`
  - `Q5113` からその QID までの経路です

このファイルは `list[dict]` の `pickle` 形式を想定しています。各 dict のキーは上記の列名と同じです。

### `processed/wikipedia_article_manifest.tsv`

列定義は [README_wikidata_pred.md](../README_wikidata_pred.md) にもあります。ここでは管理用途の観点で再掲します。

- `qid`
  - 対応する Wikidata の QID です
- `language`
  - 記事の言語です
- `title`
  - Wikipedia 記事タイトルです
- `xml_url`
  - XML エクスポート取得用 URL です
- `xml_output_path`
  - XML の保存先です
- `text_output_path`
  - 抽出テキストの保存先です

### `processed/graph/bird_taxonomy_graph.pkl`

詳細な構造は [README_graph.md](../README_graph.md) を参照してください。このファイルは `networkx.DiGraph` の `pickle` を想定しています。

graph metadata:
- `graph.graph["graph_type"]`
  - グラフ種別です
- `graph.graph["root_qid"]`
  - ルート QID です
- `graph.graph["node_count"]`
  - ノード数です
- `graph.graph["edge_count"]`
  - エッジ数です

各ノードの主な属性:
- `label_en`
  - 英語可視化ラベルです
- `label_ja`
  - 日本語ラベルです
- `taxon_rank`
  - taxon rank の QID です
- `taxon_rank_name`
  - taxon rank の英語名です
- `taxon_name`
  - 学名です

各エッジの主な属性:
- `relation`
  - 親子関係の種別です

## 運用

- 主キーは Bird の共通 ID である `bid` または Wikidata の `qid` を基準にする
- 埋め込み本体と属性テーブルは分離する
- 生成条件、次元数、モデル名、生成日時などのメタ情報を残す
- 大容量データは Git に直接入れず、配置先と取得手順だけを記録する
- `interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json` は中間生成物として扱う
- `interim/wikidata/local_dump_extract_checkpoint.json` は個人用の再開状態であり、Git に入れないローカルファイルとして扱う
- `interim/wikidata/bird_xeno_canto_ids.tsv` は `bird_ontology.pkl` から再生成できる中間成果物として扱う
- `interim/xeno-canto/api_recordings/` は Xeno-canto API 応答の中間成果物として扱う
- `interim/xeno-canto/recording_map.json` は API 応答から再生成できる中間成果物として扱う
- `raw/xeno-canto/` は Xeno-canto から取得した生音声の置き場として扱う
- `processed/` は下流利用の成果物を置く
