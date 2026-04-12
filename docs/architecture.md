# Architecture

## 実行の流れ

1. `data/raw/wikidata/query.tsv` を用意する
2. `make extract-qids` で QID 一覧を作る
3. `make download-wikidata-dump` で Wikidata dump を取得する
4. `make extract-dump-json` で対象 QID の JSON を切り出す
5. `make build-ontology` で ontology PKL を作る
6. `make build-graph` で taxonomy graph PKL を作る
7. `make visualize-graph` で taxonomy 部分グラフ PNG を作る
8. `make serve-graph` で taxonomy graph viewer を起動する
9. `make build-wikipedia-manifest` で Wikipedia 記事一覧 TSV を作る
10. `make fetch-wikipedia-xml` で英語版・日本語版記事 XML を保存する
11. `make extract-wikipedia-text` で XML からテキストを抽出する

## ディレクトリ構成

```text
Multi_Bird_DB_Project/                    # プロジェクト全体のルート
├── data/                                # 入力・中間生成物・成果物・外部データの置き場
│   ├── raw/                             # 外部から持ち込んだ未加工データ
│   ├── interim/                         # 実行途中で再生成できるファイル
│   ├── processed/                       # 最終成果物に近い整形済みデータ
│   └── external/                        # モダリティ別に管理する外部データ
│       ├── documents/                   # Wikipedia などの文書データ
│       ├── audio/                       # 音声ファイルや音声メタデータ
│       └── embeddings/                  # グラフ・文書・画像の埋め込み
├── docs/                                # 設計説明と作業メモ
├── queries/                             # データ取得元の SPARQL
└── src/multi_bird_db/                   # 実装本体
```

## データ設計

```text
data/                                     # このプロジェクトのデータ管理ルート
├── raw/                                  # 外部取得直後の元データを保持
│   └── wikidata/                         # Wikidata 入力専用領域
│       ├── query.tsv                     # Bird 配下 entity の入力 TSV
│       └── dumps/                        # Wikidata JSON dump の保存先
├── interim/                              # 再作成可能な中間生成物を保持
│   └── wikidata/                         # Wikidata 処理中の作業領域
│       ├── bird_qids.tsv                 # QID 抽出結果
│       └── json/                         # QID ごとの entity JSON
├── processed/                            # 下流処理が直接参照する整形済み成果物
│   ├── bird_ontology.pkl                 # 鳥類 ontology の構造化成果物
│   ├── graph/
│   │   ├── bird_taxonomy_graph.pkl       # 鳥類 taxonomy の NetworkX graph
│   │   └── figures/
│   │       └── bird_taxonomy_graph.png   # taxonomy 部分グラフの可視化
│   └── wikipedia_article_manifest.tsv    # Wikipedia 記事取得対象一覧
└── external/                             # 補助データと別モダリティの保管先
    ├── documents/                        # XML、抽出本文、将来の文書コーパス
    ├── audio/                            # 鳴き声音声と付随メタデータ
    └── embeddings/                       # ベクトル本体とそのメタ情報
```

## ファイルの役割

- `README.md`
  - 最初に読むファイル
- `Makefile`
  - 実行手順を短いコマンドにまとめた入口
- `pyproject.toml`
  - Python パッケージ設定
- `queries/bird_descendants.sparql`
  - Bird (`Q5113`) 配下を取得する元クエリ
- `src/multi_bird_db/config.py`
  - パス設定
- `src/multi_bird_db/cli.py`
  - CLI 本体
- `src/multi_bird_db/qids.py`
  - QID 抽出処理
- `src/multi_bird_db/dump_extract.py`
  - dump から必要な QID の JSON を切り出す処理
- `src/multi_bird_db/ontology.py`
  - JSON から ontology PKL を生成する処理
- `src/multi_bird_db/graph.py`
  - ontology PKL から taxonomy の NetworkX graph PKL を生成する処理
- `src/multi_bird_db/graph_visualization.py`
  - graph PKL から部分グラフ PNG を描画する処理
- `src/multi_bird_db/graph_dash.py`
  - graph PKL から Dash Cytoscape viewer を起動する処理
- `src/multi_bird_db/wikipedia_articles.py`
  - Wikipedia 記事一覧 TSV 生成、XML 取得、テキスト抽出の処理

## 現在の責務

- `raw/wikidata/query.tsv` を入力にして `Q5113` 配下の QID を抽出する
- Wikidata dump から必要な QID の entity JSON を切り出す
- 取得済み JSON から ontology PKL を生成する
- ontology PKL から taxonomy graph PKL を生成する

## 将来の拡張

- 文書データ
  - 鳥類説明文、論文、観察記録の取り込み
- 音声データ
  - 鳴き声音声と対応する種 ID の接続
- 画像データ
  - 画像 URL、ローカル画像、特徴量ベクトルの管理
- 埋め込み
  - Wikidata ベースのグラフ埋め込み
  - 日本語・英語文書埋め込み
  - 画像埋め込み

## 設計方針

- 実行入口は CLI に集約する
- パスはコード内で集中管理する
- 生成物は `data/interim` と `data/processed` に分離する
- 将来の modality 追加時も `qid` を中心に紐付ける
