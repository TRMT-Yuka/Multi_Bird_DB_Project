# Multi_Bird_DB_Project

Bird (`Q5113`) 配下の Wikidata エンティティを再現的に取得し、将来的な文書・音声・画像・グラフ埋め込みを同一プロジェクト内で扱えるように整理した初期コミット用の雛形です。`Bird_ontology_Project` の知見を引き継ぎつつ、固定列挙の取得コードから、再実行しやすい CLI ベースのパイプラインへ置き換えています。

## 概要

- Bird (`Q5113`) 配下の QID 一覧を再取得可能にする
- Wikidata entity JSON を保存し、ontology TSV を再生成できるようにする
- 今後追加する文書、音声、画像、埋め込みデータを置く場所を先に定義する
- Git で追跡すべき設定・コードと、生成物を明確に分離する

## 最初に読む場所

- [docs/architecture.md](docs/architecture.md)
  - ディレクトリ構成とファイルの役割をまとめています
- [data/README.md](data/README.md)
  - データの置き場所と運用方針をまとめています
- [docs/work_notes/2026-04-11_initial_scaffold.md](docs/work_notes/2026-04-11_initial_scaffold.md)
  - 作業時の判断メモです

## 参考資料

- inaturam, 「自然言語処理のためのWikipediaテキストデータ抽出」
  - https://zenn.dev/inaturam/articles/8c4705fed8f18a
  - Wikipedia 記事を XML として扱い、そこからテキストを抽出する方針を参考にしています

## ディレクトリ構成

```text
Multi_Bird_DB_Project/                    # プロジェクト全体のルート
├── README.md                            # 全体概要、実行手順、設計の入口
├── Makefile                             # 主要 CLI を短いコマンドで呼び出す入口
├── pyproject.toml                       # Python パッケージ設定と依存関係定義
├── .gitignore                           # 生成物や大容量データの除外設定
├── data/                                # 入力・中間生成物・成果物・外部モダリティの保管先
│   ├── README.md                        # data/ 配下の運用ルール
│   ├── raw/                             # 外部から取得・配置した元データ
│   │   └── wikidata/                    # Wikidata 由来の生入力
│   │       └── query.tsv                # Bird 配下エンティティ取得結果の入力 TSV
│   ├── interim/                         # 再生成可能な途中生成物
│   ├── processed/                       # 下流利用向けに整形した成果物
│   └── external/                        # 文書・音声・画像・埋め込みなどの外部データ置き場
│       ├── audio/                       # 鳴き声音声とそのメタデータ
│       ├── documents/                   # Wikipedia などの文書コーパス
│       ├── embeddings/                  # テキスト・画像・グラフの埋め込み
│       └── images/                      # 画像ファイルと画像メタデータ
├── docs/                                # 人が読む設計資料と作業メモ
│   ├── architecture.md                  # 構成と処理フローの説明
│   └── work_notes/                      # 作業時の判断ログ
├── queries/                             # SPARQL などの取得クエリ置き場
│   └── bird_descendants.sparql          # Bird (Q5113) 配下抽出用クエリ
└── src/                                 # 実装コードのルート
    └── multi_bird_db/                   # bird DB 構築用パッケージ本体
        ├── __init__.py                  # パッケージ初期化
        ├── cli.py                       # 実行コマンドの統合入口
        ├── config.py                    # パスや定数の集中管理
        ├── fetch_script.py              # JSON 取得用シェルスクリプト生成
        ├── ontology.py                  # Wikidata JSON から ontology TSV を生成
        ├── qids.py                      # query.tsv から QID 一覧を抽出
        └── wikipedia_articles.py        # Wikipedia 記事一覧生成、XML 取得、本文抽出
```

## データ設計

```text
data/                                     # このプロジェクトで扱う全データのルート
├── raw/                                  # 外部から取得した未加工データ
│   └── wikidata/                         # Wikidata の入力領域
│       └── query.tsv                     # Bird 配下の entity URL 一覧
├── interim/                              # 処理途中で生成される中間データ
│   └── wikidata/                         # Wikidata の処理作業領域
│       ├── bird_qids.tsv                 # QID 抽出結果
│       ├── fetch_entities.sh             # entity JSON 取得用スクリプト
│       └── json/                         # QID ごとの JSON 応答
├── processed/                            # 下流利用向けに整形済みの成果物
│   ├── bird_ontology.tsv                 # 鳥類 ontology 統合表
│   └── wikipedia_article_manifest.tsv    # Wikipedia 取得対象一覧
└── external/                             # 文書・音声・画像・埋め込みの保管先
    ├── documents/                        # 文書コーパス
    ├── audio/                            # 音声ファイルや音声メタデータ
    ├── images/                           # 画像と画像メタデータ
    └── embeddings/                       # グラフ・テキスト・画像埋め込み
```

既存のデータ取り扱いルールに従い、このプロジェクト内で完結するデータ配置にしています。詳細は [data/README.md](data/README.md) と [docs/work_notes/2026-04-11_initial_scaffold.md](docs/work_notes/2026-04-11_initial_scaffold.md) に残しています。

## 実行

この節は 2 段に分かれています。

- 上段
  - 目的ごとに使うコマンドをすぐ確認するための早見表です
- 下段
  - `### 0` 以降の番号付き手順です。実際に上から順に実行する場合はこちらを見ます

### 実行コマンドの早見表

- QID 一覧だけ欲しい場合
  - `make extract-qids`
- QID 一覧と JSON 取得スクリプトまで欲しい場合
  - `make bootstrap`
- QID 一覧、JSON 取得スクリプト、Wikidata JSON 本体まで欲しい場合
  - `make extract-qids`
  - `make generate-fetch`
  - `bash data/interim/wikidata/fetch_entities.sh`
- 最終的に ontology TSV を作りたい場合
  - `make build-ontology`
- Wikipedia 記事一覧を作りたい場合
  - `make build-wikipedia-manifest`
- Wikipedia 記事 XML を保存したい場合
  - `make fetch-wikipedia-xml`
- Wikipedia XML からテキストを保存したい場合
  - `make extract-wikipedia-text`
- コードの構文確認だけしたい場合
  - `make verify`

Wikipedia については、参考資料と同様に「最初から本文テキストだけを直接取る」のではなく、先に記事 XML を保存し、その後で必要な文を取り出す流れを採用しています。

### 詳細手順

### 0. 作業ディレクトリへ移動する

```bash
cd Multi_Bird_DB_Project
```

### 1. 入力ファイルを確認する

入力は [data/raw/wikidata/query.tsv](data/raw/wikidata/query.tsv) です。

### 2. QID 一覧を作る

```bash
make extract-qids
```

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/qids.py](src/multi_bird_db/qids.py)

生成物:
[data/interim/wikidata/bird_qids.tsv](data/interim/wikidata/bird_qids.tsv)

新しく作られるディレクトリ:
- `data/interim/wikidata/`

### 3. JSON 取得用スクリプトを作る

```bash
make generate-fetch
```

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/fetch_script.py](src/multi_bird_db/fetch_script.py)

生成物:
[data/interim/wikidata/fetch_entities.sh](data/interim/wikidata/fetch_entities.sh)

QID 一覧と取得スクリプトをまとめて作る場合は次だけでも構いません。

```bash
make bootstrap
```

### 4. Wikidata から JSON を取得する

このステップを実行する前に、`make generate-fetch` か `make bootstrap` を実行して
[data/interim/wikidata/fetch_entities.sh](data/interim/wikidata/fetch_entities.sh)
を生成しておく必要があります。

```bash
bash data/interim/wikidata/fetch_entities.sh
```

実行される処理:
- [data/interim/wikidata/fetch_entities.sh](data/interim/wikidata/fetch_entities.sh)

生成先:
`data/interim/wikidata/json/`

新しく作られるファイル:
- `data/interim/wikidata/json/Qxxxx.json`
  - 各 QID に対応する Wikidata entity JSON

### 5. ontology TSV を作る

```bash
make build-ontology
```

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/ontology.py](src/multi_bird_db/ontology.py)

`build-ontology` は `data/interim/wikidata/json/` に JSON が存在しない場合は失敗するようにしています。空の TSV を誤って生成しないためです。

生成物:
[data/processed/bird_ontology.tsv](data/processed/bird_ontology.tsv)

### 6. Wikipedia 記事一覧を作る

```bash
make build-wikipedia-manifest
```

この段階では、Wikipedia 記事そのものはまだ取得しません。XML と抽出テキストの保存先を整理した TSV を作ります。

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/wikipedia_articles.py](src/multi_bird_db/wikipedia_articles.py)

生成物:
[data/processed/wikipedia_article_manifest.tsv](data/processed/wikipedia_article_manifest.tsv)

新しく作られるディレクトリ:
- `data/external/documents/wikipedia/xml/en/`
- `data/external/documents/wikipedia/xml/ja/`
- `data/external/documents/wikipedia/text/en/`
- `data/external/documents/wikipedia/text/ja/`

### 7. 英語版・日本語版 Wikipedia 記事 XML を取得する

```bash
make fetch-wikipedia-xml
```

この段階で、Wikipedia の `Special:Export` を使って記事 XML を保存します。

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/wikipedia_articles.py](src/multi_bird_db/wikipedia_articles.py)

生成先:
- `data/external/documents/wikipedia/xml/en/`
- `data/external/documents/wikipedia/xml/ja/`

新しく作られるファイル:
- `data/external/documents/wikipedia/xml/en/Qxxxx.xml`
  - 英語版 Wikipedia 記事 XML
- `data/external/documents/wikipedia/xml/ja/Qxxxx.xml`
  - 日本語版 Wikipedia 記事 XML

### 8. Wikipedia XML からテキストを抽出する

```bash
make extract-wikipedia-text
```

この段階で、保存済み XML から本文に相当するテキストを取り出します。

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/wikipedia_articles.py](src/multi_bird_db/wikipedia_articles.py)

生成先:
- `data/external/documents/wikipedia/text/en/`
- `data/external/documents/wikipedia/text/ja/`

新しく作られるファイル:
- `data/external/documents/wikipedia/text/en/Qxxxx.txt`
  - 英語版 Wikipedia XML から抽出したテキスト
- `data/external/documents/wikipedia/text/ja/Qxxxx.txt`
  - 日本語版 Wikipedia XML から抽出したテキスト

### 9. コード確認だけ行う

```bash
make verify
```

実行される処理:
- Python ファイルの構文確認

新しく作られる可能性があるもの:
- `__pycache__/`
  - Python のキャッシュ

`Makefile` は `PYTHONPATH=src` を付与しているため、初期状態では追加インストールなしで実行できます。パッケージとして扱いたい場合は `pip install -e .` でも構いません。

## 出力

### 何を実行した結果、何が入力とされ、何が出力されるか

- `make extract-qids`
  - 入力: `data/raw/wikidata/query.tsv`
  - 出力: `data/interim/wikidata/bird_qids.tsv`
- `make generate-fetch`
  - 入力: `data/interim/wikidata/bird_qids.tsv`
  - 出力: `data/interim/wikidata/fetch_entities.sh`
- `bash data/interim/wikidata/fetch_entities.sh`
  - 入力: `make generate-fetch` または `make bootstrap` により生成された `data/interim/wikidata/fetch_entities.sh`
  - 出力: `data/interim/wikidata/json/Qxxxx.json`
- `make build-ontology`
  - 入力: `data/interim/wikidata/json/Qxxxx.json`
  - 出力: `data/processed/bird_ontology.tsv`
- `make build-wikipedia-manifest`
  - 入力: `data/interim/wikidata/json/Qxxxx.json`
  - 出力: `data/processed/wikipedia_article_manifest.tsv`
- `make fetch-wikipedia-xml`
  - 入力: `data/processed/wikipedia_article_manifest.tsv`
  - 出力: `data/external/documents/wikipedia/xml/en/Qxxxx.xml`, `data/external/documents/wikipedia/xml/ja/Qxxxx.xml`
- `make extract-wikipedia-text`
  - 入力: `data/processed/wikipedia_article_manifest.tsv`, `data/external/documents/wikipedia/xml/en/Qxxxx.xml`, `data/external/documents/wikipedia/xml/ja/Qxxxx.xml`
  - 出力: `data/external/documents/wikipedia/text/en/Qxxxx.txt`, `data/external/documents/wikipedia/text/ja/Qxxxx.txt`
- `make verify`
  - 入力: `src/multi_bird_db/*.py`
  - 出力: 構文確認結果

### 主な出力ファイル

- `data/interim/wikidata/bird_qids.tsv`
  - Bird 配下の QID 一覧
- `data/interim/wikidata/fetch_entities.sh`
  - Wikidata JSON 取得用シェルスクリプト
- `data/interim/wikidata/json/Qxxxx.json`
  - 各 QID に対応する Wikidata entity JSON
- `data/processed/bird_ontology.tsv`
  - Bird 配下 entity の整理済み表
- `data/processed/wikipedia_article_manifest.tsv`
  - Wikipedia XML と抽出テキストの保存先を管理する表
- `data/external/documents/wikipedia/xml/en/Qxxxx.xml`
  - 英語版 Wikipedia 記事 XML
- `data/external/documents/wikipedia/xml/ja/Qxxxx.xml`
  - 日本語版 Wikipedia 記事 XML
- `data/external/documents/wikipedia/text/en/Qxxxx.txt`
  - 英語版 Wikipedia XML から抽出したテキスト
- `data/external/documents/wikipedia/text/ja/Qxxxx.txt`
  - 日本語版 Wikipedia XML から抽出したテキスト

## ファイル

- [Makefile](Makefile)
  - よく使うコマンドの入口です
- [pyproject.toml](pyproject.toml)
  - Python 実行環境の設定です
- [queries/bird_descendants.sparql](queries/bird_descendants.sparql)
  - `query.tsv` を作るための元クエリです
- [src/multi_bird_db/config.py](src/multi_bird_db/config.py)
  - プロジェクト内のパスをまとめて管理します
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
  - すべての実行コマンドをまとめる入口です
- [src/multi_bird_db/qids.py](src/multi_bird_db/qids.py)
  - `query.tsv` から QID 一覧を作ります
- [src/multi_bird_db/fetch_script.py](src/multi_bird_db/fetch_script.py)
  - Wikidata JSON を取るシェルスクリプトを作ります
- [src/multi_bird_db/ontology.py](src/multi_bird_db/ontology.py)
  - 取得済み JSON から `bird_ontology.tsv` を作ります
- [src/multi_bird_db/wikipedia_articles.py](src/multi_bird_db/wikipedia_articles.py)
  - Wikipedia の manifest 作成、XML 取得、テキスト抽出を行います

## 列説明

### bird_ontology.tsv

- `id`
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
  - Xeno-canto における species ID です。Wikidata の `P2426` を使います
- `taxon_name`
  - 学名です。Wikidata の `P225` を使います
- `taxon_rank`
  - taxon rank の QID です。Wikidata の `P105` を使います
- `taxon_rank_name`
  - taxon rank の英語名です
- `taxon_rank_ja_name`
  - taxon rank の日本語名です
- `parent_taxon`
  - 1 つ上の分類群の QID です。Wikidata の `P171` を使います
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
  - `Q5113` からその QID までを `/Q5113/.../Qxxxx` の形で並べた経路です

### wikipedia_article_manifest.tsv

- `qid`
  - 対応する Wikidata の QID です
- `language`
  - 記事の言語です。`en` は英語版、`ja` は日本語版です
- `title`
  - Wikipedia 記事のタイトルです
- `xml_url`
  - Wikipedia の XML エクスポートを取得するための URL です
- `xml_output_path`
  - 取得した XML を保存するローカルファイルパスです
- `text_output_path`
  - XML から抽出したテキストを保存するローカルファイルパスです

この TSV は、どの Wikipedia 記事をどこから XML として取得し、どこへ XML とテキストを保存するかを整理するための管理表です。参考資料の流れと同様に、XML の取得とテキスト抽出を分離するために使います。`make fetch-wikipedia-xml` と `make extract-wikipedia-text` はこの TSV を入力として使います。

## 初期コミットで追跡するもの

- コード
- 設定ファイル
- `query.tsv`
- ドキュメント
- 作業メモ

生成物や大きなメディアは `.gitignore` で除外しています。必要に応じて Git LFS や別ストレージへの移行を前提にしてください。
