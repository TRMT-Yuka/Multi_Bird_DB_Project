# README_wikidata_pred

Bird (`Q5113`) 配下の Wikidata エンティティを再現的に取得し、将来的な文書・音声・画像・グラフ埋め込みを同一プロジェクト内で扱えるように整理した初期コミット用の雛形です。`Bird_ontology_Project` の知見を引き継ぎつつ、固定列挙の取得コードから、再実行しやすい CLI ベースのパイプラインへ置き換えています。

## 概要

- Bird (`Q5113`) 配下の QID 一覧を再取得可能にする
- Wikidata entity JSON を保存し、ontology PKL を再生成できるようにする
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
- Wikidata database downloads
  - https://dumps.wikimedia.org/wikidatawiki/entities/
  - JSON dump の公式配布場所です

## ディレクトリ構成

```text
data/                                     # このプロジェクトで扱う全データのルート
├── raw/                                  # 外部から取得した未加工データ
│   └── wikidata/                         # Wikidata の入力領域
│       ├── query.tsv                     # 手動取得して配置する Bird 配下の entity URL 一覧
│       └── dumps/                        # 取得した Wikidata JSON dump の保存先
├── interim/                              # 処理途中で生成される中間データ
│   └── wikidata/                         # Wikidata の処理作業領域
│       ├── bird_qids.tsv                 # QID 抽出結果
│       └── json/                         # QID ごとの JSON 応答
├── processed/                            # 下流利用向けに整形済みの成果物
│   ├── bird_ontology.pkl                 # 鳥類 ontology 構造化成果物
│   └── wikipedia_article_manifest.tsv    # Wikipedia 取得対象一覧
└── external/                             # 文書・音声・埋め込みの保管先
    ├── documents/                        # 文書コーパス
    ├── audio/                            # 音声ファイルや音声メタデータ
    └── embeddings/                       # グラフ・テキスト・画像埋め込み
```

既存のデータ取り扱いルールに従い、このプロジェクト内で完結するデータ配置にしています。詳細は [data/README.md](data/README.md) と [docs/work_notes/2026-04-11_initial_scaffold.md](docs/work_notes/2026-04-11_initial_scaffold.md) に残しています。

## データ生成から再現する場合の事前準備

このプロジェクトをコードだけの状態から再現する場合は、実行前に次を準備してください。

1. Python 3.10 以上を使えるようにします。
2. このリポジトリのルートへ移動します。

```bash
cd Multi_Bird_DB_Project
```

3. `data/raw/wikidata/query.tsv` を手動で用意します。
   このファイルは現時点では自動生成しません。Bird 配下の一覧取得を自動化すると、外部サービスへのアクセスが攻撃と誤認されるリスクがあるため、この段階は意図的に手動運用としています。
4. 最新の Wikidata JSON dump を取得します。

```bash
make download-wikidata-dump
```

このコマンドは実行前に、所要時間と保存容量に関する警告を表示します。非対話的に実行する場合は、明示的な確認として `CONFIRM_LARGE_DOWNLOAD=1 make download-wikidata-dump` を使ってください。

この dump は非常に大きく、`latest-all.json.bz2` の時点でおよそ `93.7 GiB` あります。展開前でも大容量なので、少なくとも 100 GiB 程度の空き容量を見込んでください。

このリポジトリの作業環境では、`2026-04-08` 版の最新版 dump をダウンロードする際、概ね `3.5〜4 MB/s` 程度で進んでおり、完了までの見込みはおよそ `7〜9 時間` でした。回線状況や Wikimedia 側の負荷で大きく変動するため、実際にはさらに余裕を見てください。

ダウンロード先は `data/raw/wikidata/dumps/latest-all.json.bz2` です。`scripts/download_wikidata_dump.sh` は `curl --continue-at -` を使っているため、中断した場合も再実行で途中から再開できます。

5. dump を扱うために `qwikidata` を現在の conda 環境へ入れます。

```bash
pip install qwikidata
```

`environment.yml` にも `qwikidata` を追加しています。`Makefile` は `PYTHONPATH=src` を付与しているため、現状の CLI は追加インストールなしでも自作コード自体は実行できます。

## 実行

この節は 2 段に分かれています。

- 上段
  - 目的ごとに使うコマンドをすぐ確認するための早見表です
- 下段
  - `### 0` 以降の番号付き手順です。実際に上から順に実行する場合はこちらを見ます

### 実行コマンドの早見表

- QID 一覧の取得
  - `make extract-qids`
- 最新の Wikidata dump の取得　※取得ファイルは非常に重いので事前に容量を確保すること
  - `make download-wikidata-dump`
- dump を直接走査して対象 QID の JSON を切り出す
  - `make extract-dump-json`
- JSONから ontology PKL を作りたい場合
  - `make build-ontology`
- ontology PKL から taxonomy graph を作りたい場合
  - `make build-graph`
- taxonomy graph の PNG 可視化を作りたい場合
  - `make visualize-graph`
- taxonomy graph をインタラクティブに観察したい場合
  - `make serve-graph`
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

入力は `data/raw/wikidata/query.tsv` です。

`data/raw/wikidata/query.tsv` は、[Wikidataクエリサービス](https://query.wikidata.org/) にアクセスし、以下のクエリを手動で実行することで得られる TSV ファイルです。

```sparql
SELECT DISTINCT ?item ?itemLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],ja,en". }
  {
    SELECT DISTINCT ?item WHERE {
      ?item p:P171 ?statement0.
      ?statement0 (ps:P171/(wdt:P171*)) wd:Q5113.
    }
  }
}
```

上記クエリは、[queries/bird_descendants.sparql](queries/bird_descendants.sparql) に保存しているものと同じです。Bird_ontology_Project と同様に、この段階は外部サービス上で手動実行する運用とします。Bird 配下一覧の取得を自動化すると、外部サービスへのアクセスが攻撃と誤認されるリスクがあるためです。

`query.tsv` には、以下のような Wikidata のエンティティページ URL が並びます。これらは、`Q5113`（Bird）をより上位の分類 `taxon` に持つエンティティ一覧です。スズメやツバメのような鳥類名だけでなく、スズメ目のような分類群も含みます。

```tsv
item
http://www.wikidata.org/entity/Q132731
http://www.wikidata.org/entity/Q136317
http://www.wikidata.org/entity/Q179112
http://www.wikidata.org/entity/Q182761
...(略)...
```

取得した TSV は `data/raw/wikidata/query.tsv` として保存してください。

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

### 3. Wikidata dump を取得する

```bash
make download-wikidata-dump
```

実行される処理:
- [scripts/download_wikidata_dump.sh](scripts/download_wikidata_dump.sh)

生成物:
- `data/raw/wikidata/dumps/latest-all.json.bz2`

この dump は Wikidata 公式の JSON dump 配布場所 `https://dumps.wikimedia.org/wikidatawiki/entities/` から取得します。

ダウンロードが完了する前に次の `make extract-dump-json` を実行すると、不完全な `.bz2` を読むことになり失敗する可能性があります。`latest-all.json.bz2` の取得完了を確認してから次へ進んでください。

### 4. dump から対象 QID の JSON を切り出す

このステップを実行する前に、`make extract-qids` と `make download-wikidata-dump` を実行しておく必要があります。

```bash
make extract-dump-json
```

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/dump_extract.py](src/multi_bird_db/dump_extract.py)

生成先:
`data/interim/wikidata/json/`

新しく作られるファイル:
- `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json`
  - 各 QID に対応する Wikidata entity JSON

JSON は、QID の数値部の 1 桁目と 2 桁目で階層化して保存します。たとえば `Q27614643` は `data/interim/wikidata/json/2/7/Q27614643.json` に保存されます。

`make extract-dump-json` は、`latest-all.json.bz2` を直接走査して対象 QID の JSON を取り出します。出力は `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json` で、既に同じ JSON が存在する場合は上書きしません。

JSON は一時ファイル経由で原子的に書き込むため、途中停止で壊れにくくなっています。実行中は標準エラー出力に簡単な進捗を表示します。


### 5. ontology PKL を作る

```bash
make build-ontology
```

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/ontology.py](src/multi_bird_db/ontology.py)

`build-ontology` は `data/interim/wikidata/json/` に JSON が存在しない場合は失敗するようにしています。空の PKL を誤って生成しないためです。

生成物:
[data/processed/bird_ontology.pkl](data/processed/bird_ontology.pkl)

### 6. taxonomy graph PKL を作る

```bash
make build-graph
```

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/graph.py](src/multi_bird_db/graph.py)

生成物:
- `data/processed/graph/bird_taxonomy_graph.pkl`

この graph は `bird_ontology.pkl` を入力にして、`parent_taxon -> id` の有向エッジで taxonomy を表現します。構造の詳細は [README_graph.md](README_graph.md) を参照してください。

### 7. taxonomy graph を可視化する

```bash
make visualize-graph
```

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/graph_visualization.py](src/multi_bird_db/graph_visualization.py)

生成物:
- `data/processed/graph/figures/bird_taxonomy_graph.png`

この可視化は、巨大な graph 全体ではなく、root QID から一定深さまでの部分グラフを PNG として保存します。可視化仕様の詳細は [README_graph.md](README_graph.md) を参照してください。

### 8. Wikipedia 記事一覧を作る

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

### 8. 英語版・日本語版 Wikipedia 記事 XML を取得する

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

### 9. Wikipedia XML からテキストを抽出する

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

## 実行例

```bash
make extract-qids
make download-wikidata-dump
make extract-dump-json
make build-ontology
make build-graph
make serve-graph
make build-wikipedia-manifest
```

## 出力

### 何を実行した結果、何が入力とされ、何が出力されるか

- `make extract-qids`
  - 入力: `data/raw/wikidata/query.tsv`
  - 出力: `data/interim/wikidata/bird_qids.tsv`
- `make download-wikidata-dump`
  - 入力: `https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2`
  - 出力: `data/raw/wikidata/dumps/latest-all.json.bz2`
- `make extract-dump-json`
  - 入力: `data/interim/wikidata/bird_qids.tsv`, `data/raw/wikidata/dumps/latest-all.json.bz2`
  - 出力: `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json`
- `make build-ontology`
  - 入力: `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json`
  - 出力: `data/processed/bird_ontology.pkl`
- `make build-graph`
  - 入力: `data/processed/bird_ontology.pkl`
  - 出力: `data/processed/graph/bird_taxonomy_graph.pkl`
- `make serve-graph`
  - 入力: `data/processed/graph/bird_taxonomy_graph.pkl`
  - 出力: ローカル Dash サーバ
- `make build-wikipedia-manifest`
  - 入力: `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json`
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
- `data/raw/wikidata/dumps/latest-all.json.bz2`
  - Wikidata 全量 JSON dump
- `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json`
  - 各 QID に対応する Wikidata entity JSON
- `data/processed/bird_ontology.pkl`
  - Bird 配下 entity の構造化成果物
- `data/processed/graph/bird_taxonomy_graph.pkl`
  - Bird taxonomy の graph 構造化成果物
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
  - 現時点ではこのクエリは外部サービスで手動実行する運用です
- [scripts/download_wikidata_dump.sh](scripts/download_wikidata_dump.sh)
  - 最新の Wikidata dump をダウンロードします
- [src/multi_bird_db/config.py](src/multi_bird_db/config.py)
  - プロジェクト内のパスをまとめて管理します
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
  - すべての実行コマンドをまとめる入口です
- [src/multi_bird_db/qids.py](src/multi_bird_db/qids.py)
  - `query.tsv` から QID 一覧を作ります
- [src/multi_bird_db/dump_extract.py](src/multi_bird_db/dump_extract.py)
  - dump の直接走査で JSON を切り出します
- [src/multi_bird_db/ontology.py](src/multi_bird_db/ontology.py)
  - 取得済み JSON から `bird_ontology.pkl` を作ります
- [src/multi_bird_db/graph.py](src/multi_bird_db/graph.py)
  - `bird_ontology.pkl` から taxonomy graph を作ります
- [src/multi_bird_db/wikipedia_articles.py](src/multi_bird_db/wikipedia_articles.py)
  - Wikipedia の manifest 作成、XML 取得、テキスト抽出を行います


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
- ドキュメント
- 作業メモ

生成物や大きなメディアは `.gitignore` で除外しています。必要に応じて Git LFS や別ストレージへの移行を前提にしてください。
