# README_wikidata_pred

`Wikidata 側` の README です。Bird (`Q5113`) 配下の QID、Wikidata JSON、ontology PKL を作る前処理をまとめます。`graph 側` の処理は [README_graph.md](README_graph.md) に分離しています。

## 概要

- Bird (`Q5113`) 配下の QID を抽出する
- Wikidata dump から対象 JSON を切り出す
- ontology PKL を生成する
- 将来の文書、音声、画像データの置き場を確保する

## 最初に読む場所

- [docs/architecture.md](docs/architecture.md)
  - 全体構成
- [data/README.md](data/README.md)
  - データ配置ルール

## 事前準備

1. Python 3.10 以上を使えるようにします。
2. リポジトリのルートへ移動します。

```bash
cd Multi_Bird_DB_Project
```

3. `data/raw/wikidata/query.tsv` を用意します。
   `queries/bird_descendants.sparql` を Wikidata Query Service で実行して得た TSV を置きます。
4. 最新の Wikidata JSON dump を取得します。

```bash
make download-wikidata-dump
```

`latest-all.json.bz2` は非常に大きいので、十分な空き容量を確保してください。保存先は `data/raw/wikidata/dumps/latest-all.json.bz2` です。

5. `qwikidata` を環境に入れます。

```bash
pip install qwikidata
```

## 実行

### 早見表

- QID 一覧の取得
  - `make extract-qids`
- Wikidata dump の取得
  - `make download-wikidata-dump`
- dump から JSON を切り出す
  - `make extract-dump-json`
- ontology PKL を作る
  - `make build-ontology`
- 構文確認
  - `make verify`

### 手順

### 0. 作業ディレクトリへ移動する

```bash
cd Multi_Bird_DB_Project
```

### 1. 入力ファイルを用意する

`data/raw/wikidata/query.tsv` は、`queries/bird_descendants.sparql` を Wikidata Query Service で実行して得る TSV です。`Q5113` 配下の entity URL を並べたものを保存してください。

### 2. QID 一覧を作る

```bash
make extract-qids
```

入力: `data/raw/wikidata/query.tsv`

出力: `data/interim/wikidata/bird_qids.tsv`

### 3. Wikidata dump を取得する

```bash
make download-wikidata-dump
```

入力: Wikidata dump

出力: `data/raw/wikidata/dumps/latest-all.json.bz2`

取得元は `https://dumps.wikimedia.org/wikidatawiki/entities/` です。完了前に次へ進むと失敗します。

### 4. dump から対象 QID の JSON を切り出す

```bash
make extract-dump-json
```

入力: `data/interim/wikidata/bird_qids.tsv`, `data/raw/wikidata/dumps/latest-all.json.bz2`

出力: `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json`

JSON は QID の数値部で階層化して保存し、既存ファイルは上書きしません。

### 5. ontology PKL を作る

```bash
make build-ontology
```

入力: `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json`

出力: `data/processed/bird_ontology.pkl`

`data/interim/wikidata/json/` が空だと失敗します。

### 6. graph 側の処理

taxonomy graph の構築、埋め込み、Dash viewer は [README_graph.md](README_graph.md) を参照してください。

## 出力

- `data/interim/wikidata/bird_qids.tsv`
  - QID 一覧
- `data/raw/wikidata/dumps/latest-all.json.bz2`
  - Wikidata 全量 dump
- `data/interim/wikidata/json/<1桁目>/<2桁目>/Qxxxx.json`
  - 各 QID の entity JSON
- `data/processed/bird_ontology.pkl`
  - ontology 成果物

## 関連ファイル

- [Makefile](Makefile)
  - 実行コマンドの入口
- [queries/bird_descendants.sparql](queries/bird_descendants.sparql)
  - `query.tsv` 用の元クエリ
- [scripts/download_wikidata_dump.sh](scripts/download_wikidata_dump.sh)
  - Wikidata dump のダウンロード
- [src/multi_bird_db/qids.py](src/multi_bird_db/qids.py)
  - QID 抽出
- [src/multi_bird_db/dump_extract.py](src/multi_bird_db/dump_extract.py)
  - dump から JSON 抽出
- [src/multi_bird_db/ontology.py](src/multi_bird_db/ontology.py)
  - ontology 生成
