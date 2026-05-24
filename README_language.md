# README_language

`language 側` の README です。ここでは、`qid` 1 件に対して複数の言語表現を持つ前提で、`surface_id` と埋め込みの対応付けをどう持つかをまとめます。

## 概要

- `language 側` の設計をまとめる
- 入力は `bird_ontology.pkl` です
- `en` と `ja` を分ける
- 各言語ディレクトリには、その言語の表現だけを入れる
- `qid` 1 件に対して複数の `surface_id` を持つ
- `surface_id` と `embeddings.npy` の行を 1 対 1 で対応させる
- 代表表現かどうかの順位づけはしない
- 出力は `surface_ids`・`qids`・`embeddings` を同じ行順で持つ保存形式にする

## 最初に読む場所

- [README.md](README.md)
  - プロジェクト全体の入口です
- [README_graph.md](README_graph.md)
  - `graph 側` の保存形式と埋め込みの対応付けの参考になります
- [README_wikidata_pred.md](README_wikidata_pred.md)
  - `Wikidata 側` で `bird_ontology.pkl` を作る手順です
- [data/README.md](data/README.md)
  - データ配置ルールです
- [docs/architecture.md](docs/architecture.md)
  - `Wikidata 側` から `graph 側`、その先の拡張までの全体像です

## 何を作るのか

言語側では、`qid` ごとに複数の言語 surface を作り、`surface_id` の行と `embeddings.npy` の行を 1 対 1 で対応させます。

つまり、次のような対応です。

- language surface ID 一覧
  - `surface_ids.json`
- QID 一覧
  - `qids.json`
- `qid -> surface_ids` と表記一覧
  - `qid_to_surfaces.json`
- 埋め込み本体
  - `embeddings.npy`
- メタデータ
  - `metadata.json`
- 補助情報
  - `summary.json`

この構成にすると、`surface_ids[i]` の埋め込みは `embeddings[i]` で直接引けます。  
`qids[i]` から、その qid に属する複数表現をたどれます。

## 入出力

入力候補:
- `data/processed/bird_ontology.pkl`
- そこから派生する名称・別名・Wikipedia タイトル

生成物:
- `data/external/embeddings/language/en/`
- `data/external/embeddings/language/ja/`
- `surface_manifest.tsv`
- `surface_ids.json`
- `qids.json`
- `qid_to_surfaces.json`
- `embeddings.npy`
- `metadata.json`
- `summary.json`

新しく作られる可能性のあるディレクトリ:
- `data/external/embeddings/language/`
- `data/external/embeddings/language/<language>/`

graph 埋め込みと同じく、language 側も `en/` と `ja/` を別ディレクトリとして保存します。混在させず、言語ごとに独立したストアにします。
`en/` には `en` 系の `surface_id` だけ、`ja/` には `ja` 系の `surface_id` だけが出現する前提です。

## 事前準備

`bird_ontology.pkl` が存在していることを確認します。これが言語側の唯一の入力です。

## 実行

`surface_id` と表記の対応表は `make build-language-surface-manifest` で作れます。  
`BERT` 埋め込みは `make build-language-embeddings` で作れます。  
`build-language-embeddings` は `torch` と `transformers` を使い、日本語側では `fugashi` と `unidic-lite` も使います。`pip install -e '.[language-bert]'` でまとめて入れられます。  
GPU を使う場合は `--device cuda` を指定します。`make check-gpu` で CUDA の見え方を確認できます。

### 実行の早見表

- QID の一覧を作りたい場合
  - `qids.json`
- surface_id と実際の表記の対応表を作りたい場合
  - `make build-language-surface-manifest`
- BERT 埋め込みを作りたい場合
  - `make build-language-embeddings`
- GPU の見え方を確認したい場合
  - `make check-gpu`
- 言語ごとの埋め込みを保存したい場合
  - `embeddings.npy`
- 生成条件や対象コーパスを残したい場合
  - `metadata.json`
- 行順と ID の対応を明示したい場合
  - `summary.json`

### 詳細手順

### 0. 作業ディレクトリへ移動する

```bash
cd Multi_Bird_DB_Project
```

### 1. 言語 surface の ID 体系

採用する形式は `"{qid}_{lang}_{ordinal}"` です。

### 2. ID の順番を固定する

`surface_ids[i]` と `embeddings[i]` を同じ順番で対応させます。

### 3. メタデータを揃える

`metadata.json` には、実装では主に次の情報が入ります。

- `created_at_utc`
- `language`
- `input_file`
- `encoder_model`
- `device`
- `batch_size`
- `max_length`
- `pooling`
- `surface_id_pattern`
- `qid_to_surfaces_format`
- `item_count`
- `unique_qid_count`
- `embedding_dim`
- `source_counts`

### 4. 保存形式

graph 側は `embeddings.npy`、`qids.json`、`metadata.json` を 1 セットにしています。

language 側は `surface_ids.json`、`qids.json`、`qid_to_surfaces.json`、`embeddings.npy`、`metadata.json`、`summary.json` を同じ順番で保存します。`surface_manifest.tsv` は確認用です。

### 5. BERT 埋め込みを生成する

```bash
make build-language-embeddings
```

前提ファイル:
- `data/processed/bird_ontology.pkl`

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/language_embeddings.py](src/multi_bird_db/language_embeddings.py)

生成物:
- `data/external/embeddings/language/en/embeddings.npy`
- `data/external/embeddings/language/ja/embeddings.npy`

このコマンドは `surface_manifest.tsv` で作った `surface_id` 順を保ったまま、英語は `google-bert/bert-base-uncased`、日本語は `tohoku-nlp/bert-base-japanese-v3` を使って埋め込みを作ります。既定では CPU を使います。GPU を使いたい場合は `--device cuda` を指定できますが、`torch` と `transformers` が別途必要です。


## 関連箇所

- [README.md](README.md)
  - リポジトリ全体の入口です
- [README_graph.md](README_graph.md)
  - graph 側の保存形式と構成の参考になります
- [README_wikidata_pred.md](README_wikidata_pred.md)
  - 元データとなる ontology 側の情報を作る手順です
- [src/multi_bird_db/embeddings.py](src/multi_bird_db/embeddings.py)
  - `qids` と `embeddings` を対応づける現在の実装です
