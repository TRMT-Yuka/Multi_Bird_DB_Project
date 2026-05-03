# README_language

`language 側` の README です。ここでは、`qid` 1 件に対して複数の言語表現を持つ前提で、`surface_id` と埋め込みの対応付けをどう持つかをまとめます。

## 概要

- `language 側` の設計をまとめる
- 入力候補は `bird_ontology.pkl` や、それから派生するテキスト・ラベル情報
- `en` と `ja` を分ける
- 各言語ディレクトリには、その言語の表現だけを入れる
- `qid` 1 件に対して複数の `surface_id` を持つ
- `surface_id` と `embeddings.npy` の行を 1 対 1 で対応させる
- 代表表現かどうかの順位づけはしない
- 出力は `EmbeddingStore` 風の保存形式を想定する

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

言語側では、`qid` ごとに複数の言語 surface を作ります。  
graph 側と同じく、**1 列の ID 配列をそのまま埋め込み行列の行番号に対応させる** 方式を基本にします。

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
必要なら `qids[i]` から、その qid に属する複数 item をたどれます。

## 入出力

入力候補:
- `data/processed/bird_ontology.pkl`
- そこから派生する名称・別名・Wikipedia タイトル

生成物:
- `data/external/embeddings/language/en/`
- `data/external/embeddings/language/ja/`
- `surface_ids.json`
- `qids.json`
- `surface_manifest.tsv`
- `embeddings.npy`
- `qid_to_surfaces.json`
- `metadata.json`
- `summary.json`

新しく作られる可能性のあるディレクトリ:
- `data/external/embeddings/language/`
- `data/external/embeddings/language/<language>/`

graph 埋め込みと同じく、language 側も `en/` と `ja/` を別ディレクトリとして保存します。混在させず、言語ごとに独立したストアにします。
`en/` には `en` 系の `surface_id` だけ、`ja/` には `ja` 系の `surface_id` だけが出現する前提です。

## 事前準備

この README に沿って言語側の埋め込み設計を始める前に、少なくとも次が整っている必要があります。

1. `README_wikidata_pred.md` の手順に従って `bird_ontology.pkl` を生成する
2. `data/processed/bird_ontology.pkl` が存在することを確認する
3. `README_graph.md` の保存形式に目を通し、ID と行の対応ルールを揃える

`bird_ontology.pkl` が存在しない状態では、言語側に使う元データの抽出ができません。

## 実行

この節は、言語側で実際に何を揃えるべきかを簡単に整理したものです。  
`surface_id` と表記の対応表だけを先に確認したい場合は、`make build-language-surface-manifest` を使えます。
`surface_id` まで含めた BERT 埋め込みを作る場合は、`make build-language-embeddings` を使えます。
`make build-language-embeddings` を使う前に、`torch` と `transformers`、日本語側では `fugashi` と `unidic-lite` を入れてください。`pip install -e '.[language-bert]'` を使うとまとめて入れられます。
GPU 利用可否を確認したい場合は `make check-gpu` を使えます。`cuda_available` が false なら、その環境では GPU は見えていません。

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

### 1. 言語 surface の ID 体系を決める

ここで決めるのは、`qid` の中にある各言語表現をどう識別するかです。

採用する形式:
- `"{qid}_{lang}_{ordinal}"`

例:
- `Q1591110_en_0`
- `Q1591110_en_1`
- `Q1591110_ja_0`

### 2. ID の順番を固定する

graph 側と同じく、`surface_id` の順番を固定し、その順番を埋め込みの行番号に対応させます。

想定:
- `surface_ids[i]`
- `embeddings[i]`

### 3. メタデータを揃える

最低限、次の情報を `metadata.json` に残すと後で追跡しやすくなります。

- `created_at_utc`
- `source`
- `languages`
- `item_sources`
- `input_files`
- `parameters`

### 4. 保存形式を graph 側と合わせる

graph 側では、以下の 3 点を 1 セットにしています。

- `embeddings.npy`
- `qids.json`
- `metadata.json`

language 側では、`surface_ids.json` を追加して同じ並びにしておくと、読み込み側の実装を共通化しやすくなります。
`surface_ids.json` に加えて、確認用の `surface_manifest.tsv` を出しておくと、`surface_id` と実際の表記の対応を目視しやすくなります。
`qid -> surface_ids` を確認したい場合は、`qid_to_surfaces.json` を見ると機械処理しやすいです。

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

## language 側の構成案

採用方針:
- `en` と `ja` を別ストアにする
- 各ストアの中では、`qid` ごとの複数表現をすべて保持する
- 代表表現かどうかは重要度として扱わない
- 各 row は `surface_id` で識別する

## 参照元の候補

`bird_ontology.pkl` には、言語別の表現に使える情報があります。

- `en_name`
- `ja_name`
- `en_aliases`
- `ja_aliases`
- `enwiki_title`
- `jawiki_title`

これらを使うと、言語ごとの `qid` と説明文の対応表を作れます。

## 対応表メモ

今の `graph 側` と同じ考え方で、`language 側` でも対応表を別ファイルに置くのがよさそうです。  
今回は [docs/language_embeddings_design.md](docs/language_embeddings_design.md) に `surface_id` の形式と保存レイアウトをまとめました。

## 保存形式の想定

graph 側の `EmbeddingStore` に合わせるなら、言語側も次の形が扱いやすいです。

- `surface_ids`
  - row-aligned な言語 surface ID 一覧
- `qids`
  - `surface_ids` と同じ順番の entity QID 一覧
- `embeddings`
  - `surface_ids` と同じ順番の行列
- `metadata`
  - 生成条件や対象データの情報

この構造にしておくと、`get(surface_id)` と `get_many(surface_ids)` のような参照方法をそのまま流用できます。`source` は必要なら `metadata.json` や補助表に逃がします。

## 関連箇所

- [README.md](README.md)
  - リポジトリ全体の入口です
- [README_graph.md](README_graph.md)
  - graph 側の保存形式と構成の参考になります
- [README_wikidata_pred.md](README_wikidata_pred.md)
  - 元データとなる ontology 側の情報を作る手順です
- [src/multi_bird_db/embeddings.py](src/multi_bird_db/embeddings.py)
  - `qids` と `embeddings` を対応づける現在の実装です
