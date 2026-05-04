# README_audio

`audio 側` の README です。ここでは、鳥類音声データを `qid` と結びつけて保存し、後段の特徴量抽出や音声埋め込みへつなぐための整理をします。

## 概要

- `audio 側` の設計をまとめる
- 入力は音声ファイル本体と、そのメタデータです
- `qid` 1 件に対して複数の音声クリップを持てる前提にする
- 音声クリップ単位の ID と `qid` を分離する
- 将来的に特徴量抽出、音声埋め込み、分類モデル学習へつなぐ
- 保存形式は row-aligned な manifest と埋め込み行列の組み合わせを基本にする
- 埋め込み backend は `wav2vec`、`perch`、`birdnet` の 3 系列を前提にする
- Xeno-canto の生音声は `data/raw/xeno-canto/after_202505/<qid>/` に置く

## 最初に読む場所

- [README.md](README.md)
  - プロジェクト全体の入口です
- [README_wikidata_pred.md](README_wikidata_pred.md)
  - `Wikidata 側` で ontology PKL を作る手順です
- [README_graph.md](README_graph.md)
  - `qid` をキーにした保存形式の参考になります
- [README_language.md](README_language.md)
  - `qid` と surface ID の分離の考え方が参考になります
- [data/README.md](data/README.md)
  - データ配置ルールです
- [docs/architecture.md](docs/architecture.md)
  - 全体の拡張方針です

## 何を作るのか

audio 側では、`qid` ごとに複数の音声クリップを持つことを想定します。  
クリップは `audio_id` で識別し、`qid` と 1 対多で対応させます。

つまり、次のような対応です。

- audio item ID 一覧
  - `audio_ids.json`
- QID 一覧
  - `qids.json`
- 音声メタデータ
  - `audio_manifest.tsv`
- 埋め込み本体
  - `embeddings.npy`
- 補助情報
  - `metadata.json`
  - `summary.json`

この構成にすると、`audio_ids[i]` の埋め込みは `embeddings[i]` で直接引けます。  
必要なら `qids[i]` から、その `qid` に属する複数クリップをたどれます。

## 入出力

入力候補:
- 音声ファイル本体
- 収録メタデータ
- `qid` とファイル名の対応表
- 録音元・観測地点・日時などの補助情報

生成物の候補:
- `data/raw/xeno-canto/after_202505/`
- `data/external/audio/`
- `data/external/audio/<dataset>/<backend>/`
- `audio_ids.json`
- `qids.json`
- `audio_manifest.tsv`
- `embeddings.npy`
- `metadata.json`
- `summary.json`

新しく作られる可能性のあるディレクトリ:
- `data/raw/xeno-canto/after_202505/`
- `data/raw/xeno-canto/after_202505/<qid>/`
- `data/external/audio/`
- `data/external/audio/<dataset>/`
- `data/external/audio/<dataset>/<backend>/`

dataset 単位でディレクトリを分けると、録音ソースごとの前処理や評価条件を分離しやすくなります。  
backend 単位でも分けると、`wav2vec`、`perch`、`birdnet` の出力を混在させずに比較できます。

## 事前準備

この README に沿って audio 側の処理を始める前に、少なくとも次が整っている必要があります。

1. `qid` と音声ファイルの対応表を用意する
2. 音声ファイルの配置ルールを決める
3. `audio_id` の採番ルールを決める
4. `qid` ごとの複数クリップをどう扱うか決める

## ID 体系

audio 側では、`qid` とクリップ ID を分けます。

採用候補:

- `audio_id`
  - 1 つの音声クリップを識別する ID
- `qid`
  - 鳥類 entity の主キー
- `source`
  - どのデータセットや出典から来たか

推奨形式:

- `"{qid}_{source}_{ordinal}"`

例:

- `Q5113_xeno_canto_0`
- `Q5113_xeno_canto_1`
- `Q5113_local_archive_0`

`ordinal` は同一 `qid`・同一 `source` の中で複数候補があるときの連番で、重要度ではありません。

## Row Alignment

graph 側や language 側と同じく、埋め込み行は 1 つの ID 列に対応させます。

audio 側では、その行対応の主キーを `audio_id` にします。

必要なら補助的に `qid` を並行保存して、`audio_id -> qid` の関係を保持します。

## Candidate Sources

今後の候補としては、次のような音声表現が考えられます。

- 鳴き声の元 WAV ファイル
- その切り出しクリップ
- スペクトログラム画像
- 手作業で切り出した高品質区間

最初の実装では、音声ファイル単位の埋め込みを基本にして、必要に応じて短いクリップ単位へ分割するのが扱いやすいです。

## Output Layout

各 dataset ごとに別ディレクトリを作る想定です。

```text
data/raw/xeno-canto/after_202505/
├── <qid>/
│   └── XCxxxxxx.mp3
├── audio_manifest.tsv
├── audio_ids.json
├── qids.json
├── metadata.json
└── summary.json

data/external/audio/
├── <dataset>/
│   ├── audio_manifest.tsv
│   ├── audio_ids.json
│   ├── qids.json
│   ├── embeddings.npy
│   ├── metadata.json
│   └── summary.json
```

## File Semantics

- `data/raw/xeno-canto/after_202505/<qid>/XCxxxxxx.mp3`
  - Xeno-canto から取得した生音声ファイルです
- `audio_manifest.tsv`
  - `audio_id`、`qid`、ファイルパス、長さ、サンプリングレート、出典などを並べる
- `audio_ids.json`
  - row-aligned な audio item ID 一覧
- `qids.json`
  - 同じ順序の `qid` 一覧
- `embeddings.npy`
  - `audio_ids.json` の順に並ぶ行列
- `metadata.json`
  - 使用した前処理、特徴量抽出器、対象データセットなど
- `summary.json`
  - `item_count`
  - `unique_qid_count`
  - `dataset`
  - `source_counts`

## Deterministic Ordering

順番は意味的な優先順位ではなく、再現性のために固定します。

推奨順序:

1. `qid` の昇順
2. `source` の昇順
3. `ordinal` の昇順
4. ファイル名の昇順

## Audio Model Baseline

最初の実装候補は、次の 3 系列を並行比較する方針です。

### 汎用音声

- `wav2vec`
  - 汎用音声表現の基準として使う
  - 鳥類に限らない比較基準を持てる

### 生態音響

- `Perch`
  - 動物音声・保全用途の表現として使う
  - 鳥類以外も含む bioacoustics 比較に向く

### 鳥類特化

- `BirdNET`
  - 鳥種識別の強い基準として使う
  - 鳥類データに最も近い比較軸になる
  - 論文公開は 2021-01-27（Ecological Informatics の online publication）で、GitHub の BirdNET-Analyzer リポジトリでは少なくとも 2025-11-07 の `v2.4.0` リリースが確認できる
  - この repo の git 履歴上では、現時点で追える最初の commit は 2026-04-11 で、BirdNET 関連のメモを置く際の参照点として使える
  - 公式サイトは continuous data-driven updates を掲げているため、モデル版によっては学習データが更新・追加されている可能性がある
  - ただし、公開情報だけでは各版の学習データ差分や明確なカットオフ日は追えないことがある
  - `birdnet` Python パッケージは `0.1.7`（2025-03-19）を採用候補とする
  - これは、`2025-04` 以前のモデルを確実に入手し、学習側と評価側を分離しやすくするためである

まずは backend ごとに特徴抽出器を固定し、`audio_id` と行の対応を壊さないことを優先します。

## API Shape

graph 側や language 側と同じく、次の形が扱いやすいです。

- `audio_ids`
- `qids`
- `embeddings`
- `metadata`

主な参照関数は次のとおりです。

- `get(audio_id)`
- `get_many(audio_ids)`
- `audio_for_qid(qid)`

## Non-Goals

- 1 つの `qid` に対する代表音声の優先順位付け
- 複数 dataset を 1 つのストアに混ぜること
- 画像やテキストと音声を同一の ID 空間に押し込むこと

## Xeno-canto Download

Xeno-canto の音声取得は次のコマンドで行います。

```bash
make fetch-xeno-canto-audio
```

既定では `bird_ontology.pkl` から `xeno_canto_species_id` を読み、`2025-05-01` 以降に投稿された録音を各 `qid` につき最大 10 件まで取得します。  
保存先は `data/raw/xeno-canto/after_202505/<qid>/` です。

## 関連箇所

- [README.md](README.md)
  - リポジトリ全体の入口です
- [README_graph.md](README_graph.md)
  - `qid` 主キーの保存形式の参考になります
- [README_language.md](README_language.md)
  - `surface_id` と `qid` を分ける考え方の参考になります
- [data/README.md](data/README.md)
  - データ配置ルールです
