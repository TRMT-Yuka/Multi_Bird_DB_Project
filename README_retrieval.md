# README_retrieval

各モダリティごとの埋め込み種類・有無を切り替えながら、`QID` 単位の検索実験を行うための README です。

この README は、現時点のリポジトリで既に生成できる埋め込みを前提に、**どの埋め込みを比較対象にするか**、**検索実験の単位をどう揃えるか**、**どの組合せを評価するか**を整理します。

## 目的

- `graph` / `audio` / `language` の各モダリティを比較する
- 各モダリティ内の埋め込み種類ごとの差を見る
- あるモダリティを `有り / 無し` にした時の検索性能差を見る
- 最終的に `QID` 単位で bird entity retrieval を比較する

## 最初に読む場所

- [README.md](README.md)
  - リポジトリ全体の入口です
- [README_graph.md](README_graph.md)
  - graph 埋め込みの生成手順です
- [README_audio.md](README_audio.md)
  - audio 埋め込みと `wav2vec2` fine-tuning の手順です
- [README_language.md](README_language.md)
  - language 埋め込みの生成手順です

## 検索実験の基本単位

この合同実験では、**検索対象を `QID` 単位に揃える** ことを前提にします。

- graph
  - もともと `QID -> 1 embedding row` です
- audio
  - 現状は `audio_id` または window 単位の行を持ちます
  - したがって、検索実験の前に `QID` ごとの集約が必要です
- language
  - 現状は `surface_id` 単位の行を持ちます
  - したがって、検索実験の前に `QID` ごとの集約が必要です

つまり、**合同検索実験の前処理として、全モダリティを `QID -> 1 vector` に揃える** 必要があります。

## 比較対象にする埋め込み

### Graph

候補:

- `node2vec`
- `gcn`
- `grace`
- `graphsage`
- `transe`
- `none`

保存先:

- `data/external/embeddings/graph/node2vec/<MMDDhhmm>/`
- `data/external/embeddings/graph/gcn/<MMDDhhmm>/`
- `data/external/embeddings/graph/grace/<MMDDhhmm>/`
- `data/external/embeddings/graph/graphsage/<MMDDhhmm>/`
- `data/external/embeddings/graph/transe/<MMDDhhmm>/`

### Audio

候補:

- `wav2vec2`
- `birdnet`
- `perch`
- `wav2vec2-finetuned fold model`
- `none`

保存先:

- `data/external/embeddings/audio/wav2vec2/<model>/<MMDDhhmm>/`
- `data/external/embeddings/audio/birdnet/<model>/<MMDDhhmm>/`
- `data/external/embeddings/audio/perch/<model>/<MMDDhhmm>/`
- `data/external/models/audio/wav2vec2-finetuned/wav2vec2-model_0/` から `wav2vec2-model_4/`

注意:

- `wav2vec2` / `birdnet` / `perch` の埋め込みはそのままだと clip 単位または window 単位です
- 検索実験では `QID` ごとに平均などで 1 本へ集約する必要があります
- `wav2vec2-finetuned` は分類モデルの学習済み重みです
- 埋め込みとして使う場合は、その encoder から別途特徴抽出する実験設計が必要です

### Language

候補:

- `bert-en`
- `bert-ja`
- `bert-en+ja`
- `none`

保存先:

- `data/external/embeddings/language/en/`
- `data/external/embeddings/language/ja/`

注意:

- `en` / `ja` ともに `surface_id` 単位です
- 検索実験では `QID` ごとに代表ベクトルへ集約する必要があります
- `bert-en+ja` を使う場合は、英語と日本語の `QID` 集約後ベクトルを平均または連結します

## 検索実験の設計

最小単位は次の通りです。

- query
  - ある `QID` のベクトル
- gallery
  - 他の全 `QID` のベクトル
- score
  - cosine similarity
- rank
  - 類似度降順

評価対象の例:

- 同一 `QID` の近傍順位
- 上位 `k` 件の中に同 rank または近縁 taxon がどれだけ含まれるか
- taxon rank ごとの retrieval 難易度差

## Ablation の切り方

### 1. 単一モダリティ比較

- graph のみ
- audio のみ
- language のみ

### 2. モダリティ有無の比較

- graph + audio
- graph + language
- audio + language
- graph + audio + language
- それぞれの `none` を含む比較

### 3. モダリティ内埋め込み種類の比較

例:

- graph
  - `node2vec` vs `gcn` vs `grace` vs `graphsage` vs `transe`
- audio
  - `wav2vec2` vs `birdnet` vs `perch`
- language
  - `bert-en` vs `bert-ja` vs `bert-en+ja`

### 4. fine-tuned wav2vec2 の比較

- `wav2vec2` 事前学習モデルのみ
- `wav2vec2-model_0` から `wav2vec2-model_4` の fold 別学習済みモデル

## 推奨する実験表

最初は次のように段階的に進めると整理しやすいです。

1. graph 単独
   - `node2vec`, `gcn`, `grace`, `graphsage`, `transe`
2. audio 単独
   - `wav2vec2`, `birdnet`, `perch`
3. language 単独
   - `bert-en`, `bert-ja`, `bert-en+ja`
4. 2 モダリティ結合
   - best graph + best audio
   - best graph + best language
   - best audio + best language
5. 3 モダリティ結合
   - best graph + best audio + best language

## 実行前に揃えておくもの

### Graph

```bash
make build-graph
make build-node2vec-embeddings
make build-gcn-embeddings
make build-grace-embeddings
make build-graphsage-embeddings
make build-transe-embeddings
```

### Audio

```bash
source .venv_BirdDB/bin/activate
make build-audio-embeddings-wav2vec2
make build-audio-embeddings-birdnet
make build-audio-embeddings-perch
```

BirdNET / Perch を Docker GPU で回す場合:

```bash
make build-audio-gpu-image
make check-audio-gpu-tensorflow
make build-audio-embeddings-birdnet-gpu
make build-audio-embeddings-perch-gpu
```

`wav2vec2` fine-tuning を回す場合:

```bash
source .venv_BirdDB/bin/activate
make finetune-wav2vec2-crossval
```

### Language

```bash
source .venv_BirdDB/bin/activate
make build-language-surface-manifest
make build-language-embeddings
```

## 集約方針

合同検索実験では、次のような `QID` 単位集約を最初の標準にするのが無難です。

- graph
  - そのまま使う
- audio
  - 同一 `QID` に属する全 clip / window の平均
- language-en
  - 同一 `QID` に属する全英語 surface の平均
- language-ja
  - 同一 `QID` に属する全日本語 surface の平均
- language-en+ja
  - `en` 平均と `ja` 平均の平均、または連結

## 融合方針

まずは単純な late fusion から始めるのを推奨します。

- 各モダリティで `QID` 単位ベクトルを作る
- 各ベクトルを L2 normalize する
- 結合方法を 2 種類試す
  - 平均
  - 連結

比較軸:

- normalize あり / なし
- 平均融合 / 連結融合
- 特定モダリティを抜いた ablation

## 現時点の実装状況

- 生成済み / 実行可能
  - graph 埋め込み生成
  - audio 埋め込み生成
  - language 埋め込み生成
  - `wav2vec2` fine-tuning
- README で設計済みだが、専用 CLI はまだ無いもの
  - `QID` 単位へのモダリティ横断集約
  - 検索スコア計算
  - retrieval 指標の自動評価
  - 融合実験の一括実行

つまり、**合同検索実験の設計はこの README に整理し、実際の retrieval パイプラインはこれから実装する** 状態です。

## 出力として最低限残すもの

合同実験を回すときは、少なくとも次を保存する前提にします。

- 使用した各埋め込み run のパス
- `QID` 単位に集約したベクトルの保存先
- 使用した融合方法
- 使用したモダリティの有無
- retrieval のスコア表
- 上位件数の qualitative 例
- 設定 JSON と summary JSON

## 関連箇所

- [src/multi_bird_db/embeddings.py](src/multi_bird_db/embeddings.py)
- [src/multi_bird_db/audio_embeddings.py](src/multi_bird_db/audio_embeddings.py)
- [src/multi_bird_db/audio_finetuning.py](src/multi_bird_db/audio_finetuning.py)
- [src/multi_bird_db/language_embeddings.py](src/multi_bird_db/language_embeddings.py)
