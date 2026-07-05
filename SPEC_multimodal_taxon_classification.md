# SPEC_multimodal_taxon_classification

この文書は、`graph`・`audio`・`language` の 3 モダリティを用いたマルチモーダル分類実験を実装するための仕様書です。  
README ではなく、実装起点の設計書として扱います。

## 1. 目的

鳥類エンティティ `QID` に対して、以下 3 種類の埋め込みを組み合わせた分類実験を行う。

- 知識グラフ埋め込み
- 音声埋め込み
- 言語埋め込み

最終目標は、各サンプルに対応する `QID` について、`QID` そのものではなく、`QID` より上位に位置する taxon カテゴリを予測することである。

## 2. 実験の基本前提

### 2.1 基準 ID

- 全モダリティの対応キーは `QID`
- ただし最終サンプル単位は常に `QID` 1 本とは限らない

### 2.2 QID 内平均は行わない

- `audio` は同一 `QID` に複数件存在しうる
- `language` は同一 `QID` に複数名称が存在しうる
- それらを平均して 1 本に潰さない

### 2.3 モダリティごとのパタン数

- `graph`
  - `QID` ごとに常に 1 本
- `audio`
  - `QID` ごとに 0 件以上の複数パタン
- `language`
  - `QID` ごとに 0 件以上の複数パタン

例:

- 同一エンティティに名称が 2 件
- 音声が 3 件
- graph 埋め込みが 1 件

なら、最終的な組合せサンプル数は `1 × 2 × 3 = 6` 件である。

## 3. 入力データ

### 3.1 graph

入力元候補:

- `data/external/embeddings/graph/<method>/<timestamp>/embeddings.npy`
- 同ディレクトリ内の `qids.json`
- 同ディレクトリ内の `metadata.json`

要件:

- `QID -> 1 vector`
- 1 行ごとに `QID` が一意に対応すること

### 3.2 audio

入力元候補:

- `data/external/embeddings/audio/<backend>/<model>/<timestamp>/embeddings.npy`
- `audio_manifest.tsv`
- `qids.json`
- `audio_ids.json`
- `metadata.json`

要件:

- 1 行が 1 音声 clip または 1 window に対応する
- 各行に `QID` が付いていること
- 同一 `QID` に複数行存在してよい

### 3.3 language

入力元候補:

- `data/external/embeddings/language/<language>/embeddings.npy`
- `qids.json`
- `surface_ids.json`
- `metadata.json`

要件:

- 1 行が 1 surface に対応する
- 各行に `QID` が付いていること
- 同一 `QID` に複数行存在してよい

### 3.4 ontology / taxonomy

入力元候補:

- `data/processed/bird_ontology.pkl`
- `data/processed/graph/bird_taxonomy_graph.pkl`

用途:

- `QID` ごとの上位 taxon ラベル付与
- taxonomic level ごとの分類ラベル生成

## 4. サンプル定義

### 4.1 単一モダリティ実験

- graph のみ:
  - 1 `QID` につき 1 サンプル
- audio のみ:
  - 1 音声行につき 1 サンプル
- language のみ:
  - 1 surface 行につき 1 サンプル

### 4.2 複数モダリティ実験

複数モダリティ時は、同一 `QID` 内の行どうしを直積展開してサンプルを作る。

例:

- graph + audio:
  - `graph(1)` × `audio(n)` = `n` サンプル
- graph + language:
  - `graph(1)` × `language(m)` = `m` サンプル
- audio + language:
  - `audio(n)` × `language(m)` = `n * m` サンプル
- graph + audio + language:
  - `graph(1)` × `audio(n)` × `language(m)` = `n * m` サンプル

## 5. 結合方式

### 5.1 初期実装

初期実装では単純連結を採用する。

```text
multimodal_vector =
  concat(
    graph_vector?,
    audio_vector?,
    language_vector?
  )
```

`?` は、その実験設定で当該モダリティを使う場合のみ含める。

### 5.2 モダリティ有無パタン

最低限、以下を比較対象とする。

- graph
- audio
- language
- graph + audio
- graph + language
- audio + language
- graph + audio + language

## 6. ラベル定義

### 6.1 基本方針

分類ラベルは `QID` そのものに直接付いた taxon ではなく、`QID` より上位に位置する taxon を用いる。

### 6.2 ラベル候補

複数の粒度を比較可能にするため、上位 taxon のうち複数 rank をラベル候補として扱う。

候補例:

- species より上位の genus
- family
- order
- class

注意:

- 実際にどの rank を使うかは taxonomy 側の網羅性を見て決める
- すべての `QID` に対して一意に引ける rank のみ採用する

### 6.3 ラベル生成ルール

各 `QID` に対して:

1. taxonomy graph もしくは ontology から祖先 taxon を取得する
2. 指定 rank に一致する祖先 taxon を探す
3. 見つかった taxon の `QID` か `name` を分類ラベルとして使う

未解決時の扱い:

- 指定 rank の祖先が取れない `QID` は、その rank 実験から除外する

## 7. 分割戦略

### 7.1 分割単位

train / validation / test の分割は、サンプル単位ではなく `QID` 単位で行う。

理由:

- 同じ `QID` から生成された複数パタンが train/test にまたがると情報漏洩になるため

### 7.2 展開順

1. まず `QID` を train / validation / test に分ける
2. その後で各 split 内の `QID` を展開し、各モダリティ組合せサンプルを作る

## 8. 学習タスク

### 8.1 タスク種別

- multi-class classification

### 8.2 入力

- 1 サンプル = 1 結合ベクトル

### 8.3 出力

- 指定 rank における上位 taxon ラベル

## 9. 実装対象

最低限、以下の機能を分離して実装する。

### 9.1 埋め込みロード層

役割:

- graph / audio / language の各埋め込みをロードする
- 行ごとの `QID` とベクトルを対応付ける

想定モジュール:

- `src/multi_bird_db/multimodal_embeddings.py`

必要関数:

- graph 埋め込みロード
- audio 埋め込みロード
- language 埋め込みロード

### 9.2 taxon ラベル生成層

役割:

- `QID` ごとに指定 rank の上位 taxon ラベルを付与する

必要関数:

- `QID -> ancestors` 取得
- `QID + target_rank -> label` 変換

### 9.3 サンプル展開層

役割:

- 単一モダリティ / 複数モダリティの組合せサンプルを生成する
- `QID` 内直積を作る

必要関数:

- `expand_graph_only`
- `expand_audio_only`
- `expand_language_only`
- `expand_graph_audio`
- `expand_graph_language`
- `expand_audio_language`
- `expand_graph_audio_language`

### 9.4 分割層

役割:

- `QID` 単位で split を作る
- その後サンプル展開する

必要関数:

- `split_qids(...)`
- `materialize_split_samples(...)`

### 9.5 学習・評価層

役割:

- ベクトル分類器を学習する
- rank ごとの分類性能を評価する

初期実装候補:

- logistic regression
- linear classifier
- MLP classifier

## 10. 出力ファイル仕様

出力先候補:

- `data/external/experiments/multimodal_taxon_classification/<timestamp>/`

最低限必要な生成物:

- `config.json`
- `dataset_summary.json`
- `split_summary.json`
- `label_summary.tsv`
- `metrics.tsv`
- `predictions.tsv`
- `classification_report.txt`

必要なら追加:

- confusion matrix 図
- rank ごとの比較図

## 11. 中間生成物

再利用しやすくするため、中間表も保存する。

候補:

- `qid_label_map.tsv`
  - `QID -> target_rank -> label`
- `sample_manifest.tsv`
  - 各展開サンプルの由来を記録
- `multimodal_embeddings.npy`
  - 実験で使う最終結合ベクトル

`sample_manifest.tsv` の最低列候補:

- `sample_id`
- `qid`
- `graph_row_id`
- `audio_row_id`
- `language_row_id`
- `modality_pattern`
- `target_rank`
- `target_label`

## 12. 検証項目

実装後は少なくとも以下を検証する。

- 同じ `QID` の複数パタンが train/test にまたがっていないこと
- graph は常に `QID` ごとに 1 本であること
- audio / language は複数行を保持していること
- `2 name × 3 audio = 6 samples` のような直積展開が正しいこと
- 指定 rank の taxon ラベルが一意に振られていること

## 13. 既知の未確定事項

以下は今後決定が必要である。

- 初期分類器を何にするか
- どの taxonomic rank を正式採用するか
- language 側で何を 1 パタンとみなすか
- audio 側で何を 1 パタンとみなすか
- 欠損モダリティを持つ `QID` をどう扱うか
- モダリティ次元差の正規化を行うか

## 14. 非目標

この仕様書の段階では、以下はまだ扱わない。

- end-to-end 深層マルチモーダル学習
- attention によるモダリティ融合
- 欠損モダリティ補完
- retrieval 実験との統合評価

## 15. 初期実装で固定するモデル

この実験系は後で差し替え可能にするが、最初の実装では以下を固定採用する。

### 15.1 graph モダリティ

採用モデル:

- `graphsage`

理由:

- このリポジトリ内で実装済み
- GPU 実行導線がある
- 既にこの環境で学習を回している
- `QID -> 1 vector` が自然に得られる

入力元の標準候補:

- `data/external/embeddings/graph/graphsage/<timestamp>/`

### 15.2 audio モダリティ

採用モデル:

- `wav2vec2`
- ベースモデル名: `facebook/wav2vec2-base-960h`

理由:

- この環境で最も安定して扱える音声埋め込み backend
- 既に埋め込み生成導線が整っている
- BirdNET / Perch より初期実装の不確定性が低い

入力元の標準候補:

- `data/external/embeddings/audio/wav2vec2/<model>/<timestamp>/`

補足:

- 初期実装では、追加学習済み wav2vec2 ではなく、まず既存の `wav2vec2` 埋め込み生成系を基礎として使う
- 追加学習済みモデルの利用は拡張項目とする

### 15.3 language モダリティ

採用モデル:

- 英語: `google-bert/bert-base-uncased`

理由:

- 現在のリポジトリ実装の既定値のうち、まず英語だけに絞る
- `surface_id` 単位の埋め込み生成が既にある
- 初期実装では構成を単純化しやすい

入力元の標準候補:

- `data/external/embeddings/language/en/`

補足:

- 日本語 `tohoku-nlp/bert-base-japanese-v3` は将来の切替候補として残す
- 初期実装では language は英語のみ採用する

### 15.4 初期固定構成のまとめ

最初のマルチモーダル分類実装では、以下の構成を標準構成とする。

- graph: `graphsage`
- audio: `wav2vec2` (`facebook/wav2vec2-base-960h`)
- language: English `google-bert/bert-base-uncased`

この固定構成で、まずデータロード、サンプル展開、ラベル付与、分類評価までの基礎部分を完成させる。  
その後、必要に応じて graph / audio / language の各モデルを差し替え可能に拡張する。
