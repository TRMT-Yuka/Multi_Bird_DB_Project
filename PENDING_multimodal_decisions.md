# PENDING_multimodal_decisions

このメモは、実装を止めずに前進するための暫定判断と未確定事項を記録する。

## 現時点の暫定判断

### 1. 実装境界

- 新規実装は `src/multi_bird_db/multimodal/` に閉じる
- 既存コードは読んで再利用するが、実験のための改変はしない

### 2. 初期採用モデル

- graph: `graphsage`
- audio: `wav2vec2` (`facebook/wav2vec2-base-960h`)
- language: 英語 `google-bert/bert-base-uncased`

### 3. ディレクトリ差し替え

- 埋め込み作成元モデルを後で差し替えられるよう、初期実装では各モダリティの「入力ディレクトリ」を設定値として持つ
- コードは特定の run ディレクトリ構造に強く依存せず、最終的に `embeddings.npy` と対応メタデータを読めればよい構造に寄せる

## 未確定事項

### 1. 初期分類器

- logistic regression
- linear classifier
- MLP

のどれを先に標準化するかは未確定だったが、実装を止めないため暫定的に

- `numpy` ベースの softmax linear classifier

を初期ベースラインとして採用する

### 2. 正式採用ラベル粒度

- genus
- family
- order
- class

のどれを最初の正式評価対象にするかは未確定

### 3. 欠損モダリティ

- ある `QID` に音声や名称が不足している場合の扱いは未確定

### 4. 言語の複数名称

- 英語側の `surface` をそのまま全採用するか
- 一部の source に限定するか

は未確定
