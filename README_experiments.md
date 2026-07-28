# README_experiments

`audio` 系の追加学習と実験手順をまとめた README です。  
現在はまず音声追加学習実験を記載しています。音声取得と埋め込み生成の基本導線は [README_audio.md](README_audio.md) を参照してください。

## 対象

- `wav2vec2` の 5 分割 cross-validation 追加学習
- 追加学習済みモデルの保存先
- fold ごとの生成物

## wav2vec2 fine-tuning

`data/raw/xeno-canto/` 配下の音声を使い、`Q...` ディレクトリ名を鳥類ラベルとして `wav2vec2` を教師あり fine-tuning できます。  
`recording_map.json` の `recording_ids` と音声ファイル名を対応付け、5 分割 cross-validation で `wav2vec2-model_0` から `wav2vec2-model_4` を作成します。

実装:

- [src/multi_bird_db/audio_finetuning.py](src/multi_bird_db/audio_finetuning.py)
- [Makefile](Makefile)

実行コマンド:

```bash
source .venv_BirdDB/bin/activate
make finetune-wav2vec2-crossval
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli finetune-wav2vec2-crossval \
  --input-dir data/raw/xeno-canto \
  --recording-map data/interim/xeno-canto/recording_map.json \
  --output-dir data/external/models/audio/wav2vec2-finetuned \
  --model-name facebook/wav2vec2-base-960h \
  --device cuda \
  --num-folds 5 \
  --num-epochs 3 \
  --seed 42
```

## Xeno-canto 音声の破損チェックと再取得

`ffmpeg` が decode できない音声ファイルが混入していると、`finetune-wav2vec2-crossval` は学習中に停止します。
その場合は、先に壊れた音声を特定し、Xeno-canto から再ダウンロードして、再度 decode できることを確認します。

通常実行:

```bash
source .venv_BirdDB/bin/activate
make repair-xeno-canto-audio
```

検査のみ行い、ファイルを置換しない場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli repair-xeno-canto-audio --check-only
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli repair-xeno-canto-audio \
  --input-dir data/raw/xeno-canto \
  --recording-map data/interim/xeno-canto/recording_map.json \
  --report data/external/models/audio/wav2vec2-finetuned/xeno_canto_audio_repair.tsv
```

出力:

- `data/external/models/audio/wav2vec2-finetuned/xeno_canto_audio_repair.tsv`
- `data/external/models/audio/wav2vec2-finetuned/xeno_canto_audio_repair.json`

`status` は主に次の値です。

- `valid`: 既存ファイルが decode 可能
- `repaired`: 既存ファイルは壊れていたが、再ダウンロード後のファイルが decode 可能だったため置換済み
- `repair_failed`: 再ダウンロードまたは再decode確認に失敗

修復後に改めて学習を実行します。

```bash
make finetune-wav2vec2-crossval
```

`finetune-wav2vec2-crossval` は既定で `xeno_canto_audio_repair.tsv` を読み、`valid` / `repaired` 以外の音声を学習対象から除外します。
除外された音声は `data/external/models/audio/wav2vec2-finetuned/excluded_audio_files.tsv` に保存されます。

修復レポートを使わず全音声を使う場合のみ、次のように明示します。

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli finetune-wav2vec2-crossval --no-audio-repair-report
```

## 分割ルール

- 5 分割 cross-validation です
- 各 fold が 1 回ずつ test 役になります
- 残り 4 fold はその回の train です
- `seed` が同じなら fold 割り当ては毎回同一です
- 入力音声集合、`recording_map.json`、`seed`、`num-folds` が同じなら分割結果は再現されます
- 1 件しか音声がない `QID` は、全 fold で train-only として扱います

## 出力先

- `data/external/models/audio/wav2vec2-finetuned/wav2vec2-model_0/`
- `data/external/models/audio/wav2vec2-finetuned/wav2vec2-model_1/`
- `data/external/models/audio/wav2vec2-finetuned/wav2vec2-model_2/`
- `data/external/models/audio/wav2vec2-finetuned/wav2vec2-model_3/`
- `data/external/models/audio/wav2vec2-finetuned/wav2vec2-model_4/`

各 fold には次を保存します。

- `train_manifest.tsv`
- `test_manifest.tsv`
- `summary.json`
- `loss_curve.png`
- `accuracy_curve.png`

## 補足

- `wav2vec2` は file 単位のベースラインです
- 音声 decode には `ffmpeg` と `ffprobe` が必要です
- 通常は `PATH` に通します。別ディレクトリに置く場合は `FFMPEG_BIN_DIR` にそのディレクトリを指定します
- `~/ffmpeg-*-static/` に置いた静的 ffmpeg は自動検出されます
- 追加学習済みモデルと、そこから作る埋め込みは保存先が別です
- モデル: `data/external/models/audio/...`
- 埋め込み: `data/external/embeddings/audio/...`

## マルチモーダル分類実験

ここでは、3 種類のモダリティを結合して 1 本の大きなマルチモーダル埋め込みを作り、`QID` に付いている taxon 情報を使って分類先予測を行う実験を想定します。  
この章は、これまで話した仕様を順にまとめた設計メモです。

### 1. 実験の目的

- `graph`
- `audio`
- `language`

の 3 モダリティをまとめて使い、単独モダリティより強い表現が得られるかを見ます。  
最終的には、各埋め込みに対応する `QID` の taxon カテゴリを予測する分類実験を行います。

### 2. 入力単位

基準となる ID は `QID` です。  
ただし、1 `QID` から最終的に 1 本だけ埋め込みを作るのではなく、同じエンティティに紐づく複数パタンを保持したまま扱います。

### 3. 各モダリティの扱い

- `graph`
  - 知識グラフ埋め込みは `QID` ごとに 1 本だけです
  - 複数パタンは存在しません
- `language`
  - 同じエンティティに対して、本名と別名のように複数名称がありえます
  - たとえば名称が 2 つある場合、言語側は 2 パタンになります
- `audio`
  - 同じエンティティに対して複数音声がありえます
  - たとえば音声データが 3 件ある場合、音声側は 3 パタンになります

### 4. QID 内平均はしない

この実験では、同一 `QID` に属する `language` や `audio` の埋め込みを平均して 1 本に潰すことはしません。  
複数パタンをそのまま保持した状態で、後段のマルチモーダル結合に渡します。

### 5. モダリティ結合

最初に試す標準形は、各モダリティのベクトルを単純に連結する方法です。

- `graph_qid_vector`
- `language_pattern_vector`
- `audio_pattern_vector`

を横に並べ、1 本の `multimodal_vector` を作ります。

### 6. 組合せ数

たとえば、同じエンティティにおいて

- 名称が 2 つ
- 音声データが 3 件
- 知識グラフ埋め込みが 1 本

存在するなら、発生するマルチモーダル埋め込みは

- `1 × 2 × 3 = 6`

パタンです。

つまり、1 `QID` に対して 6 本のマルチモーダル埋め込みが発生します。

### 7. 実験対象

まずは、モダリティ有無の比較も含めて見ます。

- `graph` のみ
- `audio` のみ
- `language` のみ
- `graph + audio`
- `graph + language`
- `audio + language`
- `graph + audio + language`

### 8. 予測対象ラベル

分類先は、各埋め込みに対応する `QID` そのものに直接付いた taxon というより、`QID` よりも上位に位置する taxon のカテゴリです。  
ここでいう taxon カテゴリは、ontology / taxonomy graph 側をたどって得られる上位 taxon のうち、いくつかの粒度をラベル候補として採用する想定です。

つまり、1 つの固定ラベルだけを見るのではなく、複数の taxonomic level を切り替えながら、どの粒度のカテゴリが予測しやすいかを比較する方針です。

### 9. 評価単位

学習・推論時の 1 サンプルは、`QID` そのものではなく、`QID` から展開された各パタンの埋め込みです。  
ただし、教師ラベルはその埋め込みに対応する `QID` の taxon カテゴリです。

### 10. この章の位置づけ

この章はまだ設計整理の段階です。  
実際の合同埋め込み生成、学習、分類評価パイプラインはこれから実装します。


## マルチモーダル baseline 実装

分離実装は `src/multi_bird_db/multimodal/` 配下に置き、既存の埋め込み生成コードは入力生成器としてそのまま利用します。  
既存コード本体へ実験ロジックは書き戻さず、CLI 接続だけ追加しています。

現時点の入口:

```bash
source .venv_BirdDB/bin/activate
make inspect-multimodal-sources
make run-multimodal-baseline
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli run-multimodal-baseline \
  --target-rank family \
  --modalities graph,audio,language
```

現時点の baseline は以下です。

- 入力元ディレクトリは設定で差し替え可能
- `QID` 単位で split してから、split 内で `graph × audio × language` を直積展開
- ベクトル結合は単純 concat
- 分類器は新規依存を増やさない `numpy` ベース softmax linear classifier
- 出力先は `data/external/experiments/multimodal_taxon_classification/<timestamp>/`
