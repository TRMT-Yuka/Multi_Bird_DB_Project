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

追加学習済みモデルから audio embedding を作る場合:

```bash
source .venv_BirdDB/bin/activate
make build-audio-embeddings-wav2vec2-finetuned
```

出力先は `data/external/embeddings/audio/wav2vec2-finetuned/` です。
各 fold は `wav2vec2-model_0` から `wav2vec2-model_4` のラベルで保存されます。

## 補足

- `wav2vec2` は file 単位のベースラインです
- 音声 decode には `ffmpeg` と `ffprobe` が必要です
- 通常は `PATH` に通します。別ディレクトリに置く場合は `FFMPEG_BIN_DIR` にそのディレクトリを指定します
- `~/ffmpeg-*-static/` に置いた静的 ffmpeg は自動検出されます
- 追加学習済みモデルと、そこから作る埋め込みは保存先が別です
- モデル: `data/external/models/audio/...`
- 埋め込み: `data/external/embeddings/audio/...`

## 実験コードの所在

旧結果と実験方針は、[experiments/old_result_1/README.md](experiments/old_result_1/README.md) に保存しています。新しい実験結果は `experiments/exp1/`、`experiments/exp2/`、`experiments/exp3/` に出力されます。
