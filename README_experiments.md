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
- `ffmpeg` が `PATH` 上で実行可能である必要があります
- 追加学習済みモデルと、そこから作る埋め込みは保存先が別です
- モデル: `data/external/models/audio/...`
- 埋め込み: `data/external/embeddings/audio/...`
