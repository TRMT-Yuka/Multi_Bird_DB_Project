# README_audio

`audio` 側の README です。音声取得と音声埋め込みの入口をまとめます。`wav2vec2` はこのリポジトリの通常作業用 conda 環境 `bird_env`、BirdNET は別 conda 環境 `birdnet` を前提に整理します。BirdNET の GPU 実行もホストの `birdnet` 環境から直接行います。Perch はこのマシンでは追わず、別マシン・別環境で実行して生成物だけを戻す前提にします。追加学習と実験手順は [README_experiments.md](README_experiments.md) に分離しています。

## 概要

- 入力は音声ファイル本体とメタデータです
- `qid` ごとに複数クリップを持てます
- 取得元は `data/interim/wikidata/bird_xeno_canto_ids.tsv` です
- Xeno-canto は `recording_map.json` を作ってから音声取得します
- Xeno-canto は品質 `A` のみを取得します
- Xeno-canto の対象は taxonomy graph の末端 QID に限定します
- 音声は `data/raw/xeno-canto/<qid>/` に保存します
- 1 `xeno_canto_species_id` につき最大 20 件まで取得します
- API キーは `xeno_canto_api_key.env.example` を `xeno_canto_api_key.env` に複製して使います
- 既存の音声ファイルは `existing_audio_manifest.json` に記録して再取得をスキップします
- 音声取得の一時ファイルは `temp/xeno-canto/` に置き、処理後に削除します

## Backend 契約

音声埋め込み backend は共通の CLI から切り替えます。  
現在は `wav2vec2`、`birdnet`、`perch` の backend 分岐があります。  
この環境でまず本番利用しやすいのは `wav2vec2` です。

共通ルール:

- 出力先は `data/external/embeddings/audio/<backend>/<model>/<MMDDhhmm>/`
- `embeddings.npy` と `audio_manifest.tsv` の行順を揃える
- `audio_ids.json` と `qids.json` は行順に対応させる
- 失敗した入力は `failed_items.json` に残す
- 長時間実行では、バッチ完了ごとに `*.partial.*` の途中経過も上書き保存する

## 実行環境の整理

- `wav2vec2`
  - `bird_env` で実行します
- `birdnet`
  - CPU 実行は `birdnet` conda 環境で行います
  - GPU 実行も同じ `birdnet` conda 環境から行います
  - GPU ライブラリが不足する場合は `conda run -n birdnet python -m pip install --upgrade "tensorflow[and-cuda]==2.21.0"` を実行します
- `perch`
  - backend 自体は残しています
  - このマシン向けの実行導線は保守しません
  - 別マシン・別環境で埋め込みを作成し、生成された `embeddings.npy` と manifest をこのリポジトリへ戻してください

backend ごとの既定:

- `wav2vec2`
  - window: ファイル全体
  - 既定サンプルレート: 16 kHz
  - 目的: ベースライン
  - 必要な Python 系: `torch`, `transformers`
  - 必要なシステム系: `ffmpeg`
  - モデル事前取得: `make download-audio-models`
- `birdnet`
  - window: 3 秒
  - 既定サンプルレート: 48 kHz
  - 目的: 鳥類特化の基準
  - 必要な Python 系: `birdnet`, `tensorflow`, `tensorflow-hub`, `soundfile`
  - このコード経路は `torch` 非依存です
  - 必要なシステム系: `ffmpeg`, `libsndfile`
  - 実装状況: 実装済み
  - GPU 実行: `birdnet` conda 環境から直接実行
- `perch`
  - window: 5 秒
  - 既定サンプルレート: 32 kHz
  - 目的: Bioacoustics Model Zoo の旧公式 `Perch` 埋め込み
  - 必要な Python 系: `bioacoustics-model-zoo`, `tensorflow`, `tensorflow-hub`, `soundfile`
  - このコード経路は `torch` 非依存ですが、依存解決の都合で `torch` 系が入ることがあります
  - 必要なシステム系: `ffmpeg`, `libsndfile`
  - 実装状況: backend のみ保持
  - 実行方針: このマシンでは保守せず、別マシン・別環境で実行

## 音声埋め込み

`wav2vec2` をこの環境で使う前提の最短手順:

```bash
conda activate bird_env
python -m pip install -e '.[audio-wav2vec2]'
make download-audio-models
make build-audio-embeddings-wav2vec2
```

モデルだけ先に取得したい場合:

```bash
conda activate bird_env
make download-audio-models
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli download-audio-models \
  --backend wav2vec2 \
  --model-name facebook/wav2vec2-base-960h \
  --device auto \
  --cache-dir data/external/models/audio/huggingface
```

`wav2vec2` のモデルは既定で `data/external/models/audio/huggingface` に保存されます。  
以後は同じ cache を使って再利用します。

注意:

- この実装では共通の音声 decode / resample を `ffmpeg` ベースで処理します
- `build-audio-embeddings-wav2vec2` の前に、`ffmpeg` と `ffprobe` を実行可能にしておく必要があります
- 通常は `PATH` に通します。別ディレクトリに置く場合は `FFMPEG_BIN_DIR` にそのディレクトリを指定します
- `~/ffmpeg-*-static/` に置いた静的 ffmpeg は自動検出されます

Xeno-canto 音声の破損チェックと再取得:

```bash
source .venv_BirdDB/bin/activate
make repair-xeno-canto-audio
```

検査のみで置換しない場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli repair-xeno-canto-audio --check-only
```

結果は `data/external/models/audio/wav2vec2-finetuned/xeno_canto_audio_repair.tsv` に保存されます。
`repair_failed` など修復不能な音声は、`finetune-wav2vec2-crossval` 実行時に既定で除外されます。

BirdNET を使う例:

```bash
conda activate birdnet
make build-audio-embeddings-birdnet
```

BirdNET を GPU で使う例:

```bash
conda activate birdnet
make check-birdnet-gpu
make build-audio-embeddings-birdnet-gpu
```

補足:

- ホスト側から一発で実行する場合は `make build-audio-embeddings-birdnet-gpu` を使います
- このターゲットは `conda run -n birdnet` と `BIRDNET_APP_DATA=temp/birdnet_appdata` を内部で設定します
- 先に `nvidia-smi` が通ることを確認してください

直接 CLI を叩く場合:

```bash
conda run -n birdnet --no-capture-output env BIRDNET_APP_DATA=$(pwd)/temp/birdnet_appdata PYTHONPATH=src python3 -m multi_bird_db.cli build-audio-embeddings \
  --backend birdnet \
  --input-dir data/raw/xeno-canto \
  --output-dir data/external/embeddings/audio \
  --device auto \
  --batch-size 8 \
  --max-seconds 30
```

GPU を明示したい場合:

```bash
conda run -n birdnet --no-capture-output env BIRDNET_APP_DATA=$(pwd)/temp/birdnet_appdata PYTHONPATH=src python3 -m multi_bird_db.cli build-audio-embeddings \
  --backend birdnet \
  --device cuda
```

このコマンドは入力ディレクトリ配下を再帰的に走査し、BirdNET なら 3 秒窓、`48 kHz` で埋め込みを作ります。`auto` では TensorFlow が GPU を認識していれば `pb` backend を選びます。  
出力は `data/external/embeddings/audio/<backend>/<model>/<MMDDhhmm>/` 配下に保存されます。

`birdnet` は BirdNET 公式の file-based `encode()` にファイルパスを直接渡し、3 秒分割も BirdNET 側に任せます。  
`--resume-existing` を付けると、既存の `embeddings.*` だけでなく `*.partial.*` も読んで、完了済み音声をスキップしながら再開できます。バッチサイズ変更後の再開にも使えます。

Perch を使う例:

```bash
conda activate bird_env
make build-audio-embeddings-perch
```

Perch について:

- このリポジトリ内のローカル Docker wrapper は保守しません
- 別マシン・別環境で `perch` backend を実行し、出力ディレクトリごとこのリポジトリへ戻してください

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli build-audio-embeddings \
  --backend perch \
  --input-dir data/raw/xeno-canto \
  --output-dir data/external/embeddings/audio \
  --model-name perch \
  --device cpu \
  --batch-size 8 \
  --max-seconds 30
```

このコマンドは入力ディレクトリ配下を再帰的に走査し、旧公式 Perch なら 5 秒窓、`32 kHz` 相当の公式前処理で埋め込みを作ります。  
ただしこのマシンでの動作確認は追わず、別マシン・別環境での実行を前提にします。出力は `data/external/embeddings/audio/<backend>/<model>/<MMDDhhmm>/` 配下に保存されます。

生成物:

- `embeddings.npy`
- `audio_ids.json`
- `qids.json`
- `audio_manifest.tsv`
- `metadata.json`
- `summary.json`
- `failed_items.json`

## 追加学習と実験

`wav2vec2` の 5 分割 cross-validation 追加学習など、実験系の手順は [README_experiments.md](README_experiments.md) に分離しています。

補足:

- `wav2vec2` は file 単位のベースラインです
- `birdnet` は 3 秒窓、`48 kHz` です。CPU 時は `birdnet.load("acoustic", "2.4", "tf")`、GPU 時は `birdnet.load("acoustic", "2.4", "pb")` を使います
- `perch` は 5 秒窓、`32 kHz`、`bioacoustics-model-zoo` の旧公式 `Perch.embed()` と clip DataFrame API を使います
- `perch` は別マシン・別環境で実行し、生成物だけをこのリポジトリへ戻してください

## Xeno-canto

### 1. API JSON を保存する

```bash
make fetch-xeno-canto-recording-json
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli fetch-xeno-canto-recording-json \
  --input data/interim/wikidata/bird_xeno_canto_ids.tsv \
  --output-dir data/interim/xeno-canto/api_recordings \
  --api-key "$XENO_CANTO_API_KEY"
```

入力は `data/interim/wikidata/bird_xeno_canto_ids.tsv` です。存在しない場合は `data/processed/bird_ontology.pkl` から自動生成してから進みます。  
各 `xeno_canto_species_id` に対して `api/3/recordings` を 1 ページだけ取得し、JSON を `data/interim/xeno-canto/api_recordings/<qid>/` に保存します。  
検索条件は品質 `A`、`per_page=20` です。

### 2. API JSON から recording-id を抽出する

```bash
make extract-xeno-canto-recording-ids
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli extract-xeno-canto-recording-ids \
  --input data/interim/xeno-canto/api_recordings \
  --output-json data/interim/xeno-canto/recording_map.json
```

`data/interim/xeno-canto/api_recordings/<qid>/page*.json` を読み、`xeno_canto_species_id -> recording_id` の対応を `recording_map.json` に保存します。  
API 応答の `recordings[].file` が音声ダウンロード URL です。

### 3. recording-id から音声を取得する

```bash
make fetch-xeno-canto-audio
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli fetch-xeno-canto-audio \
  --input data/interim/xeno-canto/recording_map.json \
  --output-dir data/raw/xeno-canto \
  --limit-per-qid 20 \
  --clip-seconds 15 \
  --sleep-seconds 0.25
```

各録音は `file` URL から取得し、先頭 15 秒を切り出して `data/raw/xeno-canto/<qid>/<recording_id>.<file_type>` に保存します。  
`recording_id` は `XC` を外した数値部分です。`ffmpeg` が必要です。
