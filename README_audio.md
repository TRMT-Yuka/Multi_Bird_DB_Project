# README_audio

`audio` 側の README です。音声取得と音声埋め込みの入口をまとめます。

## 概要

- 入力は音声ファイル本体とメタデータです
- `qid` ごとに複数クリップを持てます
- 取得元は `data/interim/wikidata/bird_xeno_canto_ids.tsv` です
- Xeno-canto は `recording_map.json` を作ってから音声取得します
- Xeno-canto は品質 `A` のみを取得します
- 音声は `data/raw/xeno-canto/<qid>/` に保存します
- 1 `xeno_canto_species_id` につき最大 30 件まで取得します
- API キーは `xeno_canto_api_key.env.example` を `xeno_canto_api_key.env` に複製して使います

## Backend 契約

音声埋め込み backend は共通の CLI から切り替えます。  
現在は `wav2vec2` と `birdnet` を実装済みで、`perch` は契約だけ先に登録しています。

共通ルール:

- 出力先は `data/external/embeddings/audio/<backend>/<model>/<MMDDhhmm>/`
- `embeddings.npy` と `audio_manifest.tsv` の行順を揃える
- `audio_ids.json` と `qids.json` は行順に対応させる
- 失敗した入力は `failed_items.json` に残す

backend ごとの既定:

- `wav2vec2`
  - window: ファイル全体
  - 目的: ベースライン
  - 必要な Python 系: `torch`, `torchaudio`, `transformers`
  - 必要なシステム系: `ffmpeg`
- `birdnet`
  - window: 3 秒
  - 既定サンプルレート: 48 kHz
  - 目的: 鳥類特化の基準
  - 必要な Python 系: `birdnet`, `tensorflow`, `tensorflow-hub`, `soundfile`
  - 必要なシステム系: `ffmpeg`, `libsndfile`
- `perch`
  - window: 5 秒
  - 目的: Perch-Hoplite の既定窓幅に合わせる
  - 必要な Python 系: `perch-hoplite`, `tensorflow`, `tensorflow-hub`, `jax`, `soundfile`
  - 必要なシステム系: `ffmpeg`, `libsndfile`

## 音声埋め込み

BirdNET を使う例:

```bash
make build-audio-embeddings
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli build-audio-embeddings \
  --backend birdnet \
  --input-dir data/raw/xeno-canto \
  --output-dir data/external/embeddings/audio \
  --model-name birdnet-acoustic-2.4-tf \
  --device cpu \
  --batch-size 8 \
  --max-seconds 30
```

このコマンドは入力ディレクトリ配下を再帰的に走査し、BirdNET なら 3 秒窓ごとに埋め込みを作ります。  
出力は `data/external/embeddings/audio/<backend>/<model>/<MMDDhhmm>/` 配下に保存されます。

生成物:

- `embeddings.npy`
- `audio_ids.json`
- `qids.json`
- `audio_manifest.tsv`
- `metadata.json`
- `summary.json`
- `failed_items.json`

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

入力は `data/interim/wikidata/bird_xeno_canto_ids.tsv` です。  
各 `xeno_canto_species_id` に対して `api/3/recordings` を取得し、JSON を `data/interim/xeno-canto/api_recordings/<qid>/` に保存します。  
検索条件は品質 `A` を含みます。

### 2. API JSON から recording-id を抽出する

```bash
make extract-xeno-canto-recording-ids
```

直接 CLI を叩く場合:

```bash
PYTHONPATH=src python3 -m multi_bird_db.cli extract-xeno-canto-recording-ids \
  --input data/interim/xeno-canto/api_recordings/api_recordings_manifest.json \
  --output-json data/interim/xeno-canto/recording_map.json
```

`api_recordings_manifest.json` を読み、`xeno_canto_species_id -> recording_id` の対応を `recording_map.json` に保存します。  
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
  --limit-per-qid 30 \
  --clip-seconds 30
```

各録音は `file` URL から取得し、`data/raw/xeno-canto/<qid>/<recording_id>.<file_type>` に保存します。  
`recording_id` は `XC` を外した数値部分です。`ffmpeg` が必要です。
