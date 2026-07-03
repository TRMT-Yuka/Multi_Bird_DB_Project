# README_Docker

BirdNET と Perch の GPU 実行だけを、ホストの `.venv_BirdDB` とは別の Docker 環境へ分離するための手順です。

## 含めるもの

- `Dockerfile.audio-gpu`
- `.dockerignore`
- `scripts/run_audio_gpu_container.sh`
- `Makefile` の GPU 用ターゲット

この構成では、リポジトリ配下のコードと手順だけを Git 管理します。モデル cache、音声データ、埋め込み結果、pull 済み image は Git に含めません。

## 目的

- `wav2vec2` はホストの `.venv_BirdDB` で継続利用する
- `BirdNET` と `Perch` のうち、TensorFlow GPU が必要な実行だけを Docker 側へ寄せる
- 出力先はホスト側の `data/` をそのまま使う

## 前提

- Docker コマンドが使えること
- `docker run --gpus all ...` で GPU を見せられること
- ベース image の pull に認証が必要な場合は、事前にその registry へ login しておくこと

## 使うファイル

- `Dockerfile.audio-gpu`
  - TensorFlow GPU 系のベース image 上に、`ffmpeg` と `libsndfile`、`audio-birdnet` / `audio-perch` 依存を入れます
- `scripts/run_audio_gpu_container.sh`
  - image build と `docker run --gpus all` を短いコマンドにまとめます

## 環境変数

- `AUDIO_GPU_BASE_IMAGE`
  - Docker build 時の base image を差し替えるときに使います
- `AUDIO_GPU_IMAGE_TAG`
  - ローカル build 後の image tag を差し替えるときに使います
- `AUDIO_GPU_BUILD`
  - `auto`、`1`、`0` を使えます
  - `auto`: image が無ければ build
  - `1`: 毎回 build
  - `0`: build をスキップ

## 基本コマンド

image を先に build する場合:

```bash
make build-audio-gpu-image
```

TensorFlow から GPU が見えるか確認する場合:

```bash
make check-audio-gpu-tensorflow
```

コンテナ shell に入る場合:

```bash
make run-audio-gpu-shell
```

BirdNET を GPU で動かす場合:

```bash
make build-audio-embeddings-birdnet-gpu
```

Perch を GPU で動かす場合:

```bash
make build-audio-embeddings-perch-gpu
```

## 補足

- リポジトリ全体は `/workspace` として mount されます
- 生成物はホスト側の `data/external/embeddings/audio/` に残ります
- ベース image を差し替えたい場合は、`AUDIO_GPU_BASE_IMAGE=... make build-audio-gpu-image` のように上書きします
- 詳細な backend ごとの実行入口は `README_audio.md` を参照してください
