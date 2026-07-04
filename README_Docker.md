# README_Docker

BirdNET の GPU 実行を、ホストの `.venv_BirdDB` とは別の Docker 環境へ分離するための手順です。Perch はこのマシンでは保守せず、別マシン・別環境で実行してください。

## 含めるもの

- `Dockerfile.audio-birdnet-gpu`
- `.dockerignore`
- `scripts/run_audio_birdnet_gpu_container.sh`
- `Makefile` の GPU 用ターゲット

この構成では、リポジトリ配下のコードと手順だけを Git 管理します。モデル cache、音声データ、埋め込み結果、pull 済み image は Git に含めません。

## 方針

TensorFlow を `arm64` / `aarch64` Linux 環境で NVIDIA GPU 利用する場合は、原則として NVIDIA NGC の TensorFlow コンテナをベースにします。

- `python:*-slim` ベースに `pip install tensorflow` はしない
- CUDA / cuDNN / TensorFlow wheel を Dockerfile 内で手動構築しない
- 先に NGC image 単体で GPU 認識を確認し、通ったタグだけを Dockerfile に固定する

## このマシンで確認したタグ

2026-07-04 時点では、次の両方で `tf.config.list_physical_devices('GPU')` に `GPU:0` が出ました。

- `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`
- `nvcr.io/nvidia/tensorflow:25.02-tf2-py3-igpu`

BirdNET 用コンテナでは、まず標準タグの `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` を採用しています。

## 事前確認

ホスト GPU:

```bash
nvidia-smi
```

Docker GPU passthrough:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

NGC TensorFlow 単体確認:

```bash
make check-birdnet-ngc-tensorflow-gpu
```

直接実行するなら:

```bash
docker run --rm --gpus all nvcr.io/nvidia/tensorflow:25.02-tf2-py3 \
  python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

## BirdNET 用 GPU コンテナ

BirdNET 用 Dockerfile は `Dockerfile.audio-birdnet-gpu` です。

特徴:

- base image は `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`
- TensorFlow 自体は NGC 側を使い、`pip install tensorflow` はしない
- 追加するのは `birdnet==0.2.16`、`tensorflow-hub==0.16.1`、`soundfile` と最小限のシステム依存だけ
- `birdnet[and-cuda]` は使わない
  - これは別の TensorFlow / CUDA wheel 群を引き込み、NGC ベースを崩すため

image を build する場合:

```bash
make build-audio-birdnet-gpu-image
```

shell に入る場合:

```bash
make run-audio-birdnet-gpu-shell
```

BirdNET 埋め込みを実行する場合:

```bash
make build-audio-embeddings-birdnet-gpu
```

BirdNET_2 埋め込みを実行する場合:

```bash
make build-audio-embeddings-birdnet-2-gpu
```

## Perch について

- `perch` backend のコード自体はリポジトリ内に残します
- ただし、このマシン向けの Docker wrapper は保守しません
- 別マシン・別環境で `perch` の埋め込みを作成し、生成された `data/external/embeddings/audio/perch/...` をこのリポジトリへ戻してください
