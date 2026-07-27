# README_Docker

BirdNET GPU の旧 Docker メモです。現在の主経路はホストの `birdnet` conda 環境から直接 GPU を使う方法なので、このファイルは補助情報としてだけ残しています。

## 現在の推奨

```bash
conda activate birdnet
make build-audio-embeddings-birdnet-gpu
```

この実行は `conda run -n birdnet` を使い、`BIRDNET_APP_DATA` はプロジェクト内 `temp/birdnet_appdata` を使います。

## 旧 Docker 方式について

- 旧 Docker wrapper や Dockerfile は残していますが、現在の主経路ではありません
- 当時の検証メモとして参照するだけで十分です

## Perch について

- `perch` backend のコード自体はリポジトリ内に残します
- ただし、このマシン向けの Docker wrapper は保守しません
- 別マシン・別環境で `perch` の埋め込みを作成し、生成された `data/external/embeddings/audio/perch/...` をこのリポジトリへ戻してください
