# Multi_Bird_DB_Project

このリポジトリは、鳥類データを複数モダリティで扱うための入口です。現在は Wikidata を中心とした事前データダウンロードと前処理を先に整備しています。

## README 一覧

- [README_wikidata_pred.md](README_wikidata_pred.md)
  - Wikidata dump、ontology PKL、Wikipedia 記事一覧までの現在の主処理です
- [README_graph.md](README_graph.md)
  - ontology PKL から taxonomy graph を構築し、埋め込み生成と Dash viewer で観察する処理です
- [README_audio.md](README_audio.md)
  - 音声データの配置方針と今後の処理整理です

## 現在の実装状況

- 実装済み
  - Wikidata dump の取得
  - dump から対象 QID の JSON 抽出
  - ontology PKL 生成
  - taxonomy graph PKL 生成
  - taxonomy graph からの埋め込み生成
  - taxonomy graph の Dash viewer
  - Wikipedia 記事 manifest 生成
- 未実装または今後整理
  - 音声データ処理

## 読み方

- まず全体構成は [docs/architecture.md](docs/architecture.md) を参照してください。
- 現在の実行手順は [README_wikidata_pred.md](README_wikidata_pred.md) を参照してください。
- データ配置ルールは [data/README.md](data/README.md) を参照してください。
