# Multi_Bird_DB_Project

このリポジトリは、鳥類データを複数モダリティで扱うための入口です。現在は `Wikidata 側` の前処理と `graph 側` の構築を分けて整備しています。

## README 一覧

- [README_wikidata_pred.md](README_wikidata_pred.md)
  - `Wikidata 側` の処理です。dump の取得、JSON 抽出、ontology PKL 生成までを扱います
- [README_graph.md](README_graph.md)
  - `graph 側` の処理です。ontology PKL から graph を構築し、埋め込み生成・Dash viewer で観察します
- [README_audio.md](README_audio.md)
  - 音声データの配置方針と今後の処理整理です
- [README_language.md](README_language.md)
  - 言語単位の ID と埋め込みの対応付けを整理するメモです

## 現在の実装状況

- 実装済み
  - `Wikidata 側`
    - Wikidata dump の取得
    - dump から対象 QID の JSON 抽出
    - ontology PKL 生成
  - `graph 側`
    - taxonomy graph PKL 生成
    - graph からの埋め込み生成
    - taxonomy graph の Dash viewer
  - `audio 側`
    - Xeno-canto 音声の再帰走査と wav2vec2 埋め込み生成
- 未実装または今後整理
  - BirdNET / Perch などの他音声バックエンド比較

## 読み方

- まず全体構成は [docs/architecture.md](docs/architecture.md) を参照してください。
- 現在の実行手順は [README_wikidata_pred.md](README_wikidata_pred.md) を参照してください。
- データ配置ルールは [data/README.md](data/README.md) を参照してください。
