# Work Note 2026-04-11

## 参照したルール

- 既存のデータ取り扱いルールを確認した
- 新規データはこのプロジェクト配下で管理する
- 既存の共有データ領域には新規データを追加しない

## 今回の判断

- `Bird_ontology_Project` の固定列挙型取得スクリプトは、そのままでは更新や再実行に弱いため採用しない
- `query.tsv` を `data/raw/wikidata/` に置き、そこから中間生成物を作る流れへ変更した
- 将来扱う文書、音声、画像、埋め込みを `data/external/` 配下に整理した
- 取得処理は `src/multi_bird_db/` に集約し、`scripts/` は薄いラッパにした

## 今後の作業候補

- `bird_ontology.tsv` に `Xeno-canto species ID` と Wikipedia 関連列を追加した
- Wikipedia 取得方針を API 直接取得から XML 取得へ変更した
- Wikipedia XML を保存し、そこからテキストを抽出する 2 段階処理へ変更した
- JSON ダウンロードの並列化と再試行処理
- ontology TSV に path 名や rank 名キャッシュを追加
- 文書、音声、画像、埋め込みのメタデータスキーマ設計
- SQLite もしくは DuckDB を用いたメタデータ統合
