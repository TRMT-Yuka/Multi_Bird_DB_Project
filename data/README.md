# Data Layout

このディレクトリはプロジェクト内でデータを扱うための作業領域です。既存のデータ運用ルールに従い、プロジェクト内で完結する配置にしています。

## ディレクトリ構成

```text
data/                                     # このプロジェクトで扱う全データのルート
├── raw/                                  # 外部から取得した未加工データ
│   └── wikidata/                         # Wikidata の入力領域
│       └── query.tsv                     # Bird 配下の entity URL 一覧
├── interim/                              # 処理途中で生成される中間データ
│   └── wikidata/                         # Wikidata の処理作業領域
│       ├── bird_qids.tsv                 # QID 抽出結果
│       ├── fetch_entities.sh             # entity JSON 取得用スクリプト
│       └── json/                         # QID ごとの JSON 応答
├── processed/                            # 下流利用向けに整形済みの成果物
│   ├── bird_ontology.tsv                 # 鳥類 ontology 統合表
│   └── wikipedia_article_manifest.tsv    # Wikipedia 取得対象一覧
└── external/                             # 文書・音声・画像・埋め込みの保管先
    ├── documents/                        # 文書コーパス
    ├── audio/                            # 音声ファイルや音声メタデータ
    ├── images/                           # 画像と画像メタデータ
    └── embeddings/                       # グラフ・テキスト・画像埋め込み
```

## 運用メモ

- 主キーは Bird の共通 ID である `bid` または Wikidata の `qid` を基準にする
- 埋め込み本体と属性テーブルは分離する
- 生成条件、次元数、モデル名、生成日時などのメタ情報を残す
- 大容量データは Git に直接入れず、配置先と取得手順だけを記録する
