# README_graph

`graph 側` の README です。Bird (`Q5113`) 配下の ontology PKL を入力にして、親子分類関係を巨大な有向グラフとして扱います。ここで作る graph は、後続のグラフ埋め込み、サブグラフ抽出、近傍探索などの共通基盤になります。

## 概要

- `graph 側` の構築をまとめる
- 入力は `data/processed/bird_ontology.pkl`
- 各 QID を 1 ノードとして扱う
- `parent_taxon -> qid` を taxonomy の有向エッジとして扱う
- 各ノードには英語ラベルと分類属性を保持する
- 出力は `networkx.DiGraph` の PKL と Dash Cytoscape viewer に対応する

## 最初に読む場所

- [README.md](README.md)
  - プロジェクト全体の入口です
- [README_wikidata_pred.md](README_wikidata_pred.md)
  - `Wikidata 側` で `bird_ontology.pkl` を作る手順です
- [data/README.md](data/README.md)
  - graph を含むデータ配置と生成物の役割をまとめています
- [docs/architecture.md](docs/architecture.md)
  - `Wikidata 側` の前処理から `graph 側` の生成までの全体像をまとめています

## 何を作るのか

このモダリティでは、`bird_ontology.pkl` の各行をノードとして扱い、`parent_taxon` に記録された親 QID から自分自身の `qid` へ向かうエッジを張ります。

つまり、次のような対応です。

- ノード
  - `bird_ontology.pkl` の各要素
- ノード ID
  - 各要素の `qid`
- エッジ
  - `parent_taxon -> qid`
- エッジ種別
  - 現在は `parent_taxon` の 1 種類

この graph は taxonomy をそのまま表現するため、親分類から子分類・種へたどる探索や、上位分類単位でのサブグラフ切り出しに使えます。

## 入出力

入力:
- `data/processed/bird_ontology.pkl`

生成物:
- `data/processed/graph/bird_taxonomy_graph.pkl`

新しく作られるディレクトリ:
- `data/processed/graph/`
- `data/processed/graph/dash/`

## 事前準備

この README の処理を始める前に、少なくとも次が完了している必要があります。

1. `README_wikidata_pred.md` の手順に従って `bird_ontology.pkl` を生成する
2. `data/processed/bird_ontology.pkl` が存在することを確認する

`bird_ontology.pkl` が存在しない状態で `make build-graph` を実行すると失敗します。graph は ontology を直接入力として構築するためです。

## 実行

この節は 2 段に分かれています。

- 上段
  - 目的ごとに使うコマンドをすぐ確認するための早見表です
- 下段
  - 番号付きの詳細手順です。実際に上から順に実行する場合はこちらを見ます

### 実行コマンドの早見表

- taxonomy graph PKL を作りたい場合
  - `make build-graph`
- graph 埋め込みを個別に作りたい場合
  - `make build-node2vec-embeddings`
  - `make build-gcn-embeddings`
  - `make build-grac-embeddings`
  - `make build-transe-embeddings`
- まとめて設定を変えたい場合
  - `make build-embeddings EMBEDDING_ALGORITHM=<node2vec|gcn|grac|transe>`
- taxonomy graph をインタラクティブに観察したい場合
  - `make serve-graph`
- コードの構文確認だけしたい場合
  - `make verify`

### 詳細手順

### 0. 作業ディレクトリへ移動する

```bash
cd Multi_Bird_DB_Project
```

### 1. ontology PKL があることを確認する

入力は `data/processed/bird_ontology.pkl` です。

このファイルは graph 単体では作られません。前段の `Wikidata 側` の処理で生成される ontology 成果物です。未生成の場合は、先に [README_wikidata_pred.md](README_wikidata_pred.md) の手順に従って `make build-ontology` まで実行してください。

### 2. taxonomy graph PKL を作る

```bash
make build-graph
```

前提ファイル:
- `data/processed/bird_ontology.pkl`

`bird_ontology.pkl` がまだない場合は、先に [README_wikidata_pred.md](README_wikidata_pred.md) の `make build-ontology` を実行してください。

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/graph.py](src/multi_bird_db/graph.py)

生成物:
- `data/processed/graph/bird_taxonomy_graph.pkl`

このコマンドは `bird_ontology.pkl` を読み、`parent_taxon -> qid` の有向エッジを持つ graph を構築して保存します。

### 3. graph 埋め込みを作る

```bash
make build-embeddings
```

前提ファイル:
- `data/processed/graph/bird_taxonomy_graph.pkl`

`bird_taxonomy_graph.pkl` がまだない場合は、先にこの README の `make build-graph` を実行してください。

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/embeddings.py](src/multi_bird_db/embeddings.py)

生成物:
- `data/external/embeddings/graph/node2vec/`
- `data/external/embeddings/graph/gcn/`
- `data/external/embeddings/graph/grac/`
- `data/external/embeddings/graph/transe/`
- `data/external/embeddings/graph/hgcn/`

このコマンドは graph PKL を入力にして、`qid` をキーに参照できる埋め込みを保存します。`node2vec` は walk ベース、`gcn` は平滑化ベース、`grac` は attention 付き平滑化、`transe` は知識グラフ埋め込みとして学習します。

`node2vec` の公式実装の既定値は概ね `dimensions=128`、`walk_length=80`、`num_walks=10`、`p=1`、`q=1`、`workers=1` です。  
このリポジトリの `node2vec` は自前実装で、今の既定値は `dim=64`、`walk_length=20`、`num_walks=10`、`window_size=5`、`negative_samples=5`、`epochs=2` になっています。重く感じる場合は `walk_length`、`num_walks`、`negative_samples`、`epochs` をまず下げるのが効果的です。

### 4. taxonomy graph を Dash Cytoscape で観察する

```bash
make serve-graph
```

前提ファイル:
- `data/processed/graph/bird_taxonomy_graph.pkl`

`bird_taxonomy_graph.pkl` がまだない場合は、先にこの README の `make build-graph` を実行してください。

実行される処理:
- [src/multi_bird_db/cli.py](src/multi_bird_db/cli.py)
- [src/multi_bird_db/graph_dash.py](src/multi_bird_db/graph_dash.py)

このコマンドはローカルで Dash サーバを起動し、`http://127.0.0.1:8050` で taxonomy graph の部分グラフをインタラクティブに観察できるようにします。

viewer の特徴:
- `root_qid`、`max_depth`、`max_nodes` をその場で変更できる
- 表示ラベルは英語論文用途に合わせて `label_en` を使う
- taxon rank ごとに色分けする
- 右側パネルで色の凡例を確認できる
- ノードをクリックすると右側のパネルに QID、rank、学名、URL を表示する

5 万件規模を一度にブラウザへ出すのは重いため、viewer は常に部分グラフだけを表示します。デフォルトでは `Q5113` を根にして深さ 2、最大 150 ノードまでを描画します。

## graph PKL の構造

トップレベルは `networkx.DiGraph` です。graph metadata と node / edge attributes を持ちます。

graph metadata:
- `graph.graph["graph_type"]`
  - 現在は `taxonomy_digraph`
- `graph.graph["root_qid"]`
  - ルートとして扱う QID です。デフォルトは `Q5113`
- `graph.graph["node_count"]`
  - graph に含まれるノード数です
- `graph.graph["edge_count"]`
  - graph に含まれるエッジ数です

各ノードの主な属性:
- `label_en`
  - 可視化で使う英語ラベルです
- `label_ja`
  - 日本語ラベルです
- `en_name`
  - 英語名です
- `ja_name`
  - 日本語名です
- `taxon_name`
  - 学名です
- `taxon_rank`
  - taxon rank の QID です
- `taxon_rank_name`
  - taxon rank の英語名です
- `parent_taxon`
  - 親分類の QID です
- `entity_url`
  - Wikidata エンティティ URL です
- `enwiki_url`
  - 英語版 Wikipedia URL です
- `jawiki_url`
  - 日本語版 Wikipedia URL です

各エッジの主な属性:
- `relation`
  - 現在は `parent_taxon` です

## taxon_rank メモ

可視化では `taxon_rank` の QID を色分けキーとして使います。対応表は [docs/taxon_rank_mapping.md](docs/taxon_rank_mapping.md) に置いてあります。

## 実装上の扱い

現在の graph は `networkx.DiGraph` として保存しています。これにより、次のような利点があります。

- 親子関係の探索をそのまま実装できる
- `successors`、`predecessors`、subgraph 抽出などを直接使える
- 可視化コードと分析コードで同じ graph を共有できる

可視化では英語論文用途を前提に、主ラベルとして `label_en` を使います。`label_ja` は保持しますが、デフォルト表示には出しません。

Dash viewer でも同様に、ノード上の主ラベルは `label_en` を使います。`label_ja` は内部メタデータとして残しますが、既定の表示には使いません。

必要であれば、将来的にこの `networkx.DiGraph` を入力にして別形式の graph 表現を追加できます。たとえば edge list、埋め込み計算用のテンソル形式、論文図用の簡略サブグラフなどです。

## 関連箇所

- [README.md](README.md)
  - 全体入口です
- [README_wikidata_pred.md](README_wikidata_pred.md)
  - `Wikidata 側` で ontology PKL までを作る手順です
- [data/README.md](data/README.md)
  - データ配置ルールと生成物の説明です
- [docs/architecture.md](docs/architecture.md)
  - 処理全体の流れです
