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

## 事前準備

この README の処理を始める前に、少なくとも次が完了している必要があります。

1. `README_wikidata_pred.md` の手順に従って `bird_ontology.pkl` を生成する
2. `data/processed/bird_ontology.pkl` が存在することを確認する

`bird_ontology.pkl` が存在しない状態で `make build-graph` を実行すると失敗します。graph は ontology を直接入力として構築するためです。

## 実行

この節は、graph の構築、埋め込み、評価を最小限の手順で回すための実験仕様です。  
graph 埋め込みは **Node2Vec / GCN / GRACE / GraphSAGE / TransE** の 5 手法を同一条件で比較します。  
音声特徴、テキスト特徴、マルチモーダル初期特徴は使わず、**graph 構造だけ**を入力にします。  
`TransE` の relation は `parent_taxon` です。`graph` の向きは設定で切り替えられますが、既定は **有向** です。  
真のラベルは学習には使わず、**クラスタリング評価と report 出力のみに使います**。

### 実行コマンドの早見表

- taxonomy graph PKL を作りたい場合
  - `make build-graph`
- taxonomy graph をインタラクティブに観察したい場合
  - `make serve-graph`
  
- graph 埋め込みを個別に作りたい場合
  - `make build-node2vec-embeddings`
  - `make build-gcn-embeddings`
  - `make build-grace-embeddings`
  - `make build-graphsage-embeddings`
  - `make build-transe-embeddings`
- まとめて設定を変えたい場合
  - `make build-embeddings EMBEDDING_ALGORITHM=<node2vec|gcn|grace|transe|graphsage>`
- graph 埋め込みのクラスタリング評価とレポートを作りたい場合
  - `make evaluate-graph-embeddings`
- コードの構文確認だけしたい場合
  - `make verify`

### 詳細手順

#### 0. 作業ディレクトリへ移動する

```bash
cd Multi_Bird_DB_Project
```

#### 1. ontology PKL を作る

`make build-graph` の前提は `data/processed/bird_ontology.pkl` です。未生成なら先に [README_wikidata_pred.md](README_wikidata_pred.md) の手順で ontology を作成してください。

#### 2. taxonomy graph PKL を作る

```bash
make build-graph
```

入力: `data/processed/bird_ontology.pkl`  
出力: `data/processed/graph/bird_taxonomy_graph.pkl`

### 3. taxonomy graph を Dash Cytoscape で観察する

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


#### 4. graph 埋め込みを作る

```bash
make build-embeddings
```

入力: `data/processed/graph/bird_taxonomy_graph.pkl`  
出力:
- `data/external/embeddings/graph/node2vec/<MMDDhhmm>/`
- `data/external/embeddings/graph/gcn/<MMDDhhmm>/`
- `data/external/embeddings/graph/grace/<MMDDhhmm>/`
- `data/external/embeddings/graph/graphsage/<MMDDhhmm>/`
- `data/external/embeddings/graph/transe/<MMDDhhmm>/`

このリポジトリでは、`node2vec`、`gcn`、`grace`、`graphsage`、`transe` を同一 graph から学習します。`grace` は公式 GRACE に沿った contrastive learning、`gcn` は自己教師あり graph autoencoder、`graphsage` は mean-aggregator + neighbor sampling による自己教師あり negative sampling、`transe` は relation=`parent_taxon` の knowledge graph embedding です。`gcn` は encoder + inner-product decoder です。`grace` では `gcn` を encoder として使います。`grace` は既定で CPU 実行です。CUDA を使う場合は `--device cuda` を明示してください。

学習ログは各手法の `metadata.json` の `training_trace` に入ります。各実行は `data/external/embeddings/graph/<method>/<MMDDhhmm>/` に保存されます。

#### 実験パラメタ

以下は現在の最終既定値です。`make build-*-embeddings` はこの値をベースに動きます。

- `node2vec`
  - 既定値: `dim=128`, `walk_length=40`, `num_walks=10`, `window_size=10`, `negative_samples=5`, `epochs=200`, `learning_rate=0.001`, `p=1.0`, `q=1.0`, `seed=42`, `undirected=False`
- `gcn`
  - 既定値: `dim=128`, `layers=1`, `residual=0.0`, `epochs=300`, `learning_rate=0.01`, `negative_samples=20`, `feature_mode=degree`, `weight_decay=0.0`, `seed=42`, `root_qid=None`, `undirected=False`
- `grace`
  - 既定値: `dim=128`, `proj_dim=128`, `layers=2`, `residual=0.0`, `epochs=200`, `learning_rate=0.001`, `tau=0.5`, `drop_edge_rate_1=0.2`, `drop_edge_rate_2=0.4`, `drop_feature_rate_1=0.0`, `drop_feature_rate_2=0.0`, `batch_size=256`, `encoder_type=gcn`, `feature_mode=degree`, `weight_decay=1e-5`, `device=cpu`, `seed=42`, `root_qid=None`, `undirected=False`
- `graphsage`
  - 既定値: `dim=128`, `layers=2`, `residual=0.0`, `epochs=200`, `learning_rate=0.001`, `negative_samples=5`, `num_neighbors=[25,10]`, `feature_mode=degree`, `seed=42`, `root_qid=None`, `undirected=False`
- `transe`
  - 既定値: `dim=128`, `epochs=200`, `learning_rate=0.001`, `margin=1.0`, `negative_samples=10`, `p_norm=1`, `weight_decay=1e-5`, `seed=42`, `root_qid=None`
  - `p_norm=1` なので L1 距離版として扱う

#### 公式リンク

- `node2vec`: [論文](https://arxiv.org/abs/1607.00653), [実装](https://github.com/eliorc/node2vec)
- `gcn`: [論文](https://arxiv.org/abs/1609.02907), [実装](https://github.com/tkipf/gcn)
- `grace`: [論文](https://arxiv.org/abs/2006.04131), [実装](https://github.com/CRIPAC-DIG/GRACE)
- `graphsage`: [論文](https://arxiv.org/abs/1706.02216), [実装](https://github.com/williamleif/GraphSAGE)
- `transe`: [論文](https://papers.nips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)



### 5. graph 埋め込みをクラスタリング評価する

```bash
make evaluate-graph-embeddings
```

前提:
- `data/processed/graph/bird_taxonomy_graph.pkl`
- `data/external/embeddings/graph/<method>/<MMDDhhmm>/embeddings.npy`

評価:
- `k-means` の `k` は真のラベル数に合わせる
- 真のラベルは既定で `taxon_rank_name`
- 指標は `NMI`, `ARI`, `Purity`, `Homogeneity`, `Completeness`, `V-measure`, `Silhouette`

生成物:
- `data/external/embeddings/graph/evaluation/metrics/clustering_metrics.csv`
- `data/external/embeddings/graph/evaluation/metrics/summary_metrics.csv`
- `data/external/embeddings/graph/evaluation/plots/clustering_metrics_barplot.png`
- `data/external/embeddings/graph/evaluation/logs/<method>_cluster_assignments.tsv`
- `data/external/embeddings/graph/evaluation/report/experiment_report.md`
- `data/external/embeddings/graph/evaluation/report/experiment_report.json`

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
