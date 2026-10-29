# Research Workflow

この文書は、本プロジェクトのデータ取得、前処理、埋込み生成、実験、評価、成果物管理までを一つの流れとして説明する全体ワークフローです。個別の実装詳細が必要な場合だけ、末尾の関連READMEやPythonスクリプトを参照してください。

## 1. 研究目的

Wikidataに基づく分類学的知識グラフ、英語・日本語の名称情報、Xeno-canto等の鳥類音声から埋込みを作成し、以下を評価します。

- Experiment 1：各モダリティの埋込み空間における類似構造の比較
- Experiment 2：グラフ・言語・音声の類似関係を統合したグラフの可視化
- Experiment 3：埋込みから音声または分類学的グループを検索できるかの評価

## 2. 全体フロー

    Wikidata Query Service
            │ bird_descendants.sparql
            ▼
    data/raw/wikidata/query.tsv
            │ QID抽出
            ▼
    data/interim/wikidata/bird_qids.tsv
            │ dump取得・対象JSON抽出
            ▼
    data/interim/wikidata/json/...
            │ ontology構築
            ▼
    data/processed/bird_ontology.pkl
            │
            ├─ taxonomy graph ── graph embeddings
            ├─ language surface ─ language embeddings
            └─ Xeno-canto ────── audio embeddings
                             │
                             ▼
            data/external/embeddings/selected_runs.json
                             │
                             ▼
                Experiment 0：共通QID確認
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Experiment 1   Experiment 2   Experiment 3
        類似度行列      統合グラフ      検索・分類評価
                                          │
                                          ├─ Sub-experiment 1：音声検索
                                          └─ Taxon probe：分類学的検索

## 3. 作業環境

プロジェクトルート：

    cd /data5/teramoto/Multi_Bird_DB_Project

- Wikidata、グラフ、言語処理：通常のプロジェクトPython環境
- wav2vec2：bird_env
- BirdNET：birdnet
- Perch：別環境で埋込みを作成し、生成物だけを持ち込む

再現時にはPython環境、依存パッケージ、CUDAの有無、Git commit IDを記録します。

## 4. Wikidata前処理

### 4.1 対象QIDの取得

queries/bird_descendants.sparqlをWikidata Query Serviceで実行し、鳥類全体を表すQID Q5113の子孫を取得します。

取得結果：

    data/raw/wikidata/query.tsv

クエリ、取得日、取得件数は記録してください。

### 4.2 QID抽出

    make extract-qids

入力：data/raw/wikidata/query.tsv  
出力：data/interim/wikidata/bird_qids.tsv

### 4.3 Wikidata dump取得

    make download-wikidata-dump

出力：

    data/raw/wikidata/dumps/latest-all.json.bz2

dumpは非常に大きいため、取得時点とディスク容量を管理します。

### 4.4 対象JSON抽出

    make extract-dump-json

入力はQID一覧とWikidata dump、出力は次です。

    data/interim/wikidata/json/<QID階層>/Qxxxx.json

既存JSONは再利用されるため、途中停止後に再開できます。

### 4.5 ontology構築

    make build-ontology
    make extract-xeno-canto-ids

主な出力：

    data/processed/bird_ontology.pkl
    data/interim/wikidata/bird_xeno_canto_ids.tsv

ontologyにはQID、英語・日本語ラベル、分類ランク、parent_taxon、URL、Xeno-canto species ID等が含まれます。

## 5. 分類学グラフ

### 5.1 グラフ構築

    make build-graph

入力：data/processed/bird_ontology.pkl  
出力：data/processed/graph/bird_taxonomy_graph.pkl

グラフは有向グラフで、エッジ方向は次です。

    親分類群 --parent_taxon--> 子分類群

ルートは原則として鳥類QID Q5113です。ノードにはQID、名称、分類ランク、URL、親分類群等を保持します。

### 5.2 グラフ確認と評価ラベル

    make serve-graph
    make build-taxon-labels

グラフviewerは構造確認用です。taxon labelsはExperiment 3の正解ラベル用であり、埋込み学習の入力ではありません。

主なラベル出力：

    data/processed/taxonomy/qid_taxon_labels.tsv
    data/processed/taxonomy/qid_taxon_labels.json
    data/processed/taxonomy/qid_taxon_labels.meta.json

## 6. グラフ埋込み

### 6.1 共通設定

    output_dim: 128
    hidden_dim: 32                 # 畳み込み系
    layers: 2                      # 畳み込み系
    epochs: 200
    weight_decay: 1e-5
    seed: 42

GCN、GraphSAGE、GRACEの初期特徴は現在 embeddingが既定値です。one_hot、constant、randomは使用対象から削除されています。hidden_dimは内部表現、output_dimは保存される最終埋込みの次元です。

### 6.2 Node2Vec

    make build-node2vec-embeddings

主な設定：

    output_dim: 128, walk_length: 40, num_walks: 10, window_size: 10
    p: 1.0, q: 1.0, negative_samples: 5
    epochs: 200, learning_rate: 0.001, weight_decay: 1e-5
    undirected: false, seed: 42, batch_size: 8192

Node2VecはランダムウォークとSkip-gram型学習を使用します。バッチ処理で高速化されていますが、途中checkpointからの再開には対応していません。

### 6.3 GCN

    make build-gcn-embeddings

    hidden_dim: 32, output_dim: 128, layers: 2, epochs: 200
    learning_rate: 0.01, negative_samples: 20
    loss: gae_negative_sampling, weight_decay: 1e-5
    initial_features: embedding

### 6.4 GraphSAGE

    make build-graphsage-embeddings

    hidden_dim: 32, output_dim: 128, layers: 2, epochs: 200
    learning_rate: 0.001, aggregator: mean
    num_neighbors: [25, 10], negative_samples: 1
    weight_decay: 1e-5, initial_features: embedding, device: cuda

### 6.5 GRACE

    make build-grace-embeddings

    hidden_dim: 32, output_dim: 128, projection_dim: 128
    layers: 2, epochs: 200, learning_rate: 0.001, tau: 0.5
    edge_drop: [0.2, 0.4], feature_mask: [0.0, 0.0]
    weight_decay: 1e-5, initial_features: embedding, device: cpu

GRACEはグラフの2つの拡張ビューを使う対照学習です。現在の既定値はCPUです。

### 6.6 TransE

    make build-transe-embeddings

    output_dim: 128, epochs: 200, learning_rate: 0.001
    margin: 1.0, p_norm: 1, negative_samples: 10
    weight_decay: 1e-5, relation: parent_taxon

### 6.7 グラフ埋込み成果物

各runは原則として次に保存されます。

    data/external/embeddings/graph/<method>/<run_id>/

最低限、次を確認します。

    embeddings.npy
    qids.json
    metadata.json

QIDの順番とembeddings.npyの行順が一致していることが重要です。

## 7. 言語埋込み

    make build-language-surface-manifest
    make build-language-embeddings

使用モデル：

- 英語：google-bert/bert-base-uncased
- 日本語：tohoku-nlp/bert-base-japanese-v3

出力：

    data/external/embeddings/language/en/
    data/external/embeddings/language/ja/

各言語ディレクトリにはsurface_manifest.tsv、surface_ids.json、qids.json、qid_to_surfaces.json、embeddings.npy、metadata.json、summary.jsonが保存されます。

1つのQIDに複数の表層形がある場合、現実装ではマニフェスト上で先に現れる表層形をQIDの代表値として使用します。

## 8. 音声データと音声埋込み

### 8.1 録音取得

    make fetch-xeno-canto-recording-json
    make extract-xeno-canto-recording-ids
    make fetch-xeno-canto-audio

入力はdata/interim/wikidata/bird_xeno_canto_ids.tsv、音声出力は次です。

    data/raw/xeno-canto/<qid>/

### 8.2 埋込み生成

wav2vec2：

    make download-audio-models
    make build-audio-embeddings-wav2vec2

BirdNET：

    make check-birdnet-gpu
    make build-audio-embeddings-birdnet-gpu

Perchは別環境で生成した成果物を持ち込みます。

出力形式：

    data/external/embeddings/audio/<backend>/<model>/<run_id>/

embeddings.npy、audio_manifest.tsv、audio_ids.json、qids.jsonの行対応を維持します。複数の時間窓がある場合、実験時に同一音声ファイル内の窓を平均して1ベクトルにします。

## 9. 使用runの確定

実験で使用する具体的な埋込みrunは次で管理します。

    data/external/embeddings/selected_runs.json

新しい埋込みを作っただけでは、必ずしも実験には使用されません。実験前にこのファイルを更新し、使用するrunのパスとmetadataを確認します。

    make inspect-multimodal-sources

確認項目：

- パスと必須ファイルの存在
- QID数と埋込み行数
- metadata
- QIDとベクトルの行対応

selected runを使う処理と、自動探索でcomplete runを使う処理があります。自動探索は「最新の1件だけを使う」という意味ではありません。

## 10. Experiment 0：共通データ確認

### 10.1 QID被覆率

    make report-qid-coverage

出力：

    experiments/exp1/qid_coverage/run_qid_counts.tsv
    experiments/exp1/qid_coverage/common_qids.tsv
    experiments/exp1/qid_coverage/summary.json

### 10.2 録音数集計

    make collect-xeno-canto-num-recordings

出力：experiments/exp1/xeno_canto_num_recordings/

この情報は、Experiment 1の録音数順版とExperiment 2の対象QID選定で使用します。

## 11. Experiment 1：類似度行列

### 11.1 通常版

    make run-exp1-simmatrix

埋込みを読み込み、共通QIDの交差集合を作り、QID順を統一してコサイン類似度行列を計算します。

出力：

    experiments/exp1/similarity_matrices/

主な成果物：

    <run>_similarity.npy
    <run>_heatmap.png
    <run>_distribution.png
    matrix_qids.json
    embedding_run_summary.tsv
    representative_audio_files.tsv
    metadata.json

分類器の学習ではなく、各埋込み空間の類似構造を比較する分析です。

### 11.2 録音数順版

    make run-exp1-simmatrix-ordered

Xeno-canto録音数の多いQIDを優先し、通常は上位100件を対象にします。同数の場合はQID順です。通常版とは別条件として報告します。

出力：experiments/exp1/similarity_matrices_ordered/

## 12. Experiment 2：統合埋込みグラフ

    make run-exp2-integrated-graph

現在の標準設定：

    Graph: Node2Vec
    Language: Japanese
    Audio: Perch

共通QIDを決定し、録音数上位の対象を選び、各モダリティのコサイン類似度行列を作成します。各モダリティの類似度上位5%をエッジとして抽出し、3つのグラフを統合して可視化します。

出力：

    experiments/exp2/integrated_embedding_graphs/

主な成果物はgraph.svg、language.svg、audio.svg、integrated.svg、metadata.jsonです。これは既存埋込みから類似グラフを作る可視化であり、新しい埋込みモデルを学習する処理ではありません。

## 13. Experiment 3：検索・評価

### 13.1 Sub-experiment 1：音声検索

    make run-exp3-sub1-audio

音声をクエリにし、同じQIDの別音声を正解として検索性能を評価します。クエリ自身は候補から除外します。

評価指標：

- mAP
- MRR
- nDCG@10、nDCG@50、nDCG@100

出力：experiments/exp3/sub1_audio/

### 13.2 Taxon probe：分類学的検索

事前に次を実行します。

    make build-taxon-labels

実行：

    make run-exp3-search

現在のMakefileでは、このコマンドはexperiments/exp3_taxon_probe.pyを実行します。

モダリティ組合せ：

    G, L, A, GL, GA, LA, GLA

埋込みを読み込み、指定モダリティを組み合わせ、QID間類似度から上位候補を検索します。family、order等の正解ラベルと比較します。

評価指標：

- Precision@1、@5、@10
- Recall@1、@5、@10
- F1@1、@5、@10
- MAP@10
- MRR

出力：experiments/exp3/taxon_probe/

主な成果物はmetrics.tsv、predictions.tsv、metadata.jsonです。

### 13.3 結果集約

    python3 experiments/collect_exp3_taxon_probe_results.py

出力：experiments/exp3/taxon_probe_collected/

## 14. 実行コマンド一覧

    # 前処理
    make extract-qids
    make download-wikidata-dump
    make extract-dump-json
    make build-ontology
    make extract-xeno-canto-ids

    # グラフ
    make build-graph
    make build-taxon-labels

    # グラフ埋込み
    make build-node2vec-embeddings
    make build-gcn-embeddings
    make build-grace-embeddings
    make build-graphsage-embeddings
    make build-transe-embeddings

    # 言語埋込み
    make build-language-surface-manifest
    make build-language-embeddings

    # 音声
    make fetch-xeno-canto-recording-json
    make extract-xeno-canto-recording-ids
    make fetch-xeno-canto-audio
    make download-audio-models
    make build-audio-embeddings-wav2vec2

    # 実験前
    make inspect-multimodal-sources
    make report-qid-coverage
    make collect-xeno-canto-num-recordings

    # Experiment 1～3
    make run-exp1-simmatrix
    make run-exp1-simmatrix-ordered
    make run-exp2-integrated-graph
    make run-exp3-sub1-audio
    make run-exp3-search

## 15. 成果物の保存場所

| 種類 | 保存場所 |
|---|---|
| Wikidata取得結果 | data/raw/wikidata/ |
| 中間QID・JSON | data/interim/wikidata/ |
| ontology | data/processed/bird_ontology.pkl |
| taxonomy graph | data/processed/graph/ |
| taxon評価ラベル | data/processed/taxonomy/ |
| グラフ埋込み | data/external/embeddings/graph/ |
| 言語埋込み | data/external/embeddings/language/ |
| 音声埋込み | data/external/embeddings/audio/ |
| 使用run | data/external/embeddings/selected_runs.json |
| Experiment 1 | experiments/exp1/ |
| Experiment 2 | experiments/exp2/ |
| Experiment 3 | experiments/exp3/ |

実験結果ディレクトリはGit管理対象外です。論文用の最終結果・図・表は別途バックアップしてください。

## 16. 再現性のために記録する情報

### データ条件

- Wikidataクエリ、dump取得日
- Xeno-canto API取得日
- 対象QID数、共通QID数
- 録音数、欠損数

### モデル条件

- 手法名、モデル名
- hidden/output dimension
- 層数、epoch数、学習率
- weight decay、negative samples
- 初期特徴量、seed、CPU/GPU
- 使用runのパス

### 実験条件

- 使用モダリティ
- 使用QID集合
- top-k
- 録音数による選択条件
- 類似度閾値
- 対象分類ランク
- 評価指標
- 実行日時、Git commit ID

## 17. 現在の実装上の注意点

1. selected_runs.jsonを使う処理と、自動探索でcomplete runを使う処理があります。自動探索は必ずしも最新1件だけを選びません。
2. make run-exp3-searchは現在taxon probeを実行します。一方、experiments/exp3_search_performance.pyも存在するため、論文では両者の関係を明記します。
3. 同じ既定出力先で再実行すると、既存結果が上書きまたは混在する可能性があります。比較実験では専用出力先を使います。
4. Node2Vecは途中checkpointから再開できません。
5. GraphSAGEのMakefile既定値はCUDA、GRACEの既定値はCPUです。
6. 類似度行列はQID数に対して二次的に大きくなるため、メモリとディスク容量を確認します。

## 18. 承認ポイントと責任分界

| 段階 | 承認する成果物 | 確認事項 |
|---|---|---|
| Wikidata | query、dump、QID一覧 | 取得日、クエリ、対象範囲 |
| ontology | bird_ontology.pkl | QID、名称、親分類、欠損 |
| graph | bird_taxonomy_graph.pkl | 方向、ルート、ノード数、エッジ数 |
| 埋込み | 各runのmetadata | パラメータ、seed、device、行対応 |
| run選択 | selected_runs.json | 論文使用runの正しさ |
| 共通データ | common_qids.tsv | 比較対象の妥当性 |
| Experiment 1 | 行列、図、metadata | QID順、対象run、欠損 |
| Experiment 2 | SVG、metadata | モダリティ、閾値、対象QID |
| Experiment 3 | metrics、predictions | ラベル、top-k、指標 |

## 19. 関連ファイル

- [README.md](README.md)
- [README_wikidata_pred.md](README_wikidata_pred.md)
- [README_graph.md](README_graph.md)
- [README_audio.md](README_audio.md)
- [README_language.md](README_language.md)
- [README_experiments.md](README_experiments.md)
- [Makefile](Makefile)
- [selected_runs.json](data/external/embeddings/selected_runs.json)
- [Experiment 0](experiments/exp0_qid_coverage.py)
- [Experiment 1](experiments/exp1_similarity_matrices.py)
- [Experiment 1 ordered](experiments/exp1_similarity_matrices_ordered.py)
- [Experiment 2](experiments/exp2_integrated_embedding_graphs.py)
- [Experiment 3 Sub-experiment 1](experiments/exp3_sub1_audio_pretraining.py)
- [Experiment 3 taxon probe](experiments/exp3_taxon_probe.py)
- [Experiment 3 search performance](experiments/exp3_search_performance.py)
