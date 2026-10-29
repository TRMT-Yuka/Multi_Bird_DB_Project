# 研究処理フローチャート

以下の一枚のフローチャートだけで、同じ入力集合、埋込み、対象QID、類似度計算、検索評価、出力を再現できるように記載する。コマンド名やコード構造は記載せず、実行する処理と条件を記載する。

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 0. 対象範囲と入力の固定                                                       │
│                                                                              │
│ 鳥類全体を表すQID Q5113を起点とし、Q5113の下部に所属する全ノードを対象とする。 │
│                                                                              │
│ Wikidata Query Serviceで鳥類子孫を取得し、entity URLをTSVで保存する。          │
│ 入力：queries/bird_descendants.sparql                                          │
│ 出力：data/raw/wikidata/query.tsv                                             │
│                                                                              │
│ query.tsvのitem列からURL末尾のQIDを抽出する。重複を除去し、                   │
│ Q5113が存在しない場合はQ5113を先頭に追加する。                                │
│ 出力：data/interim/wikidata/bird_qids.tsv                                     │
│ 形式：qid列、1行1QID                                                         │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
┌────────────────────────────────┐    ┌────────────────────────────────────────┐
│ 1A. Wikidata dump取得           │    │ 1B. 事前作成済みの対象QIDリスト        │
│                                │    │                                        │
│ Wikidata entities dumpを取得   │    │ query.tsvのitem列から作成したQID一覧。 │
│                                │    │ Wikidataを再検索する工程ではない。     │
│                                │    │ bird_qids.tsv                         │
│                                │    │                                        │
│ 出力：                         │    │                                        │
│ data/raw/wikidata/dumps/       │    └──────────────────┬─────────────────────┘
│ latest-all.json.bz2            │                       │
└────────────────┬───────────────┘                       │
                 └──────────────────────┬────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. 対象QIDのentity JSON抽出                                                    │
│                                                                              │
│ dump全体からbird_qids.tsvに含まれるQIDだけを抽出する。                        │
│                                                                              │
│ 出力：                                                                       │
│ data/interim/wikidata/json/<QID数値部の1桁目>/<2桁目>/Qxxxx.json               │
│                                                                              │
│ QIDごとに1つのWikidata entity JSONを保存する。既存ファイルは再利用し、          │
│ 抽出済みファイルを上書きしない。                                              │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. ontology生成                                                               │
│                                                                              │
│ 1つのentity JSONを1つのontologyレコードへ変換し、全レコードをlistとして保存。  │
│                                                                              │
│ 各レコードに保持する情報：                                                    │
│ qid、entity_url、en_name、ja_name、en_aliases、ja_aliases、img_names、         │
│ xeno_canto_species_id、taxon_name、taxon_rank、taxon_rank_name、               │
│ parent_taxon、parent_taxon_name、parent_taxon_ja_name、                       │
│ enwiki_title、enwiki_url、jawiki_title、jawiki_url、path                       │
│                                                                              │
│ 出力：data/processed/bird_ontology.pkl                                        │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│ 4A. taxonomy graph      │ │ 4B. 言語表層形          │ │ 4C. Xeno-canto対応     │
│                        │ │                        │ │                        │
│ ontologyの各QIDを1ノード│ │ ontologyの各QIDから   │ │ ontologyから            │
│ とする。                │ │ 英語・日本語の表層形を │ │ xeno_canto_species_idを │
│                        │ │作成する。              │ │持つQIDを抽出する。      │
│ parent_taxonを持つ各行に│ │                        │ │                        │
│対して、親から子へ1辺を  │ │ 英語候補：en_name、    │ │ 出力：                 │
│作る。                   │ │ enwiki_title            │ │ data/interim/wikidata/ │
│                        │ │ 日本語候補：ja_name、  │ │ bird_xeno_canto_ids.tsv │
│ 出力：                 │ │ jawiki_title            │ │                        │
│ data/processed/graph/  │ │                        │ └────────────┬───────────┘
│ bird_taxonomy_graph.pkl│ │ surface_idは          │              │
│                        │ │ <QID>_<lang>_<ordinal> │              ▼
│ 辺方向：               │ │                        │ ┌────────────────────────┐
│ parent_taxon → child   │ │ 出力：                 │ │ 5. 録音情報取得         │
│ 逆辺は追加しない。      │ │ language/en/           │ │                        │
│                        │ │ language/ja/           │ │ species IDに対応する   │
│ ノード属性：           │ │ surface_manifest.tsv   │ │ Xeno-canto APIのページ │
│ QID、英語名、日本語名、 │ │                        │ │ JSONを保存する。        │
│ taxon rank、parent等。  │ │ 表層形1件が1行。       │ │                        │
└───────────┬────────────┘ └────────────┬───────────┘ │ 出力：                 │
            │                           │             │ data/interim/          │
            │                           │             │ xeno-canto/            │
            │                           │             │ api_recordings/<QID>/  │
            │                           │             │ page*.json             │
            │                           │             │ api_recordings_manifest│
            │                           │             │ .json、summary.json   │
            │                           │             └────────────┬───────────┘
            │                           │                          ▼
            │                           │             ┌────────────────────────┐
            │                           │             │ 6. 録音ID対応表作成     │
            │                           │             │                        │
            │                           │             │ ページJSONから録音単位 │
            │                           │             │ の情報を抽出する。     │
            │                           │             │                        │
            │                           │             │ 出力：                 │
            │                           │             │ data/interim/xeno-     │
            │                           │             │ canto/recording_map.json│
            │                           │             └────────────┬───────────┘
            │                           │                          ▼
            │                           │             ┌────────────────────────┐
            │                           │             │ 7. 音声ファイル取得     │
            │                           │             │                        │
            │                           │             │ 録音単位で音声を保存。 │
            │                           │             │ 既存ファイルは再利用。 │
            │                           │             │ 品質A、末端QID、       │
            │                           │             │ species IDごと最大20件。│
            │                           │             │                        │
            │                           │             │ 出力：                 │
            │                           │             │ data/raw/xeno-canto/   │
            │                           │             │ <QID>/音声ファイル     │
            │                           │             │ existing_audio_manifest │
            │                           │             │ .json、取得summary     │
            │                           │             └────────────┬───────────┘
            │                           │                          ▼
            │                           │             ┌────────────────────────┐
            │                           │             │ 8. 音声窓の作成・埋込み │
            │                           │             │                        │
            │                           │             │ wav2vec2：ファイル全体、│
            │                           │             │ 16kHz                   │
            │                           │             │ BirdNET：3秒窓、48kHz   │
            │                           │             │ Perch：外部環境で生成。 │
            │                           │             │                        │
            │                           │             │ 1音声ファイルを複数窓に │
            │                           │             │ 分けた場合、窓ごとに1行 │
            │                           │             │ の埋込みを作る。        │
            │                           │             │                        │
            │                           │             │ 出力：                 │
            │                           │             │ data/external/embeddings│
            │                           │             │ /audio/<backend>/<model>│
            │                           │             │ /<run_id>/             │
            │                           │             └────────────┬───────────┘
            │                           │                          │
            ▼                           ▼                          │
┌────────────────────────┐ ┌────────────────────────┐             │
│ 9A. graph embedding     │ │ 9B. language embedding │             │
│                        │ │                        │             │
│ taxonomy graphを入力。  │ │ surface_manifest.tsvを  │             │
│ QID順：文字列昇順。     │ │ 入力。                  │             │
│                        │ │                        │             │
│ Node2Vec：             │ │ 英語：                  │             │
│ dim=128、walk_length=40、│ │ google-bert/           │             │
│ num_walks=10、window=10、│ │ bert-base-uncased       │             │
│ p=1、q=1、negative=5、  │ │                        │             │
│ epochs=200、lr=.001、   │ │ 日本語：                │             │
│ wd=1e-5、seed=42、      │ │ tohoku-nlp/            │             │
│ directed、batch=8192。  │ │ bert-base-japanese-v3  │             │
│                        │ │                        │             │
│ GCN：                  │ │ 表層形1件を1埋込み行に │             │
│ hidden=32、output=128、 │ │ 変換する。              │             │
│ layers=2、epochs=200、  │ │                        │             │
│ lr=.01、negative=20、   │ │ 出力：                 │             │
│ loss=GAE negative      │ │ data/external/         │             │
│ sampling、wd=1e-5、    │ │ embeddings/language/   │             │
│ initial=embedding。    │ │ <lang>/                │             │
│                        │ │ embeddings.npy、       │             │
│ GraphSAGE：            │ │ surface_ids.json、     │             │
│ hidden=32、output=128、 │ │ qids.json、            │             │
│ layers=2、epochs=200、  │ │ qid_to_surfaces.json、 │             │
│ lr=.001、mean、         │ │ metadata.json、        │             │
│ neighbors=[25,10]、    │ │ summary.json           │             │
│ negative=1、wd=1e-5、  │ └────────────┬───────────┘             │
│ initial=embedding、    │              │                         │
│ device=cuda。          │              │                         │
│                        │              │                         │
│ GRACE：                │              │                         │
│ hidden=32、output=128、 │              │                         │
│ projection=128、       │              │                         │
│ layers=2、epochs=200、  │              │                         │
│ lr=.001、tau=.5、       │              │                         │
│ edge_drop=[.2,.4]、     │              │                         │
│ feature_mask=[0,0]、   │              │                         │
│ wd=1e-5、encoder=GCN、  │              │                         │
│ initial=embedding。    │              │                         │
│                        │              │                         │
│ TransE：               │              │                         │
│ output=128、epochs=200、│              │                         │
│ lr=.001、margin=1、     │              │                         │
│ p_norm=1、negative=10、 │              │                         │
│ wd=1e-5、relation=      │              │                         │
│ parent_taxon。         │              │                         │
│                        │              │                         │
│ 共通出力：             │              │                         │
│ embeddings.npy、       │              │                         │
│ qids.json、metadata.json│             │                         │
│                        │              │                         │
│ initial=embeddingは    │              │                         │
│ 固定one-hotではなく、  │              │                         │
│ 学習可能なnode embedding│             │                         │
│ tableを使う。          │              │                         │
└───────────┬────────────┘              │                         │
            └──────────────────────────┴─────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 10. 埋込み行の対応付けとrunの選択                                             │
│                                                                              │
│ graph：embeddings.npy[i] = qids.json[i]のQIDのベクトル                       │
│ language：embeddings.npy[i] = surface_ids.json[i] = qids.json[i]              │
│ audio：embeddings.npy[i] = audio_manifest[i] = audio_ids[i] = qids[i]         │
│                                                                              │
│ audio_manifestの複数窓行は、同じ(QID, relative_path)ごとに平均して              │
│ 1録音ベクトルへ集約する。                                                     │
│                                                                              │
│ 使用する具体的なrunを                                                        │
│ data/external/embeddings/selected_runs.json                                  │
│ にgraph、language、audioのlabelとrun pathとして固定する。                     │
│                                                                              │
│ 新しいrunを作成しても、selected_runs.jsonを更新しない限り、                    │
│ selected runを読む実験の入力は変わらない。                                    │
│ selected_runs.jsonが存在する場合はそれを優先し、存在しない場合は各実験の        │
│ auto-discover設定によりcomplete fileを持つrunを探索する。                     │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 11. 共通QID集合と録音数の作成                                                 │
│                                                                              │
│ 11A. runごとのQID集合                                                         │
│ selected_runsに登録された各runのQIDを読む。                                  │
│ 各モダリティ内でQID集合を交差し、さらにgraph・language・audioの                 │
│ モダリティ間で交差する。                                                      │
│                                                                              │
│ 出力：                                                                       │
│ experiments/exp1/qid_coverage/run_qid_counts.tsv                             │
│   modality、label、path、qid_count、complete_qid_count                      │
│ experiments/exp1/qid_coverage/common_qids.tsv                                │
│   qid列、1行1共通QID                                                          │
│ experiments/exp1/qid_coverage/summary.json                                   │
│                                                                              │
│ 11B. 録音数                                                                   │
│ 各QIDのapi_recordings/<QID>/page001.jsonからnumRecordingsを読む。               │
│                                                                              │
│ 出力：                                                                       │
│ qid_num_recordings.tsv：qid、xeno_canto_species_id、ja_name、                 │
│   num_recordings、num_pages、page001_recordings、page001_path                  │
│ summary.json、missing_page001_qids.tsv、分布PNG                              │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│ 12A. Experiment 1通常版│ │ 12B. Experiment 1順序版 │ │ 12C. Experiment 2       │
│                        │ │                        │ │                        │
│ 対象：selected runに   │ │ 対象：common_qids.tsvに│ │ 対象：common_qids.tsvに │
│ 共通する全QID。        │ │ あり、録音数TSVにも存在 │ │ あり、選択したgraph、    │
│                        │ │ するQID。              │ │ language、audioの3runに │
│ QIDごとの代表値：     │ │                        │ │ 共通するQID。           │
│ graph：QIDのベクトル   │ │ num_recordings降順、    │ │                        │
│ language：surfaceの    │ │ 同数はQID昇順で並べ、   │ │ num_recordings降順、    │
│ ordinal/surface_id順の │ │ top_k=100を選ぶ。      │ │ 同数はQID昇順で並べ、   │
│ 先頭                  │ │                        │ │ top_k=100を選ぶ。      │
│ audio：全audio runに   │ │ 各runに選択QIDがない   │ │                        │
│ 共通するrelative_path  │ │ 場合は処理を停止する。  │ │                        │
│ の辞書順先頭。窓は平均。│ │                        │ │                        │
│                        │ │ 類似度行列を作り、      │ │ 各モダリティで以下を行う。│
│ 各runのベクトル行をL2  │ │ heatmapとdistributionを│ │                        │
│ 正規化し、次を計算：   │ │ 保存する。              │ │ X_mを行L2正規化、        │
│ S = X_norm × X_norm^T  │ │                        │ │ S_m = X_m X_m^T         │
│ 対角成分は1.0。        │ │ 出力：                 │ │                        │
│                        │ │ experiments/exp1/      │ │ 上三角要素の95 percentile│
│ 出力：                 │ │ similarity_matrices_   │ │ をt_mとし、             │
│ experiments/exp1/      │ │ ordered/<run_name>/   │ │                        │
│ similarity_matrices/   │ │ heatmap.png、          │ │ E_m={(i,j,S_mij) | i<j、│
│                        │ │ distribution.png       │ │ S_mij >= t_m}          │
│ <run>_similarity.npy   │ └────────────────────────┘ │                        │
│   float32、[N,N]      │                            │ graph、language、audio │
│ <run>_heatmap.png      │                            │ の各E_mを無向graphへ   │
│ <run>_distribution.png │                            │ 追加する。              │
│ matrix_qids.json       │                            │                        │
│ embedding_run_summary │                            │ 同じQID対が複数モダリティ│
│ .tsv                   │                            │ にある場合、modality集合、│
│ representative_audio  │                            │ modality別score、       │
│ _files.tsv             │                            │ weight=max(score)を保持。│
│ metadata.json          │                            │                        │
└────────────────────────┘                            │ union graphのweightの   │
                                                      │ 95 percentileを        │
                                                      │ union_thresholdとする。 │
                                                      │                        │
                                                      │ spring_layout(seed=42、 │
                                                      │ weight、k=.85、         │
                                                      │ iterations=200)で描画。 │
                                                      │                        │
                                                      │ 出力：                 │
                                                      │ experiments/exp2/      │
                                                      │ integrated_embedding_  │
                                                      │ graphs/G_<g>__L_<l>__  │
                                                      │ A_<a>/                 │
                                                      │ graph.svg、language.svg│
                                                      │ audio.svg、integrated   │
                                                      │ .svg、metadata.json     │
└──────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────────────────┐
    │ 12A、12B、12Cは、selected runと共通QIDの準備後に独立して実行する。      │
    │ 12Bは12Aの画像・npyを入力しない。12Cも12A/12Bの出力を入力しない。        │
│ 13A、13Bも10/11のrun・labelを直接読み、12A～12Cの出力には依存しない。     │
    └────────────────────────────────────────────────────────────────────────┘

                                     │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────┐
│ 13A. Experiment 3 Sub-experiment 1  │        │ 13B. Experiment 3 Taxon probe    │
│ 音声検索                             │        │ 分類学的検索                       │
│                                      │        │                                  │
│ 入力：音声runのみ                    │        │ 入力：selected graph/language/   │
│                                      │        │ audio run、taxon labels           │
│ 1. window行を(QID, relative_path)   │        │                                  │
│    でグループ化                       │        │ 1. 各modalityの代表値を選ぶ       │
│ 2. グループ内の窓ベクトルを平均       │        │    G：QIDの1ベクトル               │
│ 3. 1録音を1検索itemにする             │        │    L：QID内でqid、ordinal、       │
│ 4. 全録音itemを候補にする              │        │       surface_id順の先頭surface   │
│ 5. query自身と同じrelative_pathを     │        │    A：QID内でrelative_path順の    │
│    候補から除外                       │        │       先頭録音。窓は事前平均       │
│ 6. 同一QIDかつ別relative_pathを正解   │        │ 2. 複数modalityをベクトル連結     │
│    とする                             │        │    GL=[G,L]、GA=[G,A]、            │
│                                      │        │    LA=[L,A]、GLA=[G,L,A]           │
│ 7. 全ベクトルを行L2正規化し、query   │        │ 3. そのspecの全modalityに共通する │
│    との内積で順位付け                 │        │    QIDを残す                        │
│                                      │        │ 4. target rankのラベルが存在する   │
│ 正解条件：同じQID、異なる音声ファイル │        │    QIDを残す                        │
│                                      │        │ 5. 連結vectorを行L2正規化           │
│ 分割：なし。学習：なし。              │        │ 6. query自身を除外し、同じ        │
│                                      │        │    target_labelを正解とする        │
│ 指標：queryごとにAP、RR、             │        │ 7. cosine内積の降順でtop_k=10を   │
│ nDCG@10/50/100。runごとに単純平均し、 │        │    取得                             │
│ mAP、MRR、nDCGを出す。                │        │                                  │
│                                      │        │ target rank：family、order          │
│ 出力：                               │        │ 分割：なし。学習：なし。            │
│ experiments/exp3/sub1_audio/         │        │ predictionsのsplit列はall。        │
│ <run>_per_query.tsv                  │        │                                  │
│ metrics.tsv、metadata.json           │        │ 正解条件：同じtarget_label。        │
└──────────────────┬───────────────────┘        │ QID一致だけでは正解にしない。      │
                   │                            │                                  │
                   │                            │ 指標：Precision/Recall/F1@1/5/10、│
                   │                            │ AP@10、RR。query単位の算術平均。   │
                   │                            │ relevantが1件以上あるquery数を     │
                   │                            │ evaluated_query_countとする。      │
                   │                            │                                  │
                   │                            │ 出力：                             │
                   │                            │ experiments/exp3/taxon_probe/     │
                   │                            │ <rank>/<modalities>/<run_labels>/ │
                   │                            │ metrics.tsv、predictions.tsv、     │
                   │                            │ metadata.json                      │
                   │                            └──────────────────┬───────────────┘
                   │                                               │
                   └──────────────────────┬────────────────────────┘
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 14. Taxon probe結果の集約                                                     │
│                                                                              │
│ taxon_probe配下のmetrics.tsvとmetadata.jsonを集約先へコピーする。             │
│ predictions.tsvは指定した場合だけコピーする。                                │
│                                                                              │
│ 出力：                                                                       │
│ experiments/exp3/taxon_probe_collected/                                      │
│ manifest.csv（source、destination、relative_path）                           │
│                                                                              │
│ Experiment 1、Experiment 2、Sub-experiment 1の結果はこの集約には含めない。     │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 15. 再現に必要な固定ファイル                                                   │
│                                                                              │
│ selected_runs.json                                                           │
│ 各embedding runのmetadata.json                                               │
│ common_qids.tsv                                                              │
│ qid_num_recordings.tsv                                                        │
│ taxon labels                                                                 │
│ 各実験のmetadata、metrics、predictions、similarity行列、図                    │
│                                                                              │
│ Wikidata Query Serviceの取得時点、Wikidata dumpの取得時点、Xeno-canto          │
│ API応答時点は生成時に確定する。リポジトリ内のフローチャートから実値は確定しない。│
│                                                                              │
│ 未確認：Wikidata Query Serviceのquery.tsv生成日時・取得版、                    │
│ taxon labelsのgraph input省略時の入力解決順、recording_map.jsonのAPI依存キー、    │
│ Perchの実行環境・依存条件・実パラメータ、selected runの実生成日時・完了状態。     │
└──────────────────────────────────────────────────────────────────────────────┘
```
