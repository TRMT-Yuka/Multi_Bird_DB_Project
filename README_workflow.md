```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 0. 対象範囲・外部入力                                                         │
│                                                                              │
│ 鳥類全体を表すQID Q5113の下部に所属する全ノードを対象とする。                  │
│ 外部でWikidata Query Serviceの検索結果を作成する。                            │
│ 入力：鳥類の子ノードを取得する検索定義                                            │
│ 処理：検索結果の各行からentity URLを取得し、item列をTSVとして保存する。         │
│ 出力：data/raw/wikidata/query.tsv                                             │
│       （item列にQIDのURLを持つ、外部生成済み検索結果）                         │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1A. Wikidata dumpの準備                                                       │
│                                                                              │
│ 入力：Wikidata entities dump                                                  │
│ 処理：対象QIDのentity JSONを後続工程で参照できる状態にする。                   │
│ 出力：data/raw/wikidata/dumps/latest-all.json.bz2                              │
│ 既存dumpの版・取得日時・ハッシュは、この工程の成果物からは固定値として          │
│ 確認できない。                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                     （1Aと1Bは0から独立して実行できる）
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1B. 対象QIDリストの作成                                                       │
│                                                                              │
│ 入力：data/raw/wikidata/query.tsv の item 列                                 │
│ 処理：URL末尾のQIDを抽出し、出現順を保ったまま重複を除去する。                  │
│       Q5113が含まれない場合だけ、Q5113を先頭に追加する。                      │
│       この工程でWikidataへ再検索したり、子孫関係を再計算したりしない。          │
│       1Aのdumpには依存せず、0のquery.tsvだけを読む。                           │
│ 出力：data/interim/wikidata/bird_qids.tsv                                    │
│       qid列、1行1QID。1Bの対象集合は、外部で作成されたquery.tsvに依存する。    │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. 対象QIDのentity JSON抽出                                                   │
│                                                                              │
│ 入力：1Aのdump、1Bのbird_qids.tsv                                             │
│ 処理：dump全体からbird_qids.tsvに含まれるQIDだけを抽出する。                   │
│       QIDごとに1つのentity JSONを保存し、既存JSONは再利用して上書きしない。    │
│ 出力：data/interim/wikidata/json/<QID数値部の1桁目>/<2桁目>/Qxxxx.json          │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. ontologyレコード生成                                                       │
│                                                                              │
│ 入力：2のQID別entity JSON                                                     │
│ 処理：entity JSON 1件をontologyレコード1件へ変換し、全レコードをlistとして保存│
│       する。ラベル、別名、画像名、Xeno-canto ID、taxon rank、parent_taxon、    │
│       英日Wikipedia情報、JSON pathを抽出する。                               │
│ 出力：data/processed/bird_ontology.pkl                                       │
│       qid, entity_url, en_name, ja_name, en_aliases, ja_aliases, img_names,    │
│       xeno_canto_species_id, taxon_name, taxon_rank, taxon_rank_name,          │
│       parent_taxon, parent_taxon_name, parent_taxon_ja_name,                   │
│       enwiki_title, enwiki_url, jawiki_title, jawiki_url, path                 │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     ▼en_aliases, ja_aliasesも
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
┌────────────────────────────┐ ┌────────────────────────────┐ ┌────────────────────────────┐
│ 4A. taxonomy graph         │ │ 4B. 言語表層形生成          │ │ 4C. Xeno-canto対象抽出      │
│                            │ │                            │ │                            │
│入力：3のontology          │ │入力：3のontology          │ │入力：3のontology          │
│処理：ontologyの各行を1    │ │処理：QIDごとに候補を作る。 │ │処理：xeno_canto_species_id │
│ノードとする。各行の       │ │英語の候補順はen_name、    │ │を持つQIDを抽出し、QIDと   │
│parent_taxonについて、     │ │enwiki_title、日本語の候補 │ │species IDを対応付ける。   │
│parent→childの有向辺を1本 │ │順はja_name、jawiki_title。│ │末端QIDの判定はtaxonomy    │
│追加する。逆辺は追加しない。│ │空文字と同一文字列を除去し、│ │graphのout_degree=0で行う。│
│親QIDが対象ノードにない場合│ │残った順序でordinalを付す。│ │出力：data/interim/         │
│は空属性のplaceholder親を  │ │表層形1件を1行とする。     │ │wikidata/bird_xeno_canto_ids│
│追加する。                 │ │出力：data/processed/      │ │.tsv                        │
│                            │ │language/<enまたはja>/     │ └──────────────┬─────────────┘
│出力：data/processed/graph/ │ │surface_manifest.tsv       │                │
│bird_taxonomy_graph.pkl     │ │surface_id,qid,language,   │                ▼
│ノードにはQID、名称、rank、 │ │ordinal,surface_text,      │ ┌────────────────────────────┐
│親、entity/Wikipedia URL等  │ │source,source_index。      │ │ 5. 録音情報取得              │
│を保持する。                │ └──────────────┬─────────────┘ │                            │
└──────────────┬─────────────┘                │               │入力：4Cの末端QID/species ID│
               │                              │               │処理：APIでspにspecies ID    │
               ▼                              ▼               │（ハイフンを空白に置換）と   │
┌────────────────────────────┐  ┌───────────────────────────┐ │q:Aを指定して検索する。      │
│ 4D. Taxon labels生成        │  │ 4B-2. 表層形manifest確認   │ │page=1、per_page=20、最大1   │
│                            │  │                           │ │ページ、since=2025-05-01。  │
│入力：4Aのtaxonomy graph     │  │入力：4Bのsurface_manifest │ │でAPI JSONを保存する。       │
│処理：各QIDについて対象自身  │  │処理：表層形をQID単位の候補 │ │numRecordingsはAPIページの   │
│はラベル候補にせず、その親  │  │として後続入力にする。      │ │値を記録し、ローカル取得数   │
│から上方向へ辿る。指定rank   │  │同じQIDの候補はmanifestの   │ │とは別に扱う。               │
│（order/family）に最初に一致 │  │ordinal順で保持する。       │ │出力：data/interim/xeno-canto/│
│した最近傍祖先を1つ採用する。│  └───────────────────────────┘ │api_recordings/<QID>/page*.json│
│parent欠損、graph外、循環、  │                                  │および取得manifest/summary    │
│一致なしは空欄。複数親分岐は │                                  └──────────────┬─────────────┘
│実装上存在せず、単一の       │                                                 ▼
│parent_taxonを辿る。          │                                  ┌────────────────────────────┐
│出力：data/processed/         │                                  │ 6. 録音ID対応表              │
│taxonomy/qid_taxon_labels.tsv │                                  │                            │
│、.json、meta.json。各行は   │                                  │入力：5のAPI page JSON       │
│qid、rank、label_qid、        │                                  │処理：ページ内の録音IDを      │
│label_name、distance。        │                                  │QIDごとにページ順で抽出し、  │
└──────────────┬─────────────┘                                  │重複を除いて対応付ける。     │
               │ 13Bのlabel入力                                  │出力：data/interim/xeno-canto/│
               ▼                                                │recording_map.json           │
┌────────────────────────────┐                                  │QID→録音ID列。               │
│ 4D-1. Taxon label対応保持   │                                  └──────────────┬─────────────┘
│13Bで使用するrank labelを、 │                                                 ▼
│対象QIDと対応付けたまま保存 │                                  ┌────────────────────────────┐
│し、埋込みを結合しても       │                                  │ 7. 録音ファイル取得          │
│QID対応を失わない。          │                                  │                            │
└────────────────────────────┘                                  │入力：4Cの対象QID、6の録音ID│
                                                                  │処理：末端QIDごとに対応表の  │
                                                                  │先頭から最大20件を選び、      │
                                                                  │品質AのAPI結果を取得する。   │
                                                                  │ダウンロード失敗時に後続IDを │
                                                                  │補充しない。既存ファイルは   │
                                                                  │再利用し、取得済みファイルは  │
                                                                  │manifest上の既存件数として   │
                                                                  │扱う。15秒超は15秒へ切り出し、│
                                                                  │15秒以下はそのまま保存する。 │
                                                                  │出力：data/raw/xeno-canto/    │
                                                                  │<QID>/<recording_id>.<ext>、 │
                                                                  │existing_audio_manifest、     │
                                                                  │取得summary。                 │
                                                                  └──────────────┬─────────────┘
                                                                                 ▼
                                                                  ┌────────────────────────────┐
                                                                  │ 8. 音声窓作成・音声埋込み   │
                                                                  │入力：7の音声ファイル         │音声作成時の制約について追記
                                                                  │処理：backendごとに入力波形を │
                                                                  │作り、窓ごとに1行の埋込みを   │
                                                                  │保存する。wav2vec2は16kHz、  │
                                                                  │ファイル全体、モデル          │
                                                                  │facebook/wav2vec2-base-960hの│
                                                                  │last_hidden_stateを時間平均。 │
                                                                  │BirdNETは48kHz、3秒窓、       │
                                                                  │overlap=0。末尾は実装のpad処理。│
                                                                  │Perchは32kHz、5秒窓、overlap=0。│
                                                                  │外部Bioacoustics Model Zooの  │
                                                                  │Perch埋込みを取り込む。内部の │
                                                                  │出力層・時間集約は未確認。     │
                                                                  │ffmpegで読めない音声は失敗項目 │
                                                                  │として記録する。               │
                                                                  │出力：data/external/embeddings/│
                                                                  │audio/<backend>/<model>/<run>/│
                                                                  │embeddings.npy、audio_ids.json、│
                                                                  │qids.json、audio_manifest.tsv、│
                                                                  │metadata、failed_items、summary│
                                                                  │manifestはaudio_id,qid,        │
                                                                  │relative_path,window_index、   │
                                                                  │開始/終了秒、sampling rate、   │
                                                                  │duration等を持つ。             │
                                                                  └────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 9A. グラフ埋込み生成                                                         │
│                                                                              │
│ 入力：4Aのtaxonomy graph                                                     │
│ 共通：ノードをQIDの文字列昇順で並べ、embeddings.npy[i]とqids.json[i]を対応   │
│       させる。出力ベクトルはL2正規化して保存する。seedは42。                 │
│ initial_features=embedding：hidden_dim=32の学習可能なnn.Embeddingを         │
│       正規分布（標準偏差0.1、seed付き）から初期化し、各行をL2正規化する。      │
│ 辺：parent→child方向を使い、全ノードにself-loopを加える。undirected=false。   │
│                                                                              │
│ GCN：hidden_dim=32、output_dim=128、layers=2、epochs=200、learning_rate=.01、 │
│      negative_samples=20、weight_decay=1e-5、GAE negative sampling。          │
│      正例はgraphの辺、負例は既存の有向辺にならないhead/tail置換。             │
│      dot-product scoreのsoftplus正負損失をAdamで最適化する。                  │
│ GraphSAGE：hidden_dim=32、output_dim=128、layers=2、epochs=200、               │
│      learning_rate=.001、aggregator=mean、num_neighbors=[25,10]、             │
│      negative_samples=1、weight_decay=1e-5、device=cuda。                    │
│      各層で近傍をサンプリングし、負例1件を同じhead/tail置換で生成する。        │
│ GRACE：encoder=GCN、hidden_dim=32、output_dim=128、projection_dim=128、       │
│      layers=2、epochs=200、learning_rate=.001、tau=.5、                      │
│      edge_drop=[.2,.4]、feature_mask=[0,0]、weight_decay=1e-5、device=auto。  │
│      edge dropした2視点を作り、同じinitial embeddingを入力する。             │
│      同一ノードを正例、同一batchの他ノードを負例とするin-batch loss。         │
│      保存するのはprojector出力ではなくencoder出力。                         │
│ Node2Vec：output_dim=128、walk_length=40、num_walks=10、window_size=10、     │
│      p=1、q=1、negative_samples=5、epochs=200、learning_rate=.001、           │
│      weight_decay=1e-5、batch_size=8192、undirected=false。                   │
│      有向隣接でwalkし、行き止まりではwalkを終了する。負例は次数^0.75分布。   │
│ TransE：output_dim=128、epochs=200、learning_rate=.001、margin=1、            │
│      p_norm=1、negative_samples=10、weight_decay=1e-5、relation=parent_taxon。│
│                                                                              │
│ 出力：data/external/embeddings/graph/<method>/<run>/                         │
│       embeddings.npy、qids.json、metadata.json、必要時training_trace等。     │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ graph run
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 9B. 言語埋込み生成                                                           │
│                                                                              │
│ 入力：4Bのsurface_manifest.tsv                                               │
│ 処理：各行（QID・language・ordinal・surface_text）を1入力とする。              │
│       英語はgoogle-bert/bert-base-uncased、日本語は                         │
│       tohoku-nlp/bert-base-japanese-v3をAutoTokenizer/AutoModelで処理する。  │
│       max_length=32、padding=True、truncation=True。last_hidden_stateを      │
│       attention mask付きmean poolingする。paddingはmaskで除外するが、特殊token│
│       を別途除外する指定はない。出力層・projection・L2正規化はない。           │
│ 出力：data/external/embeddings/language/<lang>/<run>/                        │
│       embeddings.npy、surface_ids.json、qids.json、qid_to_surfaces.json、    │
│       surface_manifest.tsv、metadata.json、summary.json。                    │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ language run
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 10. 埋込み行の対応付け・使用runの確定                                        │
│                                                                              │
│ graph：embeddings.npy[i] ↔ qids.json[i] ↔ QID                                │
│ language：embeddings.npy[i] ↔ surface_ids.json[i] ↔ qids.json[i]。            │
│           同じQIDの複数表層形は(qid,ordinal,surface_id,text)順の先頭を採用。    │
│ audio：embeddings.npy[i] ↔ audio_manifest[i] ↔ audio_ids/qids。               │
│        同じ(QID,relative_path)の複数窓を平均して1録音ベクトルにする。         │
│                                                                              │
│ selected_runs.jsonが存在しuse-selectedが有効なら、そのlabel/pathを使う。      │
│ それ以外でauto-discoverが有効な12A/12B/12C/13Aは、必要ファイルを持つ完了runを│
│ 探索する。最新1件だけを選ぶ規則ではない。13Bはselected_runsだけを読む。       │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ 共通入力から独立分岐
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 11. 共通QID集合・録音数                                                       │
│                                                                              │
│ 入力：10で選択/探索したrun、QID・音声対応、API page001                       │
│ 処理：各runのQID集合を作り、全モダリティ間の共通集合を交差で求める。           │
│       音声は共通する(QID,relative_path)を作った後のQID集合とする。             │
│       API page001のnumRecordingsをQIDごとに読み、ローカル取得数とは分ける。    │
│ 出力：common_qids.tsv、qid_num_recordings.tsv、run_qid_counts.tsv             │
│       complete_qid_countはrun集合と他モダリティ共通集合の交差件数。            │
│       12Bは録音数表に行がないQIDを除外、12Cは値0として扱う。                  │
└───────────────────────┬──────────────────────┬───────────────────────┬──────┘
                        │                      │                       │
                        ▼                      ▼                       ▼矢印はあっているのか　実験の粒度が粗い
┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────┐
│ 12A. Experiment 1 通常版     │ │ 12B. Experiment 1 録音数順版  │ │ 12C. Experiment 2 統合graph  │
│入力：10の各run                │ │入力：10の各run、11のcommon_ │ │入力：10のG/L/A run、11の      │
│処理：全runのQID集合の共通だけ │ │qids、qid_num_recordings       │ │common_qids、録音数表          │
│を残し、QID昇順で整列する。各  │ │処理：録音数表に行があるQIDを │ │処理：3runにもあるQIDを対象に │
│行をL2正規化しS=XX^Tを計算、   │ │(-numRecordings,QID)順にし、  │ │し、録音数表にないQIDは0、    │
│対角を1.0にする。              │ │上位100件。同数時QID昇順。     │ │(-numRecordings,QID)順で上位 │
│出力：runごとのsimilarity.npy、│ │runに欠落QIDがあればそのrunは │ │100件を選ぶ。各modalityの上三 │
│heatmap.png、distribution.png、│ │作成しない。                   │ │角（対角除外）の95パーセンタイル│※なぜ95％か
│matrix_qids.json、summary、    │ │出力：heatmap、distribution。  │ │をNumPy既定の線形補間で計算し、│
│代表audio、metadata。上三角は │ └──────────────────────────────┘ │score以上のi<jを無向辺にする。│
│k=1の分布とする。              │                                  │同一pairをunionしweight=max。 │
└──────────────────────────────┘                                  │union_thresholdは再選別せず、 │
               │                                                      │描画タイトルとmetadataへ記録。│
               │                                                      │全QIDを追加して孤立ノードも描画。│
               │                                                      │spring_layout(seed=42, weight、│
               │                                                      │k=.85, iterations=200)を4図で│
               │                                                      │共有する。出力：4つのsvgと   │
               │                                                      │metadata。                     │
               │                                                      └──────────────┬─────────────┘
               │                                                                     │ 12Cの結果
               │                                                                     │（14へは接続しない）
               （11から独立した13Aへ）                                               │
┌────────────────────────────┐                                                        │
│ 13A. Experiment 3 音声検索 │                                                        │
│入力：11の共通入力から独立に │                                                        │
│選択/探索したaudio run。12A/ │                                                        │
│12B/12Cの出力は読まない。    │                                                        │
│処理：窓を録音単位へ平均し、  │                                                        │
│同一run内でquery自身を除外。 │                                                        │
│同一QIDかつ別録音を正解とし、 │                                                        │
│正解0件queryは除外する。全候 │                                                        │
│補をcosine降順、同点はstable  │                                                        │
│mergesort。APは全候補順位と全 │                                                        │
│正解数を分母、MRRは最初の正解 │                                                        │
│順位の逆数。AP@10は計算しない。│                                                        │
│平均対象は正解があるquery、    │                                                        │
│evaluated_query_countもその件数。│                                                       │
│出力：per-query、run summary。 │                                                        │
└────────────────────────────┘                                                        │
               11から独立分岐（13A）                                                    │
               ▼                                                                      │
                                                                                      │
                         4Dのtaxon label ────────────┐                               │
                                                      ▼                               │
                                       ┌────────────────────────────┐               │
                                       │ 13B. Experiment 3 Taxon probe│               │
                                       │入力：4Dのtaxon label、10の │               │
                                       │selected G/L/A run。13A結果 │               │
                                       │および12A/B/C結果は読まない。│               │
                                       │処理：rankごとにlabelのある │               │
                                       │QID共通集合を作る。各QIDの  │               │
                                       │G/L/Aを各1代表値にし、pattern│               │
                                       │順に連結する。学習・train/  │               │
                                       │test分割なし。query自身を   │               │
                                       │除外し、同じtarget labelを   │               │
                                       │正解とする。top_k=10。      │               │
                                       │P@k=上位k正解数/k、R@k=上位k│               │
                                       │正解数/全正解数、F1は調和平均。│              │
                                       │AP@10=上位10位の各正解位置   │               │
                                       │precision合計/全正解数。RRは │               │
                                       │上位10位以内の最初の正解の逆 │               │
                                       │数、なければ0。全正解0件も   │               │
                                       │per-query平均へ含めるが、    │               │
                                       │evaluated_query_countは全正解│               │
                                       │が1件以上の件数。            │               │
                                       │top-k境界はargpartition後に  │               │
                                       │mergesortするが、境界同点の  │               │
                                       │選択規則は明示されない。     │               │
                                       │出力：pattern/rank/run組合せ │               │
                                       │別のper_query、summary、metadata。│            │
                                       └──────────────┬─────────────┘               │
                                                      │ 13Bのみ                      │
                                                      ▼                              │
                                       ┌────────────────────────────┐               │
                                       │ 14. Taxon probe結果集約     │
                                       │入力：13Bだけのrank/pattern  │
                                       │結果。13A、12A/B/Cからは接続 │
                                       │しない。                     │
                                       │処理：G/L/A/GL/GA/LA/GLAの各 │
                                       │run組合せを別結果としてrank別│
                                       │に集約し、query数と平均指標を │
                                       │記録する。                   │
                                       │出力：exp3/taxon_probe以下の │
                                       │per_query、summary、metadata。│
                                       └────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 15. 再現用成果物の固定                                                       │
│                                                                              │
│ 入力：1〜14の入力、使用run、対象QID、taxon label、代表表層形/録音、結果。      │
│ 処理：embeddings.npyと、行をQID・surface_id・audio_manifestへ対応付ける        │
│       ファイルを同じrun情報とともに保持する。selected_runs、common_qids、    │
│       qid_num_recordings、qid_taxon_labels、表層形manifest、音声manifest、    │
│       各処理metadataを一組として参照する。                                   │
│       seed=42、次元、モデル名、入力条件、既存ファイル再利用条件を、実際に      │
│       保存されたmetadataの値に基づいて確認する。                             │
│                                                                              │
│ 出力：各embedding runの本体・行対応・metadata、実験ごとの入力集合と結果。      │
│ データ取得日時、ハッシュ、モデル版、依存環境が記録されていない場合は、         │
│ 固定済みと扱わず「未確認／未記録」とする。                                    │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 図内注記：未確認事項                                                         │
│ ・query.tsv作成時の実行日時、Wikidata版、完全な外部検索結果は未確認。           │
│ ・dumpの取得時点/ハッシュ、BERTの配布版、依存ライブラリ版は未確認または未記録。│
│ ・Perchの内部checkpoint、出力層、時間方向集約は未確認。                       │
│ ・selected_runsの実際のlabel/pathはファイル内容を参照して確定する。           │
│ ・自動探索のcompleteは必要ファイル存在条件であり、最新1件選択ではない。        │
│ ・実装に明示されない同点処理、欠損補充、複数親の選択規則は追加していない。     │
└──────────────────────────────────────────────────────────────────────────────┘
```
