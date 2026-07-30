# root_short reproduction experiments

`temp/root_short.tex` の論文実験を再現するための専用ディレクトリです。
既存の `src/multi_bird_db/` 本体は壊さず、再現用コードとメモをここに分離して置きます。

## 実験構成

元の論文構想は Experiment 1 から 3 まであります。
現行 LaTeX では Experiment 1 がコメントアウトされ、表示上は後続実験が 1 / 2 に詰め直されているため、ここでは元構成に戻して整理します。

## Experiment 1: Similarity and Distance Matrices Between Embeddings

目的は、各モダリティの埋め込みが鳥類同士の関係性をどのように表しているかを比較することです。

対象は以下です。

- `G(tree)`: knowledge graph 上の shortest hop distance
- `G(embedding)`: graph embedding
- `L(en)`: 英語名 embedding
- `L(ja)`: 日本語名 embedding
- `A`: audio embedding

本リポジトリでは graph 5種、language 2種、audio 複数種など、同一モダリティ内にも複数の埋め込みがあります。
そのため、再現コードではモダリティ名に関係なく、選択された全 embedding run の共通 `QID` だけを行・列に採用します。

音声は同一 `QID` に複数ファイルが存在します。
audio embedding run すべてで成功している同一音声ファイルだけを候補にし、`relative_path` 昇順で最初の1件を代表音声として使います。
BirdNET のように1音声が複数windowに分かれる場合は、同一音声ファイル内のwindowベクトルを平均して1本にします。

## Experiment 2: Integrated Embedding Graphs

目的は、Experiment 1 で得た類似度行列から edge を作り、複数モダリティで共通する鳥類間関係を可視化することです。

論文では各モダリティの類似度上位 95 percentile を edge とし、少なくとも1つのモダリティで現れた edge を統合しています。

## Experiment 3: Search Performance With Multi-Modality

目的は、`G`, `L`, `A` およびその組合せで検索性能がどう変わるかを評価することです。

評価指標は以下です。

- `mAP`
- `MRR`
- `nDCG@10`
- `nDCG@50`
- `nDCG@100`

### EXP3-sub1: Audio pretraining / fine-tuning comparison

`root_short.tex` の Table 1 に対応する補助実験です。
目的は、audio embedding の作り方、特に wav2vec 系の事前学習・追加学習の違いが audio-only retrieval にどの程度効くかを見ることです。

この実装では、各 audio embedding run を独立に評価します。
query は1つの音声ファイル、candidate は同じ run 内の他の音声ファイルです。
正解は「同じ `QID` に属する別音声」です。query 音声そのものは candidate から除外します。
BirdNET のように1音声ファイルが複数 window に分かれる場合は、同一 `relative_path` 内の window embedding を平均して1音声1ベクトルにします。

通常実行コマンド:

```bash
make run-exp3-sub1-audio
```

直接 Python を叩く場合:

```bash
python3 experiments/root_short_reproduction/src/exp3_sub1_audio_pretraining.py
```

出力先:

```text
experiments/root_short_reproduction/exp3_sub1_audio/
```

主な出力予定:

- `metrics.tsv`
- `<audio_run>_per_query.tsv`
- `metadata.json`


## Embedding run selection

実験コードは既定で [data/external/embeddings/selected_runs.json](../../data/external/embeddings/selected_runs.json) を読みます。
このファイルに、各埋め込み種類で最終版として採用する run ディレクトリを明示します。

自動検出で全 run を拾いたい場合のみ、次のように `--no-use-selected-runs` を付けます。

```bash
python3 experiments/root_short_reproduction/src/exp1_similarity_matrices.py --no-use-selected-runs
python3 experiments/root_short_reproduction/src/exp3_sub1_audio_pretraining.py --no-use-selected-runs
```

## 現在のコード

Experiment 1 用の通常実行コマンド:

```bash
make run-exp1-simmatrix
```

直接 Python を叩く場合:

```bash
python3 experiments/root_short_reproduction/src/exp1_similarity_matrices.py
```

このコマンドを実行すると、既定では以下に出力します。

```text
experiments/root_short_reproduction/exp1_img/
```

主な出力予定:

- `matrix_qids.json`
- `embedding_run_summary.tsv`
- `representative_audio_files.tsv`
- `*_similarity.npy`
- `*_heatmap.png`
- `*_distribution.png`
- `metadata.json`

このREADME作成時点では、コードのみを追加し、行列・画像はまだ生成していません。
