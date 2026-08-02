# Selected Embedding Runs

Use [selected_runs.json](selected_runs.json) as the canonical list of embedding run directories.
It records the directories that downstream experiments should use instead of blindly scanning every run under `data/external/embeddings`.

Use [check_selected_embedding_runs_remote.sh](check_selected_embedding_runs_remote.sh) to verify that the selected run directories exist and still contain the required output files.
