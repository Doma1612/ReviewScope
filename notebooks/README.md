# Notebooks

Experiments that produced the pipeline decisions. They call the `reviewscope_ml` package; `utils/` holds the shared notebook configuration, caching and result tracking. Results are appended to `data/cache/results.csv`.

| Notebook | Question |
|---|---|
| `00_config` | Shared configuration and GPU limits |
| `01_data_exploration` | Structure and size of the Yelp reviews |
| `02_business_data_exploration` | Joining businesses and filtering hotels |
| `02_preprocessing` | Effect of text cleaning on the embeddings |
| `03_bertopic_baseline` | Off the shelf BERTopic as baseline |
| `04_embeddings` | Embedding model comparison |
| `05_dimensionality_reduction` | UMAP and PCA settings |
| `06_clustering` | Clustering algorithms and parameters |
| `07_comparison` | Comparison of the configurations above |
| `08_llm_labeling` | Prompts and context for LLM labels |
| `10_pipeline_comparison` | End to end comparison of the pipeline variants |
| `11_hitl_roundtrip` | Applying reviewer feedback to a rerun |

Setup: see "Development from source" in the root README, then `pip install jupyter` and start Jupyter from the repository root. The notebooks expect the raw Yelp download in `data/raw/`.
