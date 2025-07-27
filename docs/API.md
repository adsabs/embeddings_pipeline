# SciEmbed API / CLI Reference

## 1. Command-Line Interface (`sciembed`)

### 1.1 `ingest`

```
sciembed ingest -i INPUT_DIR -o OUTPUT_DIR --years 2020:2023 [options]
```

Option | Description | Default
-------|-------------|---------
`--fields` | Comma-separated fields to embed | `title,abstract`
`--model/-m` | Embedding model URI | `hf://sentence-transformers/all-MiniLM-L6-v2`
`--batch-size/-b` | Batch size hint | 32
`--prefix / --suffix` | Text added before/after each record | ″
`--device` | `cuda`, `cpu`, `auto` | `auto`
`--async` | Use asynchronous pipeline | off
`--workers` | Threads for async loader | 4
`--no-faiss` | Do **not** create Faiss index | off
`--config/-c` | YAML/JSON config to load | –
`--api-key` | API key for OpenAI | env var or flag
`--no-progress` | Disable TQDM bars | off

### 1.2 `search`

```
sciembed search -o EMB_DIR -q "query text" -y 2023 [-k 10] [--model MODEL]
```

Returns list of bibcodes + cosine similarities using existing Faiss index.

### 1.3 `info`

```
sciembed info -o EMB_DIR
```

Shows total embeddings, models, years, manifests.

---

## 2. Python Classes

### 2.1 `Config`

Path-aware dataclass with automatic validation. Key nested configs:

* `preparer_config` – see components.preparer.PreparerConfig
* `embedder_config` – see components.embedder.EmbedderConfig
* `deduplication_config` – see components.deduplicator.DeduplicationConfig

Use `load_config(path, **overrides)` helper to merge YAML/JSON with CLI overrides.

### 2.2 `Pipeline`

Method | Purpose
-------|---------
`run()` | Main ingestion loop (per-year batching)
`search_similar(text, year, k)` | Vector search
`get_embedding(bibcode)` | Retrieve stored vector

### 2.3 Embedders

Factory: `create_embedder(EmbedderConfig)` → one of

* `OpenAIEmbedder`
* `HuggingFaceEmbedder`
* `LlamaCppEmbedder`

All expose `.embed_batch(list[str])`, `.dim`, `.name`.

---

## 3. Streamlit Dashboard API

Run: `streamlit run src/sciembed/dashboard.py`

Main functions (internals):

* `embedding_playground`
* `corpus_processing`
* `similarity_search`
* `model_comparison`
* `hybrid_search_testing`

The dashboard relies exclusively on the public pipeline and component APIs – no hidden behaviour, making it a useful example for custom UIs.
