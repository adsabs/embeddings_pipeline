# SciX Experimental Embeddings Pipeline 

End-to-end pipeline that converts **SciX literature (25 M+ papers)** into dense vector embeddings for:

* semantic similarity search  
* hybrid keyword + vector search in Solr / Elastic  
* large-scale downstream ML

It offers both a **command-line interface** (`sciembed`) for production ingestion and a **Streamlit dashboard** for experimentation.

## Key Features
* Multiple embedding back-ends: **OpenAI**, **HuggingFace**, **local GGUF/Llama.cpp**
* Pluggable, typed component architecture (Loader → Preparer → Embedder → Persister → Index)
* Year-level sharding, resumability & deduplication (RocksDB or SQLite)
* Optional **Faiss** vector indexes (Flat / IVF / HNSW) created on the fly
* Metadata-rich manifests for full provenance tracking
* Output formats: NumPy + manifest, JSON/CSV export, Solr-ready JSONL
* Async pipeline for maximum GPU/TPU utilisation
* Streamlit dashboard with embedding playground, corpus processing GUI, model comparison and hybrid-search simulator

---

## Quick Start

```bash
# 1. Install (Python ≥3.9, Linux/OS-X)
git clone https://github.com/adsabs/embeddings_pipeline.git
cd embeddings_pipeline
pip install -e .[all]          # extras install faiss, streamlit, sentence-transformers…

# 2. Ingest two years of ADS JSONL into embeddings/
sciembed ingest \
    -i /data/ads/metadata_by_year        \
    -o ./embeddings \
    --years 2022,2023 \
    --model hf://sentence-transformers/all-MiniLM-L6-v2 \
    --batch-size 128

# 3. Interactive dashboard (another terminal)
streamlit run src/sciembed/dashboard.py
```

Search from CLI:

```bash
sciembed search -o ./embeddings -q "gravitational wave merger" -y 2023 -k 15
```

---

## Installation

### Prerequisites
* Python 3.9+
* For GPU: CUDA-enabled PyTorch (`pip install torch --index-url …`)
* (optional) FAISS – automatically installed via `pip install faiss-cpu` or `faiss-gpu`

### Variants
* Minimal (no dashboard, no faiss):
  ```bash
  pip install sciembed
  ```
* Full (everything):
  ```bash
  pip install sciembed[all]
  ```

### Extra Dependencies (excerpt)
See `requirements.txt`.
```
sentence-transformers>=2.6
faiss-cpu>=1.7          # or faiss-gpu
openai>=1.12
llama-cpp-python>=0.2
tqdm, click, PyYAML, pydantic, streamlit, plotly, scikit-learn, rocksdb
```

---

## Embedding Model URI Scheme

| Prefix          | Example                                            | Notes                                   |
|-----------------|----------------------------------------------------|-----------------------------------------|
| `hf://`         | `hf://sentence-transformers/all-MiniLM-L6-v2`      | Loaded with `SentenceTransformer`       |
| `openai://`     | `openai://text-embedding-3-small`                  | Needs `OPENAI_API_KEY` / `--api-key`    |
| `gguf://`       | `gguf:///models/mxbai-embed-large.gguf`            | CPU-only via `llama-cpp-python`         |

---

## Folder Structure After Ingestion

```
embeddings/
 ├─ 2023_MiniLM_L6_v2_embeddings.npy          # float32 or float16
 ├─ index.db                                  # SQLite bibcode ↔ file index
 ├─ index_MiniLM_L6_v2_2023_flat.faiss        # optional Faiss index
 ├─ manifests/
 │   └─ manifest_2023_MiniLM_L6_v2.json
 └─ deduplication.db                          # RocksDB / SQLite hashes
```

Manifests contain: year, model, preparer hash, prompt hash, dimensions, checksum, etc.

---

## Usage

### CLI (Full reference in docs/API.md)

```bash
sciembed ingest --help
```

Highlights:

* `--input /path`        Directory with `ads_metadata_<year>_full.jsonl` or `2023.json`
* `--years 2010:2023`    Range or comma list
* `--fields title,abstract`
* `--prefix "[TITLE] "`  Adds to every string before embedding
* `--suffix " [EOS]"`
* `--async --workers 8`  Enable asyncio + thread pool
* `--no-faiss`           Skip index creation (faster)
* `--device cuda`        Manually force GPU/CPU

The pipeline is **fully resumable**; rerun with the same output directory and already-processed years are skipped when `--resume` (default).

### Streamlit Dashboard

```bash
streamlit run src/sciembed/dashboard.py
```

* Embedding playground – quick vectors, heatmaps, PCA/t-SNE
* Corpus Processing – GUI wrapper around `ingest`
* Similarity Search – query live Faiss indexes
* Model Comparison – compare embeddings across models
* Hybrid Search – craft Solr script_score + MLT queries

---

## Configuration Files

Create `production_config.yaml`:

```yaml
input_dir: /data/ads/metadata_by_year_full
output_dir: /data/embeddings_ads
years: [2000, 2001, 2002]
model: hf://sentence-transformers/all-MiniLM-L6-v2
batch_size: 256
create_faiss_index: true
faiss_index_type: ivf
deduplicate: true
use_float16: true
```

Load with overrides:

```bash
sciembed ingest -c production_config.yaml --years 2000:2010 --device cuda
```

---

## Library Usage

```python
from sciembed.config import load_config
from sciembed.pipeline import Pipeline

cfg = load_config("production_config.yaml", years=[2022], batch_size=128)
pipe = Pipeline(cfg)
stats = pipe.run()

# Retrieve embedding for a bibcode
vec = pipe.get_embedding("2022ApJ...000..123A")
```

---

## Contributing

* Pre-commit: `black`, `isort`, `ruff`
* Unit tests: `pytest`
* Please open issues / PRs for new embedding back-ends or bug fixes.

---

## License

Apache 2.0 © 2024 ADS
