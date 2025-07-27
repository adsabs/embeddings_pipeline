# Embeddings Pipeline – Design & Implementation Plan

## 1. Goals & Non-Functional Requirements

- **Throughput**: ≥ 100k papers/hour on a modern 32-core server  
- **Scalability**: Process >10M records without exhausting RAM  
- **Configurability**: CLI / API flags for:  
  - Fields to embed (title, abstract, author list, …)  
  - Optional pre-/post-prompt strings per field or globally  
  - Any supported embedding model (local or API)  
- **Re-entrancy**: Safe to resume / append new years without duplicating work  
- **Traceability**: Every embedding row retains bibcode + metadata hash  
- **Storage efficiency**: ≤ 1 KB / vector on disk, O(1) look-up by bibcode  

## 2. High-Level Architecture

```
           ┌──────────┐  JSON streams  ┌──────────┐  batched tensors ┌──────────┐
           │  Loader  ├───────────────►│ Preparer │──────────────────►│ Embedder │
           └──────────┘                └──────────┘                   └────┬─────┘
                ▲                                 │                        │
binary offsets  │                                 ▼                        │
           ┌────┴────┐                    ┌────────────┐             ┌─────▼─────┐
           │  Index  │◄───vector+meta─────┤ Persister  ├────────────►│  Store    │
           └─────────┘                    └────────────┘             └───────────┘
```

### Components

1. **Loader** – memory-mapped gzip/JSONL reader yielding dicts  
2. **Preparer** – builds text snippets according to user config  
3. **Embedder** – pluggable interface (OpenAI, HuggingFace, llamacpp, etc.)  
4. **Persister** – writes vectors and metadata in columnar format  
5. **Index** – lightweight key→offset map for random retrieval  
6. **CLI / Python API wrapper**  

## 3. Data Flow & File Formats

### Input
- Directory hierarchy: `/ads_metadata_by_year_full/{YYYY}.json(.gz)`  
- Assume each line is a JSON dict with at minimum `{bibcode,title,abstract,…}`

### Intermediate
- Parquet "prep" files (optional) if users want to inspect text before embedding

### Output
1. **Vectors**: float16 array, stored as Arrow IPC or Faiss mmap index  
   File: `embeddings_{model}_{year}.f16`  
2. **Bibcode index**: Plaintext newline-delimited list aligned with vector row-id  
   File: `bibcodes_{year}.txt`  
3. **Metadata sidecar** (fields hash, prompt hash, model id, dim, date)  
   File: `manifest_{year}.json`  
4. **Global SQLite or DuckDB catalog** summarizing manifests for quick look-up

## 4. Detailed Component Design

### 4.1 Loader
- Uses Python's orjson & ijson for iterative, zero-copy parsing  
- Supports .json, .jsonl, .json.gz (detected by magic bytes)  
- Yields only requested fields (reduces GC pressure)  
- Provides tqdm-style progress callbacks

### 4.2 Preparer

**Config schema (YAML/JSON):**
```yaml
fields:       ["title", "abstract"]
prefix:       "You are an astrophysicist. "
suffix:       ""
delimiter:    "\n\n"
lowercase:    true
truncate:     3_000   # characters
```

**Algorithm:**
```python
for rec in loader:
    parts = [cfg.prefix] + [rec[f] for f in cfg.fields if f in rec] + [cfg.suffix]
    text  = cfg.delimiter.join(parts)[:cfg.truncate]
    yield (rec["bibcode"], text)
```

### 4.3 Embedder

**Interface:**
```python
class Embedder:
    name: str           # e.g. "text-embedding-ada-002"
    dim:  int
    batch_size(doc_len): int
    def embed_batch(self, list[str]) -> np.ndarray  # shape (n,dim)
```

**Implementations:**
- OpenAI API wrapper (async, parallel connections)  
- HuggingFace local (sentence-transformers), using PyTorch or ONNX  
- Llama.cpp or other GGUF via ctransformers for CPU-only nodes  
- Factory chooses implementation based on config (model=…)

**Performance:** 
- Adaptive batch_size = min( (32k tokens / avg_len), max_bs)  
- Batching queue fed by producer thread; consumer GPU process

### 4.4 Persister & Index
- Use pyarrow IPC stream writer with float16; 50% smaller than float32  
- Simultaneously append to Faiss on-disk index (IndexFlatIP or HNSW)  
- Ensure vector row-ids align with insertion order → bibcode text file acts as primary key  
- Provide create_index --merge to combine yearly blocks into decade/global indexes later

## 5. Parallelism & Speed Optimizations

- **Use multiprocessing or asyncio workers:**
  - I/O thread (Loader)
  - 2-4 CPU workers (Preparer/Token counting)
  - 1-N GPU/remote embedding workers (Embedder)
  - 1 I/O writer (Persister + Faiss)
- **Pipeline uses bounded async queues** to avoid producer/consumer imbalance
- **Memory budget**: (batch_size × dim × 2 bytes) + queues ≤ 4 GB
- **If remote API rate-limited**, fallback to multi-key round-robin

## 6. User Interfaces

### CLI Examples
```bash
$ sciembed ingest \
    --input /home/scixmuse/scix_data/ads_metadata_by_year_full/ \
    --years 2015:2020 \
    --fields title abstract \
    --prefix "You are an astrophysicist." \
    --model hf://sentence-transformers/all-MiniLM-L6-v2 \
    --batch 512 \
    --out /embeddings
```

### Python API
```python
from sciembed import Pipeline, Config
pipe = Pipeline(Config(
    input_dir   = "...",
    years       = [2021],
    fields      = ["title","abstract"],
    model       = "openai://text-embedding-3-small",
    prefix      = "Summarize:",
))
pipe.run()
```

## 7. Resumability & Deduplication

During ingest create a `.lock` and `manifest_*.json` containing:
```json
{ "cursor": "last_bibcode", "row": "n", ... }
```
On restart pipeline seeks to first unprocessed row.

**For dedup across years/updates:**
- Maintain a global RocksDB with bibcode → SHA256(prepared_text)  
- Skip if hash already present (O(logN) lookup, on-disk)

## 8. Testing & Benchmarking

- Unit tests for Loader with synthetic gzipped JSON  
- Golden-vector test: fixed paper should yield same vector across runs  
- Stress test: 1M random abstracts, measure throughput & RAM  
- Integration test: full year 2020, verify Faiss recall vs cosine python calculation

## 9. Deployment & Ops

- Publish as pip package `sciembed` (entrypoint `sciembed`)  
- Dockerfile with optional CUDA support  
- Configurable logging (rich + json)  
- Prometheus metrics: processed_records_total, tokens_total, seconds_per_batch

## 10. Future Extensions

- Chunking of full-text PDFs when they become available  
- Rerank or co-embed with citations graph via multi-modal encoder  
- Hybrid search (BM25 + Vector) endpoints leveraging Qdrant / Weaviate  
- Online incremental embedding for newly published papers via webhook listener

## Summary

The proposed pipeline separates I/O, preparation, embedding, and persistence into modular, highly parallel components. Users can select fields, add custom prompts, pick any supported model, and process decades of ADS metadata quickly while storing vectors alongside bibcodes in compact, searchable formats.

## Implementation Roadmap

### Phase 1: Core Components (High Priority)
1. Set up project structure and core pipeline architecture
2. Implement JSON data loader with memory mapping and progress tracking
3. Create configurable text preparer for field selection and prompting
4. Build pluggable embedder interface supporting multiple models
5. Implement efficient vector storage with bibcode indexing

### Phase 2: Performance & UX (Medium Priority)
6. Add parallelization with async queues for speed optimization
7. Create CLI interface for user configuration
8. Add resumability and deduplication features

### Phase 3: Production Ready (Lower Priority)
9. Write comprehensive tests and benchmarks
10. Add monitoring, logging, and deployment infrastructure
