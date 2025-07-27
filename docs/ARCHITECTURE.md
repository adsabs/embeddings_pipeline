# SciEmbed Architecture Overview

```
            ┌───────────────┐
            │   CLI / UI    │───┐
            └───────────────┘   │
                    Dashboard   │
                                ▼
         ┌─────────────────────────────────────┐
         │           Pipeline (or Runner)      │
         └─────────────────────────────────────┘
          │          │            │      │
          ▼          ▼            ▼      ▼
   ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌─────────┐
   │  Loader  │ │ Preparer  │ │ Embedder │ │Persister│
   └──────────┘ └───────────┘ └──────────┘ └─────────┘
        │               │           │         │
        ▼               │           ▼         ▼
   JSONL records   cleaned text   vectors   .npy + manifest
        │                           │
        └────────► Deduplicator ◄───┘
                            │
                            ▼
                       Index Manager
                 (SQLite + optional Faiss)
```

## Components

Component | Responsibility | Implementation
----------|----------------|---------------
Loader | Discover yearly ADS JSON(L) files and stream records | `components.loader.{JSONLoader,DirectoryLoader}`
Preparer | Concatenate fields, prefix/suffix, lowercase, truncate | `components.preparer.Preparer`
Deduplicator | Cross-year SHA-1 hash DB (SQLite or RocksDB) | `components.deduplicator.Deduplicator`
Embedder | Convert texts → np.float32 vectors | see embedder classes
Persister | Save embeddings (`float32` / `float16`) + manifests | `components.persister.Persister`
Index | Map bibcode → file/row (`index.db`) and build Faiss index | `components.index.{Index,VectorIndex}`

## Data Flow

1. **For each year** (shard): stream records → Preparer produces `(bibcode, text)` pairs  
2. Deduplicator filters duplicates (optional)  
3. Batched texts → Embedder (adaptive batch size)  
4. Resulting vectors + bibcodes stored by Persister  
5. Manifest registered in SQLite; Faiss index built if enabled  

### Concurrency Model

* Loader streams IO in worker threads (async option)  
* Deduplicator & SQLite are thread-safe  
* Embedding batch dispatches to GPU in the main thread to avoid CUDA context contention  

---

## Output Layout

```
output_dir/
 ├─ manifests/manifest_{YEAR}_{MODEL}.json
 ├─ {YEAR}_{MODEL}_embeddings.npy
 ├─ index.db                    # bibcode lookups
 ├─ index_{MODEL}_{YEAR}.faiss  # optional
 ├─ deduplication.db            # if enabled
```

Manifest example:

```json
{
  "year": 2023,
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "dim": 384,
  "fields_hash": "4e89…",
  "prompt_hash": "4e89…",
  "rows": 549321,
  "file": "2023_MiniLM_L6_v2_embeddings.npy",
  "sha256": "…"
}
```

---

## Extending

* **New data source**: implement `components.loader.BaseLoader`
* **New embedder**: subclass `Embedder`, add to `create_embedder`
* **Custom index**: subclass `VectorIndex` (e.g., HNSWlib)
