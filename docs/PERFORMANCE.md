# Performance Guide – Processing 25 M+ Papers

Large-scale embedding is IO, CPU and GPU heavy. Below are tested practices for multi-year ADS corpora.

## 1. Hardware Recommendations

Resource | Minimum | Optimal
---------|---------|--------
CPU | 8 cores | 32 cores
RAM | 32 GB | 128 GB (enables large Faiss indexes in memory)
GPU | Any 8 GB | A100 40 GB / MI250
Disk | 1 TB SSD | NVMe RAID; > 2 GB/s read
Network (OpenAI) | n/a | 1 Gbps (if API back-end)

## 2. Configuration Tips

Setting | Effect | Recommendation
--------|--------|--------------
`batch_size` | GPU utilisation | 256–512 for MiniLM on 16 GB; 64–128 for mpnet
`use_float16` | halved storage & faster Faiss | keep **on**
`--async --workers N` | parallel IO + preparation | `N = #CPU cores / 2`
`deduplicate` | avoid re-embedding same text | enable for multi-year corpora
`faiss_index_type` | index build time vs search speed | `flat` during ingest; convert to `ivf` or `hnsw` offline
`truncate` | cut texts > 8k chars | 8 192 tokens is safe for SBERT
`device cuda` | forces GPU even if multiple are visible | set per-node

### Example 25 M Run (single GPU)

```bash
sciembed ingest \
  -i /mnt/ads_data -o /mnt/embeddings \
  --years 1995:2023 \
  --model hf://sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 256 --async --workers 32 \
  --no-faiss        # build Faiss offline later
```

Throughput on A100 40 GB ≈ **9 000 docs/s** (≈ 32 h for 25 M)  
Disk writes peak at 250 MB/s (`float16`).

## 3. Distributed Strategy

Year sharding makes horizontal scaling trivial:

* Provide **distinct output_dir per worker** _or_ pre-split by year ranges.
* After all workers finish, merge manifests into a single index with:

```python
from sciembed.components.index import Index
idx = Index("merged.db")
for shard in ["node1/index.db", "node2/index.db"]:
    idx.import_db(shard)
```

## 4. Memory Footprint

* MiniLM float16 = 384 dim × 2 B ≈ 768 B / doc → 25 M ≈ 18 GB
* Keep `use_float16=True` to fit entire corpus into RAM for Faiss Flat search.

## 5. Post-Processing

Convert Flat → IVF:

```python
from faiss import read_index, index_factory, write_index
idx = read_index("index_flat.faiss")
dim = idx.d
quantizer = index_factory(dim, "IVF4096,Flat")
quantizer.train(idx.reconstruct_n(0, idx.ntotal))
index_ivf = faiss.index_cpu_to_gpu(resources, 0, quantizer)
index_ivf.add(idx.reconstruct_n(0, idx.ntotal))
write_index(index_ivf, "index_ivf4096.faiss")
```

Compute PQ codes or HNSW similarly.
