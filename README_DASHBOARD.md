# 🔬 SciEmbed Dashboard

An interactive Streamlit dashboard for experimenting with scientific embeddings pipeline.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install dashboard requirements
pip install -r dashboard_requirements.txt

# Or install specific models you want to test
pip install sentence-transformers torch  # For HuggingFace models
pip install openai                       # For OpenAI models (optional)
```

### 2. Launch Dashboard

```bash
# Simple launcher
python dashboard_launcher.py

# Or direct Streamlit
streamlit run src/sciembed/dashboard.py
```

### 3. Open in Browser

The dashboard will open automatically at `http://localhost:8502`

## 📋 Dashboard Features

### 🧪 Embedding Playground
- **Model Selection**: Choose from NASA-SMD, sentence-transformers, or OpenAI models
- **Text Input**: Single text, multiple texts, or sample astronomical papers
- **Text Preprocessing**: Add custom prefixes/suffixes
- **Visualization**: 2D projections (PCA/t-SNE), similarity matrices
- **Export**: Download embeddings as JSON

### 🔍 Similarity Search
- **Load Existing Data**: Browse embeddings from previous ingest runs
- **Search Interface**: Query existing paper embeddings
- **Statistics**: View corpus statistics (papers, years, models)
- **Results Display**: Ranked similarity results with scores

### ⚖️ Model Comparison
- **Side-by-Side Testing**: Compare multiple models on same text
- **Performance Metrics**: Embedding dimensions, norms, statistics
- **Similarity Analysis**: Pairwise model similarities
- **Benchmark Results**: Compare model behaviors

### 🎯 Hybrid Search Testing
- **Solr Integration**: Experiment with vector + keyword search combinations
- **Weight Tuning**: Adjust semantic vs keyword search weights
- **Field Selection**: Choose which fields to use for each search type
- **Configuration Export**: Generate YAML configs for production

## 🔧 Configuration

### Model URLs
```python
# HuggingFace models
"hf://nasa-impact/nasa-smd-ibm-st-v2"           # NASA astronomy model
"hf://sentence-transformers/all-MiniLM-L6-v2"  # Small general model
"hf://sentence-transformers/all-mpnet-base-v2" # Better general model

# OpenAI models (requires API key)
"openai://text-embedding-3-small"  # 1536 dimensions
"openai://text-embedding-3-large"  # 3072 dimensions
"openai://text-embedding-ada-002"  # Legacy model
```

### Environment Variables
```bash
# For OpenAI models
export OPENAI_API_KEY="your-api-key-here"

# For CUDA GPU acceleration
export CUDA_VISIBLE_DEVICES=0
```

## 🎯 Use Cases

### Experimenting with Different Models
1. Go to **Embedding Playground**
2. Select different models from the sidebar
3. Input sample astronomical text
4. Compare embedding characteristics

### Testing Hybrid Search for Solr
1. Go to **Hybrid Search Testing**
2. Adjust semantic vs keyword weights
3. Select relevant fields for each search type
4. Export configuration for Solr integration

### Evaluating Model Performance
1. Go to **Model Comparison**
2. Select multiple models to compare
3. Input representative text samples
4. Analyze similarity patterns and statistics

### Searching Existing Embeddings
1. Go to **Similarity Search**
2. Point to existing embeddings directory
3. Search for papers similar to your query
4. Explore corpus statistics

## 📊 Sample Papers Included

The dashboard includes sample astronomical abstracts for testing:
- Exoplanet detection with Kepler
- Dark matter cluster analysis
- Cosmic microwave background studies
- Gravitational wave detection
- Stellar evolution in globular clusters
- High-redshift galaxy formation

## 🔗 Integration with Main Pipeline

The dashboard uses the same core components as the CLI:
- `src/sciembed/components/embedder.py` - Embedding models
- `src/sciembed/config.py` - Configuration system
- `src/sciembed/pipeline.py` - Main pipeline logic

This ensures consistency between experimental results and production runs.

## 🐛 Troubleshooting

### Model Loading Issues
```bash
# Ensure models are installed
pip install sentence-transformers torch

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
- Reduce batch sizes in the sidebar
- Use smaller models (MiniLM vs MPNet)
- Close other applications using GPU/RAM

### OpenAI API Issues
- Verify API key is set correctly
- Check quota and billing status
- Use smaller test texts to avoid token limits

## 📝 Export Formats

### Embeddings Export (JSON)
```json
{
  "texts": ["original text 1", "original text 2"],
  "processed_texts": ["processed text 1", "processed text 2"],
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}
```

### Hybrid Search Config (YAML)
```yaml
hybrid_search_config:
  semantic_weight: 0.7
  keyword_weight: 0.3
  vector_fields: ["title_vector", "abstract_vector"]
  keyword_fields: ["title", "abstract"]
  embedding_model: "hf://nasa-impact/nasa-smd-ibm-st-v2"
```

## 🚀 Next Steps

After experimenting with the dashboard:

1. **Production Config**: Use insights to configure `production_config.yaml`
2. **Model Selection**: Choose the best-performing model for your use case
3. **Hybrid Search**: Implement the tested configuration in Solr
4. **Batch Processing**: Run the full pipeline with `sciembed ingest`

The dashboard serves as your experimental playground before committing to large-scale embedding generation.
