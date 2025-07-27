"""Pluggable embedding interface supporting multiple models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbedderConfig:
    """Configuration for embedder models."""
    model: str
    batch_size: int = 32
    max_tokens: int = 32000
    api_key: Optional[str] = None
    device: str = "auto"
    
    @property
    def model_type(self) -> str:
        """Determine model type from model string."""
        if self.model.startswith("openai://"):
            return "openai"
        elif self.model.startswith("hf://"):
            return "huggingface"
        elif self.model.startswith("gguf://"):
            return "llamacpp"
        else:
            return "unknown"
    
    @property
    def model_name(self) -> str:
        """Extract model name from model string."""
        if "://" in self.model:
            return self.model.split("://", 1)[1]
        return self.model


class Embedder(ABC):
    """Abstract base class for embedding models."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name identifier."""
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        pass
    
    @abstractmethod
    def batch_size(self, doc_len: int) -> int:
        """
        Calculate optimal batch size based on document length.
        
        Args:
            doc_len: Average document length in characters
            
        Returns:
            Optimal batch size for this document length
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of shape (len(texts), dim) containing embeddings
        """
        pass
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Array of shape (dim,) containing embedding
        """
        batch_result = self.embed_batch([text])
        return batch_result[0]


class OpenAIEmbedder(Embedder):
    """OpenAI API embedder implementation."""
    
    def __init__(self, config: EmbedderConfig):
        """
        Initialize OpenAI embedder.
        
        Args:
            config: Embedder configuration
        """
        self.config = config
        self._name = config.model_name
        self._client = None
        self._dim = None
        
        # Initialize client lazily
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.config.api_key)
            
            # Get dimension by testing with a small text
            test_response = self._client.embeddings.create(
                model=self._name,
                input=["test"]
            )
            self._dim = len(test_response.data[0].embedding)
            
        except ImportError:
            raise ImportError("OpenAI package required: pip install openai")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def dim(self) -> int:
        if self._dim is None:
            self._init_client()
        return self._dim
    
    def batch_size(self, doc_len: int) -> int:
        """Calculate batch size based on token limits."""
        # Rough estimation: 4 chars per token
        tokens_per_doc = doc_len // 4
        max_docs = self.config.max_tokens // max(tokens_per_doc, 1)
        return min(max_docs, self.config.batch_size)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch using OpenAI API."""
        if not texts:
            return np.array([]).reshape(0, self.dim)
        
        try:
            response = self._client.embeddings.create(
                model=self._name,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed: {e}")


class HuggingFaceEmbedder(Embedder):
    """HuggingFace sentence-transformers embedder implementation."""
    
    def __init__(self, config: EmbedderConfig):
        """
        Initialize HuggingFace embedder.
        
        Args:
            config: Embedder configuration
        """
        self.config = config
        self._name = config.model_name
        self._model = None
        self._dim = None
        
        # Initialize model lazily
        self._init_model()
    
    def _init_model(self):
        """Initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            device = self.config.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._model = SentenceTransformer(self._name, device=device)
            self._dim = self._model.get_sentence_embedding_dimension()
            
        except ImportError:
            raise ImportError("sentence-transformers package required: pip install sentence-transformers")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def dim(self) -> int:
        if self._dim is None:
            self._init_model()
        return self._dim
    
    def batch_size(self, doc_len: int) -> int:
        """Calculate batch size based on memory constraints."""
        # Conservative estimate for GPU memory
        tokens_per_doc = doc_len // 4
        if tokens_per_doc > 512:
            return min(8, self.config.batch_size)
        elif tokens_per_doc > 256:
            return min(16, self.config.batch_size)
        else:
            return self.config.batch_size
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch using HuggingFace model."""
        if not texts:
            return np.array([]).reshape(0, self.dim)
        
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace embedding failed: {e}")


class LlamaCppEmbedder(Embedder):
    """Llama.cpp GGUF embedder implementation for CPU-only inference."""
    
    def __init__(self, config: EmbedderConfig):
        """
        Initialize LlamaCpp embedder.
        
        Args:
            config: Embedder configuration
        """
        self.config = config
        self._name = config.model_name
        self._model = None
        self._dim = None
        
        # Initialize model lazily
        self._init_model()
    
    def _init_model(self):
        """Initialize llama-cpp-python model."""
        try:
            from llama_cpp import Llama
            
            # Load GGUF model with embedding support
            self._model = Llama(
                model_path=self._name,
                embedding=True,
                verbose=False,
                n_ctx=2048,  # Context window
                n_threads=None,  # Use all available cores
            )
            
            # Test with a small text to get dimensions
            test_embedding = self._model.create_embedding("test")
            self._dim = len(test_embedding['data'][0]['embedding'])
            
        except ImportError:
            raise ImportError("llama-cpp-python package required: pip install llama-cpp-python")
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model {self._name}: {e}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def dim(self) -> int:
        if self._dim is None:
            self._init_model()
        return self._dim
    
    def batch_size(self, doc_len: int) -> int:
        """Calculate batch size based on context window constraints."""
        # GGUF models are typically CPU-bound, use smaller batches
        # Estimate tokens (rough 4:1 char to token ratio)
        tokens_per_doc = doc_len // 4
        
        if tokens_per_doc > 1024:
            return min(2, self.config.batch_size)
        elif tokens_per_doc > 512:
            return min(4, self.config.batch_size)
        else:
            return min(8, self.config.batch_size)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch using llama.cpp."""
        if not texts:
            return np.array([]).reshape(0, self.dim)
        
        try:
            embeddings = []
            
            # Process texts individually as llama.cpp doesn't support true batching
            for text in texts:
                result = self._model.create_embedding(text)
                embedding = result['data'][0]['embedding']
                embeddings.append(embedding)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            raise RuntimeError(f"LlamaCpp embedding failed: {e}")


def create_embedder(config: EmbedderConfig) -> Embedder:
    """
    Factory function to create embedder based on configuration.
    
    Args:
        config: Embedder configuration
        
    Returns:
        Embedder instance
        
    Raises:
        ValueError: If model type is not supported
    """
    model_type = config.model_type
    
    if model_type == "openai":
        return OpenAIEmbedder(config)
    elif model_type == "huggingface":
        return HuggingFaceEmbedder(config)
    elif model_type == "llamacpp":
        return LlamaCppEmbedder(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: openai, huggingface, llamacpp")
