"""Tests for embedder components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from sciembed.components.embedder import (
    EmbedderConfig, 
    Embedder,
    OpenAIEmbedder,
    HuggingFaceEmbedder,
    LlamaCppEmbedder,
    create_embedder
)


class TestEmbedderConfig:
    """Test embedder configuration."""
    
    def test_model_type_detection(self):
        """Test model type detection from model string."""
        config = EmbedderConfig(model="openai://text-embedding-3-small")
        assert config.model_type == "openai"
        
        config = EmbedderConfig(model="hf://sentence-transformers/all-MiniLM-L6-v2")
        assert config.model_type == "huggingface"
        
        config = EmbedderConfig(model="gguf://path/to/model.gguf")
        assert config.model_type == "llamacpp"
        
        config = EmbedderConfig(model="unknown-model")
        assert config.model_type == "unknown"
    
    def test_model_name_extraction(self):
        """Test model name extraction."""
        config = EmbedderConfig(model="openai://text-embedding-3-small")
        assert config.model_name == "text-embedding-3-small"
        
        config = EmbedderConfig(model="plain-model-name")
        assert config.model_name == "plain-model-name"


class TestOpenAIEmbedder:
    """Test OpenAI embedder implementation."""
    
    @patch('sciembed.components.embedder.openai')
    def test_initialization(self, mock_openai):
        """Test OpenAI embedder initialization."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        
        config = EmbedderConfig(model="openai://text-embedding-3-small", api_key="test-key")
        embedder = OpenAIEmbedder(config)
        
        assert embedder.name == "text-embedding-3-small"
        assert embedder.dim == 3
        mock_openai.OpenAI.assert_called_once_with(api_key="test-key")
    
    @patch('sciembed.components.embedder.openai')
    def test_embed_batch(self, mock_openai):
        """Test batch embedding."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        
        config = EmbedderConfig(model="openai://text-embedding-3-small")
        embedder = OpenAIEmbedder(config)
        
        texts = ["Hello world", "Another text"]
        embeddings = embedder.embed_batch(texts)
        
        assert embeddings.shape == (2, 3)
        assert embeddings.dtype == np.float32
        np.testing.assert_array_equal(embeddings[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(embeddings[1], [0.4, 0.5, 0.6])
    
    @patch('sciembed.components.embedder.openai')
    def test_batch_size_calculation(self, mock_openai):
        """Test batch size calculation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        
        config = EmbedderConfig(model="openai://test", max_tokens=1000, batch_size=50)
        embedder = OpenAIEmbedder(config)
        
        # Short docs should allow larger batches
        batch_size = embedder.batch_size(100)  # ~25 tokens
        assert batch_size == 40  # min(1000//25, 50)
        
        # Long docs should reduce batch size
        batch_size = embedder.batch_size(2000)  # ~500 tokens
        assert batch_size == 2  # min(1000//500, 50)


class TestHuggingFaceEmbedder:
    """Test HuggingFace embedder implementation."""
    
    @patch('sciembed.components.embedder.SentenceTransformer')
    @patch('sciembed.components.embedder.torch')
    def test_initialization(self, mock_torch, mock_st):
        """Test HuggingFace embedder initialization."""
        mock_torch.cuda.is_available.return_value = True
        
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        config = EmbedderConfig(model="hf://sentence-transformers/all-MiniLM-L6-v2")
        embedder = HuggingFaceEmbedder(config)
        
        assert embedder.name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.dim == 384
        mock_st.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    
    @patch('sciembed.components.embedder.SentenceTransformer')
    @patch('sciembed.components.embedder.torch')
    def test_embed_batch(self, mock_torch, mock_st):
        """Test batch embedding."""
        mock_torch.cuda.is_available.return_value = False
        
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mock_st.return_value = mock_model
        
        config = EmbedderConfig(model="hf://test-model")
        embedder = HuggingFaceEmbedder(config)
        
        texts = ["Hello world", "Another text"]
        embeddings = embedder.embed_batch(texts)
        
        assert embeddings.shape == (2, 2)
        assert embeddings.dtype == np.float32
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=2,
            show_progress_bar=False,
            convert_to_numpy=True
        )


class TestLlamaCppEmbedder:
    """Test LlamaCpp embedder implementation."""
    
    @patch('sciembed.components.embedder.Llama')
    def test_initialization(self, mock_llama):
        """Test LlamaCpp embedder initialization."""
        mock_model = Mock()
        mock_model.create_embedding.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3, 0.4]}]
        }
        mock_llama.return_value = mock_model
        
        config = EmbedderConfig(model="gguf://path/to/model.gguf")
        embedder = LlamaCppEmbedder(config)
        
        assert embedder.name == "path/to/model.gguf"
        assert embedder.dim == 4
        mock_llama.assert_called_once_with(
            model_path="path/to/model.gguf",
            embedding=True,
            verbose=False,
            n_ctx=2048,
            n_threads=None
        )
    
    @patch('sciembed.components.embedder.Llama')
    def test_embed_batch(self, mock_llama):
        """Test batch embedding."""
        mock_model = Mock()
        # Mock responses for each text
        mock_model.create_embedding.side_effect = [
            {'data': [{'embedding': [0.1, 0.2, 0.3]}]},
            {'data': [{'embedding': [0.4, 0.5, 0.6]}]}
        ]
        mock_llama.return_value = mock_model
        
        config = EmbedderConfig(model="gguf://test.gguf")
        embedder = LlamaCppEmbedder(config)
        
        texts = ["Hello world", "Another text"]
        embeddings = embedder.embed_batch(texts)
        
        assert embeddings.shape == (2, 3)
        assert embeddings.dtype == np.float32
        np.testing.assert_array_equal(embeddings[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(embeddings[1], [0.4, 0.5, 0.6])
        
        # Verify individual calls
        assert mock_model.create_embedding.call_count == 2
    
    @patch('sciembed.components.embedder.Llama')
    def test_batch_size_calculation(self, mock_llama):
        """Test batch size calculation for CPU-bound processing."""
        mock_model = Mock()
        mock_model.create_embedding.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}]
        }
        mock_llama.return_value = mock_model
        
        config = EmbedderConfig(model="gguf://test.gguf", batch_size=16)
        embedder = LlamaCppEmbedder(config)
        
        # Very long documents should use smallest batch size
        batch_size = embedder.batch_size(5000)  # >1024 tokens
        assert batch_size == 2
        
        # Medium documents
        batch_size = embedder.batch_size(2000)  # >512 tokens  
        assert batch_size == 4
        
        # Short documents
        batch_size = embedder.batch_size(1000)  # <512 tokens
        assert batch_size == 8


class TestEmbedderFactory:
    """Test embedder factory function."""
    
    @patch('sciembed.components.embedder.OpenAIEmbedder')
    def test_create_openai_embedder(self, mock_openai_embedder):
        """Test creating OpenAI embedder."""
        config = EmbedderConfig(model="openai://text-embedding-3-small")
        create_embedder(config)
        mock_openai_embedder.assert_called_once_with(config)
    
    @patch('sciembed.components.embedder.HuggingFaceEmbedder')
    def test_create_huggingface_embedder(self, mock_hf_embedder):
        """Test creating HuggingFace embedder."""
        config = EmbedderConfig(model="hf://sentence-transformers/all-MiniLM-L6-v2")
        create_embedder(config)
        mock_hf_embedder.assert_called_once_with(config)
    
    @patch('sciembed.components.embedder.LlamaCppEmbedder')
    def test_create_llamacpp_embedder(self, mock_llama_embedder):
        """Test creating LlamaCpp embedder."""
        config = EmbedderConfig(model="gguf://path/to/model.gguf")
        create_embedder(config)
        mock_llama_embedder.assert_called_once_with(config)
    
    def test_unsupported_model_type(self):
        """Test error for unsupported model type."""
        config = EmbedderConfig(model="unsupported://model")
        with pytest.raises(ValueError, match="Unsupported model type"):
            create_embedder(config)


class TestEmbedderErrorHandling:
    """Test error handling in embedders."""
    
    @patch('sciembed.components.embedder.openai')
    def test_openai_import_error(self, mock_openai):
        """Test handling of missing OpenAI package."""
        mock_openai.side_effect = ImportError("No module named 'openai'")
        
        config = EmbedderConfig(model="openai://test")
        with pytest.raises(ImportError, match="OpenAI package required"):
            OpenAIEmbedder(config)
    
    @patch('sciembed.components.embedder.SentenceTransformer')
    def test_huggingface_import_error(self, mock_st):
        """Test handling of missing sentence-transformers package."""
        mock_st.side_effect = ImportError("No module named 'sentence_transformers'")
        
        config = EmbedderConfig(model="hf://test")
        with pytest.raises(ImportError, match="sentence-transformers package required"):
            HuggingFaceEmbedder(config)
    
    @patch('sciembed.components.embedder.Llama')
    def test_llamacpp_import_error(self, mock_llama):
        """Test handling of missing llama-cpp-python package."""
        mock_llama.side_effect = ImportError("No module named 'llama_cpp'")
        
        config = EmbedderConfig(model="gguf://test")
        with pytest.raises(ImportError, match="llama-cpp-python package required"):
            LlamaCppEmbedder(config)
    
    @patch('sciembed.components.embedder.openai')
    def test_openai_api_error(self, mock_openai):
        """Test handling of OpenAI API errors."""
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client
        
        config = EmbedderConfig(model="openai://test")
        embedder = OpenAIEmbedder(config)
        
        with pytest.raises(RuntimeError, match="OpenAI embedding failed"):
            embedder.embed_batch(["test"])


class TestEmbedderIntegration:
    """Integration tests for embedders."""
    
    def test_embed_single_convenience_method(self):
        """Test the embed_single convenience method."""
        # Create a mock embedder
        class MockEmbedder(Embedder):
            @property
            def name(self) -> str:
                return "mock"
            
            @property
            def dim(self) -> int:
                return 3
            
            def batch_size(self, doc_len: int) -> int:
                return 1
            
            def embed_batch(self, texts):
                return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)
        
        embedder = MockEmbedder()
        result = embedder.embed_single("test text")
        
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3])
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        class MockEmbedder(Embedder):
            @property
            def name(self) -> str:
                return "mock"
            
            @property
            def dim(self) -> int:
                return 3
            
            def batch_size(self, doc_len: int) -> int:
                return 1
            
            def embed_batch(self, texts):
                if not texts:
                    return np.array([]).reshape(0, self.dim)
                return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)
        
        embedder = MockEmbedder()
        result = embedder.embed_batch([])
        
        assert result.shape == (0, 3)
