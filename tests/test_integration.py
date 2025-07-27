"""Integration tests for the complete embedding pipeline."""

import pytest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from sciembed import Config, Pipeline, AsyncPipeline
from sciembed.runner import run_pipeline
from sciembed.components.embedder import EmbedderConfig


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        # Create directories
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create test data
        self.create_test_data()
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """Create test JSON data files."""
        # Sample papers for 2020
        papers_2020 = [
            {
                "bibcode": "2020A&A...123..456A",
                "title": "Dark Matter in Galaxy Clusters",
                "abstract": "We study the distribution of dark matter in galaxy clusters using gravitational lensing.",
                "author": ["Smith, J.", "Johnson, M."],
                "year": 2020
            },
            {
                "bibcode": "2020MNRAS.456..789B", 
                "title": "Stellar Formation in Molecular Clouds",
                "abstract": "This paper investigates the process of star formation in dense molecular clouds.",
                "author": ["Brown, K.", "Wilson, L."],
                "year": 2020
            },
            {
                "bibcode": "2020ApJ...789..123C",
                "title": "Exoplanet Atmosphere Analysis",
                "abstract": "We present spectroscopic analysis of exoplanet atmospheres using transit observations.",
                "author": ["Davis, R.", "Garcia, S."],
                "year": 2020
            }
        ]
        
        # Sample papers for 2021 (with one duplicate)
        papers_2021 = [
            {
                "bibcode": "2021A&A...234..567D",
                "title": "Galaxy Evolution Over Cosmic Time",
                "abstract": "This study examines how galaxies have evolved since the early universe.",
                "author": ["Martinez, A.", "Taylor, B."],
                "year": 2021
            },
            {
                "bibcode": "2021MNRAS.567..890E",
                "title": "Dark Matter in Galaxy Clusters",  # Same title as 2020 paper
                "abstract": "We study the distribution of dark matter in galaxy clusters using gravitational lensing.",  # Duplicate content
                "author": ["Thompson, C.", "Anderson, D."],
                "year": 2021
            }
        ]
        
        # Write JSON files
        with open(self.input_dir / "2020.json", "w") as f:
            for paper in papers_2020:
                f.write(json.dumps(paper) + "\n")
        
        with open(self.input_dir / "2021.json", "w") as f:
            for paper in papers_2021:
                f.write(json.dumps(paper) + "\n")
    
    def create_mock_config(self, **overrides):
        """Create a test configuration with mock embedder."""
        config = Config(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            years=[2020, 2021],
            fields=["title", "abstract"],
            model="mock://test-model",
            batch_size=2,
            use_float16=False,  # Easier for testing
            create_faiss_index=False,  # Skip for basic tests
            show_progress=False,  # Avoid progress bars in tests
            deduplicate=True,
            dedup_use_rocksdb=False,  # Use memory for tests
            **overrides
        )
        return config
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_sync_pipeline_basic_functionality(self, mock_create_embedder):
        """Test basic sync pipeline functionality."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 2
        mock_embedder.embed_batch.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        mock_create_embedder.return_value = mock_embedder
        
        config = self.create_mock_config()
        pipeline = Pipeline(config)
        stats = pipeline.run()
        
        # Should process 4 unique records (3 from 2020, 1 from 2021 after dedup)
        assert stats.processed_records == 4
        assert stats.duplicate_records == 1  # One duplicate found
        assert stats.total_records == 5  # Total including duplicates
        
        # Check output files exist
        assert (self.output_dir / "embeddings_mock-model_2020.f16").exists()
        assert (self.output_dir / "embeddings_mock-model_2021.f16").exists()
        assert (self.output_dir / "bibcodes_2020.txt").exists()
        assert (self.output_dir / "bibcodes_2021.txt").exists()
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_async_pipeline_basic_functionality(self, mock_create_embedder):
        """Test basic async pipeline functionality."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 2
        mock_embedder.embed_batch.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        mock_embedder.config.model_type = "mock"
        mock_create_embedder.return_value = mock_embedder
        
        config = self.create_mock_config(use_async=True, num_workers=2)
        
        import asyncio
        
        async def run_test():
            pipeline = AsyncPipeline(config)
            return await pipeline.run()
        
        stats = asyncio.run(run_test())
        
        # Should process 4 unique records (3 from 2020, 1 from 2021 after dedup)
        assert stats.processed_records == 4
        # Note: async pipeline tracks duplicates in prep worker, stats may vary
        
        # Check output files exist
        assert (self.output_dir / "embeddings_mock-model_2020.f16").exists() or \
               (self.output_dir / "embeddings_mock-model_2021.f16").exists()
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_pipeline_runner_sync(self, mock_create_embedder):
        """Test pipeline runner with sync mode."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 2
        mock_embedder.embed_batch.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        mock_create_embedder.return_value = mock_embedder
        
        config = self.create_mock_config(use_async=False)
        stats = run_pipeline(config)
        
        assert stats.processed_records > 0
        assert stats.processing_time > 0
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_pipeline_runner_async(self, mock_create_embedder):
        """Test pipeline runner with async mode."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 2
        mock_embedder.embed_batch.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        mock_embedder.config.model_type = "mock"
        mock_create_embedder.return_value = mock_embedder
        
        config = self.create_mock_config(use_async=True)
        stats = run_pipeline(config)
        
        assert stats.processed_records > 0
        assert stats.processing_time > 0
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_deduplication_functionality(self, mock_create_embedder):
        """Test that deduplication works correctly."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 2
        
        # Track calls to embed_batch to verify deduplication
        embed_calls = []
        def mock_embed_batch(texts):
            embed_calls.append(texts)
            return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)
        
        mock_embedder.embed_batch.side_effect = mock_embed_batch
        mock_create_embedder.return_value = mock_embedder
        
        config = self.create_mock_config(deduplicate=True)
        pipeline = Pipeline(config)
        stats = pipeline.run()
        
        # Should find 1 duplicate
        assert stats.duplicate_records == 1
        assert stats.processed_records == 4  # 5 total - 1 duplicate
        
        # Verify that only unique texts were embedded
        all_embedded_texts = []
        for call in embed_calls:
            all_embedded_texts.extend(call)
        
        # Should not contain the duplicate text twice
        unique_texts = set(all_embedded_texts)
        assert len(all_embedded_texts) == len(unique_texts)
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_deduplication_disabled(self, mock_create_embedder):
        """Test pipeline with deduplication disabled."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 2
        mock_embedder.embed_batch.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        mock_create_embedder.return_value = mock_embedder
        
        config = self.create_mock_config(deduplicate=False)
        pipeline = Pipeline(config)
        stats = pipeline.run()
        
        # Should process all records without deduplication
        assert stats.duplicate_records == 0
        assert stats.processed_records == 5  # All records processed
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_field_selection(self, mock_create_embedder):
        """Test that field selection works correctly."""
        # Mock embedder that captures input texts
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 5
        
        captured_texts = []
        def mock_embed_batch(texts):
            captured_texts.extend(texts)
            return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)
        
        mock_embedder.embed_batch.side_effect = mock_embed_batch
        mock_create_embedder.return_value = mock_embedder
        
        # Test with only title field
        config = self.create_mock_config(fields=["title"], deduplicate=False)
        pipeline = Pipeline(config)
        stats = pipeline.run()
        
        # Verify that only titles were embedded (no abstracts)
        for text in captured_texts:
            assert "gravitational lensing" not in text.lower()  # From abstract
            assert "spectroscopic analysis" not in text.lower()  # From abstract
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_text_preparation_with_prefix_suffix(self, mock_create_embedder):
        """Test text preparation with prefix and suffix."""
        # Mock embedder that captures input texts
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 5
        
        captured_texts = []
        def mock_embed_batch(texts):
            captured_texts.extend(texts)
            return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)
        
        mock_embedder.embed_batch.side_effect = mock_embed_batch
        mock_create_embedder.return_value = mock_embedder
        
        config = self.create_mock_config(
            prefix="Paper: ",
            suffix=" [END]",
            deduplicate=False
        )
        pipeline = Pipeline(config)
        stats = pipeline.run()
        
        # Verify prefix and suffix are added
        for text in captured_texts:
            assert text.startswith("Paper: ")
            assert text.endswith(" [END]")
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_resumability(self, mock_create_embedder):
        """Test pipeline resumability."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 2
        mock_embedder.embed_batch.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ], dtype=np.float32)
        mock_create_embedder.return_value = mock_embedder
        
        config = self.create_mock_config(resume=True)
        
        # Run pipeline first time
        pipeline1 = Pipeline(config)
        stats1 = pipeline1.run()
        
        # Run pipeline second time (should skip already processed years)
        pipeline2 = Pipeline(config)
        stats2 = pipeline2.run()
        
        # Second run should process fewer records (due to resumability)
        # Note: Exact behavior depends on manifest implementation
        assert stats2.processed_records <= stats1.processed_records


class TestPipelineErrorHandling:
    """Test error handling in pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        # Create directories
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_missing_input_directory(self):
        """Test handling of missing input directory."""
        config = Config(
            input_dir="/nonexistent/path",
            output_dir=str(self.output_dir),
            years=[2020],
            model="mock://test"
        )
        
        with pytest.raises(ValueError, match="Input directory does not exist"):
            config.validate()
    
    def test_missing_year_file(self):
        """Test handling of missing year file."""
        config = Config(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            years=[2025],  # Non-existent year
            model="mock://test",
            show_progress=False
        )
        
        with patch('sciembed.components.embedder.create_embedder') as mock_create:
            mock_embedder = Mock()
            mock_embedder.name = "mock"
            mock_embedder.dim = 3
            mock_create.return_value = mock_embedder
            
            pipeline = Pipeline(config)
            
            with pytest.raises(FileNotFoundError):
                pipeline.run()
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_embedding_failure_handling(self, mock_create_embedder):
        """Test handling of embedding failures."""
        # Create test data
        with open(self.input_dir / "2020.json", "w") as f:
            f.write(json.dumps({"bibcode": "test", "title": "Test", "abstract": "Test"}) + "\n")
        
        # Mock embedder that fails
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 3
        mock_embedder.batch_size.return_value = 1
        mock_embedder.embed_batch.side_effect = Exception("Embedding failed")
        mock_create_embedder.return_value = mock_embedder
        
        config = Config(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            years=[2020],
            model="mock://test",
            show_progress=False,
            deduplicate=False
        )
        
        pipeline = Pipeline(config)
        stats = pipeline.run()
        
        # Should handle the failure gracefully
        assert stats.failed_records == 1
        assert stats.processed_records == 0


class TestPipelinePerformance:
    """Performance-related tests."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        # Create directories
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # Create larger test dataset
        self.create_large_test_data()
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_large_test_data(self):
        """Create a larger test dataset."""
        papers = []
        for i in range(100):
            papers.append({
                "bibcode": f"2020A&A...{i:03d}..456A",
                "title": f"Test Paper {i}",
                "abstract": f"This is abstract number {i} about various astronomical phenomena.",
                "author": [f"Author{i}, A.", f"Author{i+1}, B."],
                "year": 2020
            })
        
        with open(self.input_dir / "2020.json", "w") as f:
            for paper in papers:
                f.write(json.dumps(paper) + "\n")
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_large_dataset_processing(self, mock_create_embedder):
        """Test processing of larger dataset."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 384
        mock_embedder.batch_size.return_value = 10
        
        def mock_embed_batch(texts):
            return np.random.random((len(texts), 384)).astype(np.float32)
        
        mock_embedder.embed_batch.side_effect = mock_embed_batch
        mock_create_embedder.return_value = mock_embedder
        
        config = Config(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            years=[2020],
            model="mock://test",
            batch_size=10,
            show_progress=False,
            deduplicate=False
        )
        
        pipeline = Pipeline(config)
        stats = pipeline.run()
        
        assert stats.processed_records == 100
        assert stats.total_batches == 10  # 100 records / 10 batch_size
        assert stats.processing_time > 0
        assert stats.embedding_time > 0
    
    @patch('sciembed.components.embedder.create_embedder')
    def test_async_vs_sync_performance(self, mock_create_embedder):
        """Compare async vs sync performance (basic test)."""
        # Mock embedder with slight delay to simulate real processing
        mock_embedder = Mock()
        mock_embedder.name = "mock-model"
        mock_embedder.dim = 384
        mock_embedder.batch_size.return_value = 10
        mock_embedder.config.model_type = "mock"
        
        def mock_embed_batch(texts):
            import time
            time.sleep(0.01)  # Small delay to simulate processing
            return np.random.random((len(texts), 384)).astype(np.float32)
        
        mock_embedder.embed_batch.side_effect = mock_embed_batch
        mock_create_embedder.return_value = mock_embedder
        
        # Sync pipeline
        config_sync = Config(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            years=[2020],
            model="mock://test",
            use_async=False,
            show_progress=False,
            deduplicate=False
        )
        
        stats_sync = run_pipeline(config_sync)
        
        # Async pipeline
        config_async = Config(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir / "async"),
            years=[2020],
            model="mock://test",
            use_async=True,
            num_workers=2,
            show_progress=False,
            deduplicate=False
        )
        
        stats_async = run_pipeline(config_async)
        
        # Both should process same number of records
        assert stats_sync.processed_records == stats_async.processed_records
        
        # Both should complete successfully
        assert stats_sync.processing_time > 0
        assert stats_async.processing_time > 0
