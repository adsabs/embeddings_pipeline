"""Tests for the text preparation component."""

import pytest
from typing import Dict, Any, List

from src.sciembed.components.preparer import Preparer, PreparerConfig


class TestPreparerConfig:
    """Test PreparerConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PreparerConfig(fields=["title", "abstract"])
        
        assert config.fields == ["title", "abstract"]
        assert config.prefix == ""
        assert config.suffix == ""
        assert config.delimiter == "\n\n"
        assert config.lowercase is True
        assert config.truncate == 3000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PreparerConfig(
            fields=["title"],
            prefix="Prefix: ",
            suffix=" :Suffix",
            delimiter=" | ",
            lowercase=False,
            truncate=1000
        )
        
        assert config.fields == ["title"]
        assert config.prefix == "Prefix: "
        assert config.suffix == " :Suffix"
        assert config.delimiter == " | "
        assert config.lowercase is False
        assert config.truncate == 1000
    
    def test_config_hash(self):
        """Test configuration hashing for reproducibility."""
        config1 = PreparerConfig(fields=["title", "abstract"])
        config2 = PreparerConfig(fields=["title", "abstract"])
        config3 = PreparerConfig(fields=["title"])
        
        # Same configs should have same hash
        assert config1.hash() == config2.hash()
        
        # Different configs should have different hashes
        assert config1.hash() != config3.hash()
        
        # Hash should be 16 characters
        assert len(config1.hash()) == 16


class TestPreparer:
    """Test Preparer functionality."""
    
    @pytest.fixture
    def sample_record(self) -> Dict[str, Any]:
        """Sample scientific record for testing."""
        return {
            "bibcode": "2023ApJ...123..456S",
            "title": "A Study of Stellar Formation",
            "abstract": "This paper investigates stellar formation processes in nearby galaxies. We analyzed data from the Hubble Space Telescope.",
            "authors": ["Smith, J.", "Doe, A."],
            "year": 2023
        }
    
    @pytest.fixture
    def basic_config(self) -> PreparerConfig:
        """Basic configuration for testing."""
        return PreparerConfig(fields=["title", "abstract"])
    
    def test_basic_text_preparation(self, sample_record, basic_config):
        """Test basic text preparation functionality."""
        preparer = Preparer(basic_config)
        bibcode, text = preparer.prepare_record(sample_record)
        
        assert bibcode == "2023ApJ...123..456S"
        
        expected_text = "a study of stellar formation\n\nthis paper investigates stellar formation processes in nearby galaxies. we analyzed data from the hubble space telescope."
        assert text == expected_text
    
    def test_prefix_and_suffix(self, sample_record):
        """Test prefix and suffix addition."""
        config = PreparerConfig(
            fields=["title"],
            prefix="You are an astrophysicist. ",
            suffix=" Please analyze this."
        )
        preparer = Preparer(config)
        bibcode, text = preparer.prepare_record(sample_record)
        
        expected_text = "You are an astrophysicist. \n\na study of stellar formation\n\n Please analyze this."
        assert text == expected_text
    
    def test_custom_delimiter(self, sample_record):
        """Test custom delimiter."""
        config = PreparerConfig(
            fields=["title", "abstract"],
            delimiter=" | "
        )
        preparer = Preparer(config)
        bibcode, text = preparer.prepare_record(sample_record)
        
        expected_text = "a study of stellar formation | this paper investigates stellar formation processes in nearby galaxies. we analyzed data from the hubble space telescope."
        assert text == expected_text
    
    def test_no_lowercase(self, sample_record):
        """Test text preparation without lowercasing."""
        config = PreparerConfig(
            fields=["title"],
            lowercase=False
        )
        preparer = Preparer(config)
        bibcode, text = preparer.prepare_record(sample_record)
        
        assert text == "A Study of Stellar Formation"
    
    def test_text_truncation(self, sample_record):
        """Test text truncation."""
        config = PreparerConfig(
            fields=["title", "abstract"],
            truncate=50
        )
        preparer = Preparer(config)
        bibcode, text = preparer.prepare_record(sample_record)
        
        assert len(text) == 50
        # Should truncate at character 50
        expected_start = "a study of stellar formation\n\nthis paper investiga"
        assert text == expected_start
    
    def test_missing_fields(self, basic_config):
        """Test handling of missing fields."""
        record = {
            "bibcode": "2023ApJ...123..456S",
            "title": "Test Title"
            # abstract missing
        }
        
        preparer = Preparer(basic_config)
        bibcode, text = preparer.prepare_record(record)
        
        assert bibcode == "2023ApJ...123..456S"
        assert text == "test title"
    
    def test_empty_field_values(self, basic_config):
        """Test handling of empty field values."""
        record = {
            "bibcode": "2023ApJ...123..456S",
            "title": "Test Title",
            "abstract": ""  # empty abstract
        }
        
        preparer = Preparer(basic_config)
        bibcode, text = preparer.prepare_record(record)
        
        assert text == "test title"
    
    def test_missing_bibcode_error(self, basic_config):
        """Test error when bibcode is missing."""
        record = {
            "title": "Test Title",
            "abstract": "Test Abstract"
            # bibcode missing
        }
        
        preparer = Preparer(basic_config)
        
        with pytest.raises(KeyError, match="Record missing required 'bibcode' field"):
            preparer.prepare_record(record)
    
    def test_no_valid_fields_error(self, basic_config):
        """Test error when no valid fields are found."""
        record = {
            "bibcode": "2023ApJ...123..456S",
            "year": 2023
            # no title or abstract
        }
        
        preparer = Preparer(basic_config)
        
        with pytest.raises(ValueError, match="No content found in specified fields"):
            preparer.prepare_record(record)
    
    def test_only_prefix_suffix_error(self):
        """Test error when only prefix/suffix exist without content."""
        config = PreparerConfig(
            fields=["title", "abstract"],
            prefix="Prefix: ",
            suffix=" :Suffix"
        )
        record = {
            "bibcode": "2023ApJ...123..456S",
            "year": 2023
            # no title or abstract
        }
        
        preparer = Preparer(config)
        
        with pytest.raises(ValueError, match="No content found in specified fields"):
            preparer.prepare_record(record)
    
    def test_batch_preparation(self, basic_config):
        """Test batch text preparation."""
        records = [
            {
                "bibcode": "2023ApJ...123..456S",
                "title": "First Paper",
                "abstract": "First abstract"
            },
            {
                "bibcode": "2023ApJ...789..012D",
                "title": "Second Paper",
                "abstract": "Second abstract"
            }
        ]
        
        preparer = Preparer(basic_config)
        results = preparer.prepare_batch(records)
        
        assert len(results) == 2
        assert results[0][0] == "2023ApJ...123..456S"
        assert results[0][1] == "first paper\n\nfirst abstract"
        assert results[1][0] == "2023ApJ...789..012D"
        assert results[1][1] == "second paper\n\nsecond abstract"
    
    def test_batch_with_errors(self, basic_config):
        """Test batch preparation with some invalid records."""
        records = [
            {
                "bibcode": "2023ApJ...123..456S",
                "title": "Valid Paper",
                "abstract": "Valid abstract"
            },
            {
                # Missing bibcode
                "title": "Invalid Paper",
                "abstract": "Invalid abstract"
            },
            {
                "bibcode": "2023ApJ...789..012D",
                "title": "Another Valid Paper",
                "abstract": "Another valid abstract"
            }
        ]
        
        preparer = Preparer(basic_config)
        results = preparer.prepare_batch(records)
        
        # Should skip the invalid record
        assert len(results) == 2
        assert results[0][0] == "2023ApJ...123..456S"
        assert results[1][0] == "2023ApJ...789..012D"
    
    def test_stream_preparation(self, basic_config):
        """Test stream text preparation."""
        def record_generator():
            yield {
                "bibcode": "2023ApJ...123..456S",
                "title": "First Paper",
                "abstract": "First abstract"
            }
            yield {
                "bibcode": "2023ApJ...789..012D", 
                "title": "Second Paper",
                "abstract": "Second abstract"
            }
        
        preparer = Preparer(basic_config)
        results = list(preparer.prepare_stream(record_generator()))
        
        assert len(results) == 2
        assert results[0][0] == "2023ApJ...123..456S"
        assert results[1][0] == "2023ApJ...789..012D"
    
    def test_stream_with_errors(self, basic_config):
        """Test stream preparation with some invalid records."""
        def record_generator():
            yield {
                "bibcode": "2023ApJ...123..456S",
                "title": "Valid Paper",
                "abstract": "Valid abstract"
            }
            yield {
                # Missing bibcode
                "title": "Invalid Paper",
                "abstract": "Invalid abstract"
            }
            yield {
                "bibcode": "2023ApJ...789..012D",
                "title": "Another Valid Paper",
                "abstract": "Another valid abstract"
            }
        
        preparer = Preparer(basic_config)
        results = list(preparer.prepare_stream(record_generator()))
        
        # Should skip the invalid record
        assert len(results) == 2
        assert results[0][0] == "2023ApJ...123..456S"
        assert results[1][0] == "2023ApJ...789..012D"
    
    def test_text_hash(self, basic_config):
        """Test text hashing for deduplication."""
        preparer = Preparer(basic_config)
        
        text1 = "sample text for hashing"
        text2 = "sample text for hashing"
        text3 = "different text for hashing"
        
        hash1 = preparer.get_text_hash(text1)
        hash2 = preparer.get_text_hash(text2)
        hash3 = preparer.get_text_hash(text3)
        
        # Same text should have same hash
        assert hash1 == hash2
        
        # Different text should have different hash
        assert hash1 != hash3
        
        # Hash should be 64 characters (SHA256 hex)
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)
    
    def test_multiple_field_types(self, basic_config):
        """Test handling of different field types."""
        record = {
            "bibcode": "2023ApJ...123..456S",
            "title": "Test Title",
            "abstract": 12345,  # numeric value
            "year": 2023
        }
        
        preparer = Preparer(basic_config)
        bibcode, text = preparer.prepare_record(record)
        
        # Numeric fields should be converted to strings
        assert "12345" in text
    
    def test_astrophysics_realistic_example(self):
        """Test with realistic astrophysics paper data."""
        config = PreparerConfig(
            fields=["title", "abstract"],
            prefix="You are an astrophysicist. ",
            delimiter="\n\n"
        )
        
        record = {
            "bibcode": "2023ApJ...945..123A",
            "title": "Dark Matter Substructure in Galaxy Clusters: A Machine Learning Approach",
            "abstract": "We present a novel machine learning methodology for detecting dark matter substructure in galaxy clusters using gravitational lensing data. Our deep neural network architecture achieves 95% accuracy in identifying subhalos with masses above 10^9 solar masses. The results have implications for understanding the nature of dark matter and cluster formation history."
        }
        
        preparer = Preparer(config)
        bibcode, text = preparer.prepare_record(record)
        
        assert bibcode == "2023ApJ...945..123A"
        assert text.startswith("You are an astrophysicist. ")
        assert "dark matter substructure" in text
        assert "machine learning" in text
        assert "gravitational lensing" in text


class TestPreparerIntegration:
    """Test integration aspects of the preparer."""
    
    def test_config_reproducibility(self):
        """Test that identical configs produce identical results."""
        config1 = PreparerConfig(
            fields=["title", "abstract"],
            prefix="Test: ",
            lowercase=False
        )
        config2 = PreparerConfig(
            fields=["title", "abstract"],
            prefix="Test: ",
            lowercase=False
        )
        
        record = {
            "bibcode": "2023ApJ...123..456S",
            "title": "Test Title",
            "abstract": "Test Abstract"
        }
        
        preparer1 = Preparer(config1)
        preparer2 = Preparer(config2)
        
        result1 = preparer1.prepare_record(record)
        result2 = preparer2.prepare_record(record)
        
        assert result1 == result2
        assert config1.hash() == config2.hash()
    
    def test_performance_with_large_text(self):
        """Test performance with large text fields."""
        # Create a record with very large abstract
        large_abstract = "This is a test abstract. " * 1000  # ~25KB
        
        config = PreparerConfig(
            fields=["title", "abstract"],
            truncate=5000
        )
        
        record = {
            "bibcode": "2023ApJ...123..456S",
            "title": "Test Title",
            "abstract": large_abstract
        }
        
        preparer = Preparer(config)
        bibcode, text = preparer.prepare_record(record)
        
        # Should be truncated
        assert len(text) == 5000
        assert bibcode == "2023ApJ...123..456S"
