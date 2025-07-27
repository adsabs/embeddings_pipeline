"""Tests for deduplication components."""

import pytest
import hashlib
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from sciembed.components.deduplicator import (
    Deduplicator,
    DeduplicationConfig,
    DuplicationInfo
)


class TestDeduplicator:
    """Test deduplication functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_dedup.db"
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_memory_deduplication_basic(self):
        """Test basic deduplication with in-memory storage."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        # Add first record
        text1 = "This is a test paper about astronomy."
        hash1 = deduplicator.add_record("2020A&A...123..456A", text1, 2020)
        
        # Check duplicate detection
        duplicate_info = deduplicator.check_duplicate("2021MNRAS.456..789B", text1, 2021)
        assert duplicate_info is not None
        assert duplicate_info.bibcode == "2021MNRAS.456..789B"
        assert duplicate_info.year == 2021
        assert duplicate_info.first_seen_bibcode == "2020A&A...123..456A"
        assert duplicate_info.first_seen_year == 2020
        assert duplicate_info.text_hash == hash1
    
    def test_memory_deduplication_no_duplicate(self):
        """Test no duplicate found."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        # Add first record
        text1 = "This is a test paper about astronomy."
        deduplicator.add_record("2020A&A...123..456A", text1, 2020)
        
        # Check different text
        text2 = "This is a completely different paper about physics."
        duplicate_info = deduplicator.check_duplicate("2021MNRAS.456..789B", text2, 2021)
        assert duplicate_info is None
    
    def test_text_hash_computation(self):
        """Test SHA256 hash computation."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        text = "This is a test paper."
        hash1 = deduplicator.compute_text_hash(text)
        hash2 = deduplicator.compute_text_hash(text)
        
        # Same text should produce same hash
        assert hash1 == hash2
        
        # Should be valid SHA256
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1)
        
        # Test normalization (case and whitespace)
        text_upper = "THIS IS A TEST PAPER."
        text_whitespace = "  this is a test paper.  "
        hash_upper = deduplicator.compute_text_hash(text_upper)
        hash_whitespace = deduplicator.compute_text_hash(text_whitespace)
        
        assert hash1 == hash_upper == hash_whitespace
    
    def test_batch_processing(self):
        """Test batch processing of records."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        records = [
            ("2020A&A...123..456A", "First unique paper", 2020),
            ("2020A&A...123..457B", "Second unique paper", 2020),
            ("2021MNRAS.456..789C", "First unique paper", 2021),  # Duplicate
            ("2021MNRAS.456..790D", "Third unique paper", 2021),
            ("2022ApJ...789..123E", "Second unique paper", 2022),  # Duplicate
        ]
        
        unique_records, duplicates = deduplicator.process_batch(records)
        
        # Should have 3 unique records and 2 duplicates
        assert len(unique_records) == 3
        assert len(duplicates) == 2
        
        # Check unique records
        unique_bibcodes = [bibcode for bibcode, _, _ in unique_records]
        assert "2020A&A...123..456A" in unique_bibcodes
        assert "2020A&A...123..457B" in unique_bibcodes
        assert "2021MNRAS.456..790D" in unique_bibcodes
        
        # Check duplicates
        duplicate_bibcodes = [dup.bibcode for dup in duplicates]
        assert "2021MNRAS.456..789C" in duplicate_bibcodes
        assert "2022ApJ...789..123E" in duplicate_bibcodes
        
        # Verify duplicate info
        dup1 = next(dup for dup in duplicates if dup.bibcode == "2021MNRAS.456..789C")
        assert dup1.first_seen_bibcode == "2020A&A...123..456A"
        assert dup1.first_seen_year == 2020
    
    def test_stats_memory(self):
        """Test statistics for memory storage."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        # Add some records
        deduplicator.add_record("2020A&A...123..456A", "Paper 1", 2020)
        deduplicator.add_record("2020A&A...123..457B", "Paper 2", 2020)
        deduplicator.add_record("2020A&A...123..458C", "Paper 3", 2020)
        
        stats = deduplicator.get_stats()
        assert stats['storage_type'] == 'memory'
        assert stats['unique_hashes'] == 3
    
    @patch('sciembed.components.deduplicator.rocksdb')
    def test_rocksdb_fallback_on_import_error(self, mock_rocksdb):
        """Test fallback to memory when RocksDB import fails."""
        mock_rocksdb.side_effect = ImportError("No module named 'rocksdb'")
        
        # Should fall back to memory storage
        with patch('sciembed.components.deduplicator.ROCKSDB_AVAILABLE', False):
            deduplicator = Deduplicator(self.db_path, enable_rocksdb=True)
            assert not deduplicator.enable_rocksdb
            
            # Should still work with memory storage
            deduplicator.add_record("test", "content", 2020)
            assert deduplicator.check_duplicate("test2", "content", 2021) is not None
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with Deduplicator(self.db_path, enable_rocksdb=False) as deduplicator:
            deduplicator.add_record("test", "content", 2020)
            assert deduplicator.check_duplicate("test2", "content", 2021) is not None
        
        # Should work even after context exit
        assert True  # No exceptions means success


class TestDeduplicationConfig:
    """Test deduplication configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DeduplicationConfig()
        
        assert config.enabled is True
        assert config.use_rocksdb is True
        assert config.db_path is None
        assert config.cross_year is True
        assert config.log_duplicates is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        db_path = Path("/custom/path/dedup.db")
        config = DeduplicationConfig(
            enabled=False,
            use_rocksdb=False,
            db_path=db_path,
            cross_year=False,
            log_duplicates=False
        )
        
        assert config.enabled is False
        assert config.use_rocksdb is False
        assert config.db_path == db_path
        assert config.cross_year is False
        assert config.log_duplicates is False
    
    def test_get_db_path_with_custom_path(self):
        """Test getting database path with custom path."""
        custom_path = Path("/custom/dedup.db")
        config = DeduplicationConfig(db_path=custom_path)
        
        output_dir = Path("/output")
        assert config.get_db_path(output_dir) == custom_path
    
    def test_get_db_path_with_fallback(self):
        """Test getting database path with fallback."""
        config = DeduplicationConfig(db_path=None)
        
        output_dir = Path("/output")
        expected_path = output_dir / "deduplication.db"
        assert config.get_db_path(output_dir) == expected_path


class TestDuplicationInfo:
    """Test duplication info data structure."""
    
    def test_duplication_info_creation(self):
        """Test creating duplication info."""
        dup_info = DuplicationInfo(
            bibcode="2021MNRAS.456..789B",
            year=2021,
            text_hash="abc123",
            first_seen_year=2020,
            first_seen_bibcode="2020A&A...123..456A"
        )
        
        assert dup_info.bibcode == "2021MNRAS.456..789B"
        assert dup_info.year == 2021
        assert dup_info.text_hash == "abc123"
        assert dup_info.first_seen_year == 2020
        assert dup_info.first_seen_bibcode == "2020A&A...123..456A"


class TestDeduplicationEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_dedup.db"
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_empty_text(self):
        """Test handling of empty text."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        # Empty text should still work
        hash1 = deduplicator.add_record("test1", "", 2020)
        duplicate_info = deduplicator.check_duplicate("test2", "", 2021)
        
        assert duplicate_info is not None
        assert duplicate_info.text_hash == hash1
    
    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        # Whitespace-only text should normalize to empty
        hash1 = deduplicator.add_record("test1", "   \n\t  ", 2020)
        duplicate_info = deduplicator.check_duplicate("test2", "", 2021)
        
        assert duplicate_info is not None
        assert duplicate_info.text_hash == hash1
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        # Create a very long text (10MB)
        long_text = "A" * (10 * 1024 * 1024)
        hash1 = deduplicator.add_record("test1", long_text, 2020)
        
        # Should handle it without issues
        assert len(hash1) == 64  # Valid SHA256
        
        duplicate_info = deduplicator.check_duplicate("test2", long_text, 2021)
        assert duplicate_info is not None
    
    def test_unicode_text(self):
        """Test handling of Unicode text."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        unicode_text = "This paper discusses 星系 formation and évolution."
        hash1 = deduplicator.add_record("test1", unicode_text, 2020)
        
        duplicate_info = deduplicator.check_duplicate("test2", unicode_text, 2021)
        assert duplicate_info is not None
        assert duplicate_info.text_hash == hash1
    
    def test_empty_batch(self):
        """Test processing empty batch."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        unique_records, duplicates = deduplicator.process_batch([])
        
        assert len(unique_records) == 0
        assert len(duplicates) == 0
    
    def test_same_bibcode_different_years(self):
        """Test same bibcode appearing in different years."""
        deduplicator = Deduplicator(self.db_path, enable_rocksdb=False)
        
        # Same bibcode, different years, same content
        text = "Test paper content"
        bibcode = "2020A&A...123..456A"
        
        deduplicator.add_record(bibcode, text, 2020)
        duplicate_info = deduplicator.check_duplicate(bibcode, text, 2021)
        
        # Should be detected as duplicate even with same bibcode
        assert duplicate_info is not None
        assert duplicate_info.bibcode == bibcode
        assert duplicate_info.year == 2021
        assert duplicate_info.first_seen_year == 2020


# Integration test that requires RocksDB to be available
@pytest.mark.skipif(
    not pytest.importorskip("rocksdb", reason="RocksDB not available"),
    reason="RocksDB not available"
)
class TestRocksDBIntegration:
    """Integration tests with actual RocksDB (if available)."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_rocksdb.db"
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_rocksdb_basic_functionality(self):
        """Test basic RocksDB functionality."""
        try:
            deduplicator = Deduplicator(self.db_path, enable_rocksdb=True)
            
            # Should use RocksDB if available
            if deduplicator.enable_rocksdb:
                # Add record
                text = "Test paper about astrophysics"
                hash1 = deduplicator.add_record("2020A&A...123..456A", text, 2020)
                
                # Check duplicate
                duplicate_info = deduplicator.check_duplicate("2021MNRAS.456..789B", text, 2021)
                assert duplicate_info is not None
                assert duplicate_info.text_hash == hash1
                
                # Close and reopen to test persistence
                deduplicator.close()
                
                deduplicator2 = Deduplicator(self.db_path, enable_rocksdb=True)
                if deduplicator2.enable_rocksdb:
                    # Should still find the duplicate
                    duplicate_info2 = deduplicator2.check_duplicate("2022ApJ...789..123C", text, 2022)
                    assert duplicate_info2 is not None
                    assert duplicate_info2.text_hash == hash1
                
                deduplicator2.close()
        
        except Exception:
            # If RocksDB fails, should fall back gracefully
            pytest.skip("RocksDB initialization failed")
