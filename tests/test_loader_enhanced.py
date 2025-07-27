"""Tests for the enhanced JSON loader with memory mapping and progress tracking."""

import json
import gzip
import tempfile
from pathlib import Path
import pytest

from src.sciembed.components.loader import JSONLoader, DirectoryLoader


class TestEnhancedJSONLoader:
    """Test enhanced JSON loader functionality."""

    def test_memory_mapped_jsonl_loading(self):
        """Test memory-mapped loading of JSONL files."""
        # Create test JSONL data
        test_records = [
            {"bibcode": "2023test001", "title": "Test Paper 1", "abstract": "This is abstract 1"},
            {"bibcode": "2023test002", "title": "Test Paper 2", "abstract": "This is abstract 2"},
            {"bibcode": "2023test003", "title": "Test Paper 3", "abstract": "This is abstract 3"},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')
            temp_path = Path(f.name)
        
        try:
            loader = JSONLoader(show_progress=False)
            
            # Test loading all fields
            records = list(loader.load(temp_path))
            assert len(records) == 3
            assert loader.record_count == 3
            assert records[0]["bibcode"] == "2023test001"
            assert records[1]["title"] == "Test Paper 2"
            
            # Test field filtering
            filtered_records = list(loader.load(temp_path, fields=["bibcode", "title"]))
            assert len(filtered_records) == 3
            assert "abstract" not in filtered_records[0]
            assert "bibcode" in filtered_records[0]
            assert "title" in filtered_records[0]
            
        finally:
            temp_path.unlink()
    
    def test_gzipped_file_loading(self):
        """Test loading of gzipped JSON files."""
        test_records = [
            {"bibcode": "2023gz001", "title": "Compressed Paper 1"},
            {"bibcode": "2023gz002", "title": "Compressed Paper 2"},
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as f:
            with gzip.open(f.name, 'wt') as gz_file:
                for record in test_records:
                    gz_file.write(json.dumps(record) + '\n')
            temp_path = Path(f.name)
        
        try:
            loader = JSONLoader(show_progress=False)
            records = list(loader.load(temp_path))
            
            assert len(records) == 2
            assert loader.record_count == 2
            assert records[0]["bibcode"] == "2023gz001"
            
        finally:
            temp_path.unlink()
    
    def test_format_detection(self):
        """Test automatic format detection."""
        loader = JSONLoader(show_progress=False)
        
        # Test extension-based detection
        assert loader._detect_format(Path("test.json")) == "plain"
        assert loader._detect_format(Path("test.jsonl")) == "plain"
        assert loader._detect_format(Path("test.json.gz")) == "gzip"
        
        # Test magic byte detection for gzipped file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            with gzip.open(f.name, 'wb') as gz:
                gz.write(b'{"test": "data"}')
            
            temp_path = Path(f.name)
            try:
                assert loader._detect_format(temp_path) == "gzip"
            finally:
                temp_path.unlink()
    
    def test_jsonl_format_detection(self):
        """Test JSONL format detection."""
        loader = JSONLoader(show_progress=False)
        
        # Valid JSONL
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"line": 1}\n{"line": 2}\n{"line": 3}\n')
            temp_path = Path(f.name)
        
        try:
            assert loader._is_jsonl_format(temp_path) is True
        finally:
            temp_path.unlink()
        
        # Invalid JSONL (array format)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('[{"line": 1}, {"line": 2}]')
            temp_path = Path(f.name)
        
        try:
            assert loader._is_jsonl_format(temp_path) is False
        finally:
            temp_path.unlink()
    
    def test_error_handling(self):
        """Test error handling for invalid files."""
        loader = JSONLoader(show_progress=False)
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            list(loader.load(Path("nonexistent.json")))
        
        # Test invalid JSON with memory mapping
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
            temp_path = Path(f.name)
        
        try:
            # Should skip invalid lines and continue
            records = list(loader.load(temp_path))
            assert len(records) == 2
            assert records[0]["valid"] == "json"
            assert records[1]["another"] == "valid"
        finally:
            temp_path.unlink()
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        # Create larger test file
        test_records = [{"id": i, "data": f"record_{i}"} for i in range(100)]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')
            temp_path = Path(f.name)
        
        try:
            # Test with progress enabled (won't show in tests but should not error)
            loader = JSONLoader(show_progress=True, chunk_size=1024)
            records = list(loader.load(temp_path))
            
            assert len(records) == 100
            assert loader.record_count == 100
            
        finally:
            temp_path.unlink()


class TestDirectoryLoader:
    """Test directory loader with multiple files."""
    
    def test_load_multiple_years(self):
        """Test loading data from multiple year files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files for different years
            for year in [2020, 2021, 2022]:
                year_records = [
                    {"bibcode": f"{year}test{i:03d}", "year": year, "title": f"Paper {i}"}
                    for i in range(1, 4)
                ]
                
                year_file = temp_path / f"{year}.jsonl"
                with open(year_file, 'w') as f:
                    for record in year_records:
                        f.write(json.dumps(record) + '\n')
            
            # Test loading all years
            base_loader = JSONLoader(show_progress=False)
            dir_loader = DirectoryLoader(base_loader)
            
            all_records = list(dir_loader.load_years(temp_path, [2020, 2021, 2022]))
            
            assert len(all_records) == 9  # 3 records per year × 3 years
            assert dir_loader.total_records == 9
            
            # Check data integrity
            years_found = {record["year"] for record in all_records}
            assert years_found == {2020, 2021, 2022}
    
    def test_missing_year_handling(self):
        """Test handling of missing year files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create only 2021 file
            year_records = [{"bibcode": "2021test001", "year": 2021}]
            year_file = temp_path / "2021.jsonl"
            with open(year_file, 'w') as f:
                for record in year_records:
                    f.write(json.dumps(record) + '\n')
            
            base_loader = JSONLoader(show_progress=False)
            dir_loader = DirectoryLoader(base_loader)
            
            # Request 2020, 2021, 2022 but only 2021 exists
            records = list(dir_loader.load_years(temp_path, [2020, 2021, 2022]))
            
            assert len(records) == 1
            assert records[0]["year"] == 2021
            assert dir_loader.total_records == 1
    
    def test_different_file_formats(self):
        """Test loading different file formats in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create .json file for 2020
            with open(temp_path / "2020.json", 'w') as f:
                f.write('{"bibcode": "2020test001", "format": "json"}\n')
            
            # Create .jsonl.gz file for 2021  
            with gzip.open(temp_path / "2021.jsonl.gz", 'wt') as f:
                f.write('{"bibcode": "2021test001", "format": "jsonl.gz"}\n')
            
            base_loader = JSONLoader(show_progress=False)
            dir_loader = DirectoryLoader(base_loader)
            
            records = list(dir_loader.load_years(temp_path, [2020, 2021]))
            
            assert len(records) == 2
            formats = {record["format"] for record in records}
            assert formats == {"json", "jsonl.gz"}


if __name__ == "__main__":
    pytest.main([__file__])
