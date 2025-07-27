"""Performance tests for the enhanced JSON loader."""

import json
import tempfile
import time
from pathlib import Path
import pytest

from src.sciembed.components.loader import JSONLoader, DirectoryLoader


@pytest.mark.slow
class TestLoaderPerformance:
    """Test loader performance with larger datasets."""

    def test_large_jsonl_file_performance(self):
        """Test performance with a large JSONL file (10k records)."""
        num_records = 10_000
        
        # Create test data
        test_records = [
            {
                "bibcode": f"2023test{i:06d}",
                "title": f"Test Paper {i}: A Very Long Title That Simulates Real Papers",
                "abstract": f"This is a detailed abstract for paper {i}. " * 20,  # ~500 chars
                "authors": [f"Author {j}" for j in range(i % 5 + 1)],
                "year": 2023,
                "citations": i % 100
            }
            for i in range(num_records)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')
            temp_path = Path(f.name)
        
        try:
            loader = JSONLoader(show_progress=False)
            
            # Measure loading time
            start_time = time.time()
            records = list(loader.load(temp_path))
            load_time = time.time() - start_time
            
            # Verify results
            assert len(records) == num_records
            assert loader.record_count == num_records
            
            # Performance assertions
            records_per_second = num_records / load_time
            print(f"Processed {records_per_second:.0f} records/second")
            
            # Should process at least 1000 records/second for small records
            assert records_per_second > 1000, f"Too slow: {records_per_second:.0f} records/second"
            
        finally:
            temp_path.unlink()
    
    def test_memory_mapped_vs_streaming_performance(self):
        """Compare performance between memory mapped and streaming approaches."""
        num_records = 5_000
        
        test_records = [
            {
                "bibcode": f"2023perf{i:06d}",
                "title": f"Performance Test Paper {i}",
                "abstract": "Short abstract for performance testing." * 10
            }
            for i in range(num_records)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')
            temp_path = Path(f.name)
        
        try:
            loader = JSONLoader(show_progress=False)
            
            # Test memory mapped approach (should be automatically selected)
            start_time = time.time()
            mmap_records = list(loader.load(temp_path))
            mmap_time = time.time() - start_time
            
            # Test streaming approach
            start_time = time.time()  
            streaming_records = list(loader._load_with_streaming_parser(temp_path, None, 'plain'))
            streaming_time = time.time() - start_time
            
            # Verify both approaches return same data
            assert len(mmap_records) == len(streaming_records) == num_records
            
            print(f"Memory mapped: {mmap_time:.3f}s, Streaming: {streaming_time:.3f}s")
            
            # Memory mapped should generally be faster (though not guaranteed for small files)
            assert mmap_time < streaming_time * 2, "Memory mapped approach is significantly slower"
            
        finally:
            temp_path.unlink()
    
    def test_field_filtering_performance(self):
        """Test performance impact of field filtering."""
        num_records = 5_000
        
        # Create records with many fields
        test_records = [
            {
                "bibcode": f"2023filter{i:06d}",
                "title": f"Filter Test Paper {i}",
                "abstract": "Abstract text " * 20,
                "authors": [f"Author {j}" for j in range(10)],
                "keywords": [f"keyword{j}" for j in range(15)],
                "citations": list(range(i % 50)),
                "metadata": {"field1": "value1", "field2": "value2", "field3": "value3"},
                "year": 2023,
                "journal": "Test Journal",
                "doi": f"10.1000/test.{i}"
            }
            for i in range(num_records)
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in test_records:
                f.write(json.dumps(record) + '\n')
            temp_path = Path(f.name)
        
        try:
            loader = JSONLoader(show_progress=False)
            
            # Test loading all fields
            start_time = time.time()
            all_records = list(loader.load(temp_path))
            all_fields_time = time.time() - start_time
            
            # Test loading only specific fields
            start_time = time.time()
            filtered_records = list(loader.load(temp_path, fields=["bibcode", "title"]))
            filtered_time = time.time() - start_time
            
            # Verify filtering worked
            assert len(all_records) == len(filtered_records) == num_records
            assert "abstract" in all_records[0]
            assert "abstract" not in filtered_records[0]
            assert "bibcode" in filtered_records[0]
            assert "title" in filtered_records[0]
            
            print(f"All fields: {all_fields_time:.3f}s, Filtered: {filtered_time:.3f}s")
            
            # Filtering should be at least as fast (may have some overhead)
            assert filtered_time < all_fields_time * 1.5, "Field filtering is too slow"
            
        finally:
            temp_path.unlink()
    
    def test_directory_loading_performance(self):
        """Test performance of loading multiple year files."""
        years = [2020, 2021, 2022]
        records_per_year = 2_000
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files for each year
            for year in years:
                year_records = [
                    {
                        "bibcode": f"{year}perf{i:06d}",
                        "title": f"Directory Test Paper {i}",
                        "year": year,
                        "abstract": "Directory loading test abstract." * 15
                    }
                    for i in range(records_per_year)
                ]
                
                year_file = temp_path / f"{year}.jsonl"
                with open(year_file, 'w') as f:
                    for record in year_records:
                        f.write(json.dumps(record) + '\n')
            
            # Test directory loading performance
            base_loader = JSONLoader(show_progress=False)
            dir_loader = DirectoryLoader(base_loader)
            
            start_time = time.time()
            all_records = list(dir_loader.load_years(temp_path, years))
            load_time = time.time() - start_time
            
            # Verify results
            total_expected = len(years) * records_per_year
            assert len(all_records) == total_expected
            assert dir_loader.total_records == total_expected
            
            # Performance check
            records_per_second = total_expected / load_time
            print(f"Directory loading: {records_per_second:.0f} records/second")
            
            # Should handle at least 500 records/second across multiple files
            assert records_per_second > 500, f"Directory loading too slow: {records_per_second:.0f} records/second"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
