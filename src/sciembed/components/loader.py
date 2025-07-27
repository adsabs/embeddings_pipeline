"""Data loader for JSON/JSONL files with memory mapping and progress tracking."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional, List, Tuple
from pathlib import Path
import gzip
import io
import mmap
import orjson
import ijson
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Loader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, file_path: Path, fields: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """
        Load records from file, yielding only requested fields.
        
        Args:
            file_path: Path to the input file
            fields: List of fields to extract, None for all fields
            
        Yields:
            Dict containing record data with requested fields
        """
        pass


class JSONLoader(Loader):
    """Loader for JSON/JSONL files with memory mapping and progress tracking."""
    
    def __init__(self, show_progress: bool = True, chunk_size: int = 64 * 1024):
        """
        Initialize JSON loader.
        
        Args:
            show_progress: Whether to show progress bar
            chunk_size: Size of chunks for progress tracking (bytes)
        """
        self.show_progress = show_progress
        self.chunk_size = chunk_size
        self._record_count = 0
    
    @property
    def record_count(self) -> int:
        """Get the number of records processed in the last load operation."""
        return self._record_count
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect file format based on extension and magic bytes."""
        if file_path.suffix == '.gz':
            return 'gzip'
        elif file_path.suffix in ['.json', '.jsonl']:
            return 'plain'
        else:
            # Check magic bytes
            with open(file_path, 'rb') as f:
                magic = f.read(2)
                if magic == b'\x1f\x8b':
                    return 'gzip'
                else:
                    return 'plain'
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size for progress tracking."""
        return file_path.stat().st_size
    
    def _count_lines(self, file_path: Path) -> int:
        """Count total number of records in file for progress tracking."""
        format_type = self._detect_format(file_path)
        count = 0
        
        try:
            if format_type == 'gzip':
                file_obj = gzip.open(file_path, 'rb')
            else:
                file_obj = open(file_path, 'rb')
            
            with file_obj:
                # Use memory mapping for efficient line counting
                if format_type == 'plain':
                    with mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        for line in iter(mm.readline, b""):
                            line = line.strip()
                            if line and (line.startswith(b'{') or line.startswith(b'[')):
                                count += 1
                else:
                    # For gzipped files, count without memory mapping
                    for line in file_obj:
                        line = line.strip()
                        if line and (line.startswith(b'{') or line.startswith(b'[')):
                            count += 1
        except Exception as e:
            logger.warning(f"Could not count records in {file_path}: {e}")
            return 0
            
        return count
    
    def _load_jsonl_mmap(self, file_path: Path, fields: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Load JSONL using memory mapping for maximum performance."""
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                progress_bar = None
                if self.show_progress:
                    total_size = len(mm)
                    progress_bar = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Loading {file_path.name}",
                        postfix={'records': 0}
                    )
                
                current_pos = 0
                record_count = 0
                
                while current_pos < len(mm):
                    # Find end of line
                    line_end = mm.find(b'\n', current_pos)
                    if line_end == -1:
                        line_end = len(mm)
                    
                    line = mm[current_pos:line_end].strip()
                    if line:
                        try:
                            record = orjson.loads(line)
                            
                            # Filter fields if specified
                            if fields is not None:
                                filtered_record = {k: v for k, v in record.items() if k in fields}
                                if filtered_record:  # Only yield if we have requested fields
                                    yield filtered_record
                                    record_count += 1
                            else:
                                yield record
                                record_count += 1
                                
                        except orjson.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line at position {current_pos}: {e}")
                    
                    current_pos = line_end + 1
                    
                    # Update progress bar
                    if progress_bar and current_pos % self.chunk_size == 0:
                        progress_bar.update(self.chunk_size)
                        progress_bar.set_postfix({'records': record_count})
                
                if progress_bar:
                    progress_bar.update(len(mm) % self.chunk_size)
                    progress_bar.set_postfix({'records': record_count})
                    progress_bar.close()
                    
                self._record_count = record_count
    
    def load(self, file_path: Path, fields: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """
        Load records from JSON/JSONL file with optimized memory mapping.
        
        Args:
            file_path: Path to the input file
            fields: List of fields to extract, None for all fields
            
        Yields:
            Dict containing record data with requested fields
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        format_type = self._detect_format(file_path)
        logger.info(f"Loading {file_path.name} (format: {format_type})")
        
        # For plain JSON files, use optimized memory mapping
        if format_type == 'plain' and self._is_jsonl_format(file_path):
            yield from self._load_jsonl_mmap(file_path, fields)
            return
        
        # Fallback to streaming parser for gzipped files or complex JSON
        yield from self._load_with_streaming_parser(file_path, fields, format_type)
    
    def _is_jsonl_format(self, file_path: Path) -> bool:
        """Check if file is in JSONL format (one JSON object per line)."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024)  # Read first 1KB
                lines = content.decode('utf-8', errors='ignore').split('\n')
                
                valid_json_lines = 0
                total_lines = 0
                
                for line in lines[:5]:  # Check first 5 lines
                    line = line.strip()
                    if line:
                        total_lines += 1
                        try:
                            obj = orjson.loads(line)
                            # Must be a dict (object) for JSONL
                            if isinstance(obj, dict):
                                valid_json_lines += 1
                        except orjson.JSONDecodeError:
                            pass
                
                # Must have at least one line and all non-empty lines must be valid JSON objects
                return total_lines > 0 and valid_json_lines == total_lines
                
        except Exception:
            return False
    
    def _load_with_streaming_parser(self, file_path: Path, fields: Optional[List[str]], format_type: str) -> Iterator[Dict[str, Any]]:
        """Load using line-by-line parsing for JSONL or ijson for complex JSON."""
        file_size = self._get_file_size(file_path)
        
        # Open file with appropriate decompression
        if format_type == 'gzip':
            file_obj = gzip.open(file_path, 'rt', encoding='utf-8')
        else:
            file_obj = open(file_path, 'r', encoding='utf-8')
        
        try:
            progress_bar = None
            if self.show_progress:
                progress_bar = tqdm(
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    desc=f"Loading {file_path.name}",
                    postfix={'records': 0}
                )
            
            # Try line-by-line parsing first (works for JSONL)
            record_count = 0
            bytes_read = 0
            
            for line_num, line in enumerate(file_obj):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    record = orjson.loads(line)
                    
                    # Filter fields if specified
                    if fields is not None:
                        filtered_record = {k: v for k, v in record.items() if k in fields}
                        if filtered_record:  # Only yield if we have requested fields
                            yield filtered_record
                            record_count += 1
                    else:
                        yield record
                        record_count += 1
                        
                    # Update progress
                    if progress_bar:
                        bytes_read += len(line.encode('utf-8'))
                        if line_num % 100 == 0:  # Update every 100 lines
                            progress_bar.update(min(self.chunk_size, bytes_read))
                            progress_bar.set_postfix({'records': record_count})
                            bytes_read = 0
                            
                except orjson.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num + 1}: {e}")
                    continue
            
            if progress_bar:
                if bytes_read > 0:
                    progress_bar.update(bytes_read)
                progress_bar.set_postfix({'records': record_count})
                progress_bar.close()
                
            self._record_count = record_count
            logger.info(f"Loaded {record_count} records from {file_path.name}")
                
        except UnicodeDecodeError:
            # Fallback to ijson for binary data or complex JSON
            file_obj.close()
            yield from self._load_with_ijson_parser(file_path, fields, format_type)
        finally:
            if not file_obj.closed:
                file_obj.close()
    
    def _load_with_ijson_parser(self, file_path: Path, fields: Optional[List[str]], format_type: str) -> Iterator[Dict[str, Any]]:
        """Fallback to ijson streaming parser for complex JSON structures."""
        # Open file with appropriate decompression
        if format_type == 'gzip':
            file_obj = gzip.open(file_path, 'rb')
        else:
            file_obj = open(file_path, 'rb')
        
        try:
            # Use ijson for streaming JSON parsing
            parser = ijson.parse(file_obj)
            current_record = {}
            in_array = False
            record_count = 0
            
            for prefix, event, value in parser:
                # Handle array of objects
                if prefix == '' and event == 'start_array':
                    in_array = True
                elif prefix == 'item' and event == 'start_map':
                    current_record = {}
                elif prefix.startswith('item.') and event in ['string', 'number', 'boolean', 'null']:
                    field_name = prefix.split('.', 1)[1]
                    if fields is None or field_name in fields:
                        current_record[field_name] = value
                elif prefix == 'item' and event == 'end_map':
                    if current_record:
                        yield current_record
                        record_count += 1
                        current_record = {}
                
                # Handle single object
                elif not in_array and event in ['string', 'number', 'boolean', 'null']:
                    field_name = prefix
                    if fields is None or field_name in fields:
                        current_record[field_name] = value
                elif not in_array and prefix == '' and event == 'end_map':
                    if current_record:
                        yield current_record
                        record_count += 1
                        current_record = {}
            
            self._record_count = record_count
            logger.info(f"Loaded {record_count} records from {file_path.name} using ijson")
                
        finally:
            file_obj.close()


class DirectoryLoader:
    """Loader for directories containing multiple JSON files with progress tracking."""
    
    def __init__(self, loader: Loader):
        """
        Initialize directory loader.
        
        Args:
            loader: Base loader instance to use for individual files
        """
        self.loader = loader
        self._total_records = 0
    
    def load_years(self, input_dir: Path, years: List[int], fields: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """
        Load records from multiple year files with progress tracking.
        
        Args:
            input_dir: Directory containing year files
            years: List of years to process
            fields: List of fields to extract
            
        Yields:
            Dict containing record data with requested fields
        """
        self._total_records = 0
        
        for year in years:
            year_files = [
                input_dir / f"{year}.json",
                input_dir / f"{year}.json.gz", 
                input_dir / f"{year}.jsonl",
                input_dir / f"{year}.jsonl.gz"
            ]
            
            # Find the first existing file for this year
            year_file = None
            for candidate in year_files:
                if candidate.exists():
                    year_file = candidate
                    break
            
            if year_file is None:
                logger.warning(f"No data file found for year {year} in {input_dir}")
                continue
            
            logger.info(f"Processing year {year} from {year_file.name}")
            
            year_start_count = self._total_records
            yield from self.loader.load(year_file, fields)
            
            # Update total count after processing year
            if hasattr(self.loader, 'record_count'):
                year_records = self.loader.record_count
                self._total_records += year_records
                logger.info(f"Year {year}: processed {year_records} records")
    
    @property 
    def total_records(self) -> int:
        """Get the total number of records processed across all files."""
        return self._total_records
