"""Deduplication system using RocksDB and SHA256 hashing."""

import hashlib
import logging
from typing import Set, Dict, List, Optional, Tuple
from pathlib import Path
import pickle
from dataclasses import dataclass

try:
    import rocksdb
    ROCKSDB_AVAILABLE = True
except ImportError:
    ROCKSDB_AVAILABLE = False


@dataclass
class DuplicationInfo:
    """Information about a duplicated record."""
    bibcode: str
    year: int
    text_hash: str
    first_seen_year: int
    first_seen_bibcode: str


class Deduplicator:
    """Deduplication system using SHA256 hashing and RocksDB storage."""
    
    def __init__(self, db_path: Path, enable_rocksdb: bool = True):
        """
        Initialize deduplicator.
        
        Args:
            db_path: Path to RocksDB database
            enable_rocksdb: Whether to use RocksDB (fallback to in-memory if False)
        """
        self.db_path = db_path
        self.enable_rocksdb = enable_rocksdb and ROCKSDB_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        if self.enable_rocksdb:
            self._init_rocksdb()
        else:
            self._init_memory_storage()
            if enable_rocksdb:
                self.logger.warning("RocksDB not available, using in-memory deduplication")
    
    def _init_rocksdb(self) -> None:
        """Initialize RocksDB storage."""
        try:
            # Create database directory
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # RocksDB options for performance
            opts = rocksdb.Options()
            opts.create_if_missing = True
            opts.max_open_files = 300000
            opts.write_buffer_size = 67108864  # 64MB
            opts.max_write_buffer_number = 3
            opts.target_file_size_base = 67108864  # 64MB
            
            # Compression
            opts.compression = rocksdb.CompressionType.snappy_compression
            
            self.db = rocksdb.DB(str(self.db_path), opts)
            self.logger.info(f"Initialized RocksDB deduplication at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RocksDB: {e}")
            self.logger.warning("Falling back to in-memory deduplication")
            self.enable_rocksdb = False
            self._init_memory_storage()
    
    def _init_memory_storage(self) -> None:
        """Initialize in-memory storage."""
        self.memory_hashes: Dict[str, DuplicationInfo] = {}
        self.logger.info("Initialized in-memory deduplication")
    
    def compute_text_hash(self, text: str) -> str:
        """
        Compute SHA256 hash of text content.
        
        Args:
            text: Text to hash
            
        Returns:
            Hexadecimal SHA256 hash
        """
        # Normalize text for consistent hashing
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def check_duplicate(self, bibcode: str, text: str, year: int) -> Optional[DuplicationInfo]:
        """
        Check if text is a duplicate of previously seen content.
        
        Args:
            bibcode: Bibcode of the record
            text: Text content to check
            year: Year of the record
            
        Returns:
            DuplicationInfo if duplicate found, None otherwise
        """
        text_hash = self.compute_text_hash(text)
        
        if self.enable_rocksdb:
            return self._check_duplicate_rocksdb(bibcode, text_hash, year)
        else:
            return self._check_duplicate_memory(bibcode, text_hash, year)
    
    def _check_duplicate_rocksdb(
        self, 
        bibcode: str, 
        text_hash: str, 
        year: int
    ) -> Optional[DuplicationInfo]:
        """Check duplicate using RocksDB."""
        try:
            # Look up hash in database
            key = text_hash.encode('utf-8')
            value_bytes = self.db.get(key)
            
            if value_bytes is not None:
                # Deserialize stored info
                stored_info = pickle.loads(value_bytes)
                
                # Return duplication info
                return DuplicationInfo(
                    bibcode=bibcode,
                    year=year,
                    text_hash=text_hash,
                    first_seen_year=stored_info['year'],
                    first_seen_bibcode=stored_info['bibcode']
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking duplicate in RocksDB: {e}")
            return None
    
    def _check_duplicate_memory(
        self, 
        bibcode: str, 
        text_hash: str, 
        year: int
    ) -> Optional[DuplicationInfo]:
        """Check duplicate using in-memory storage."""
        if text_hash in self.memory_hashes:
            stored_info = self.memory_hashes[text_hash]
            return DuplicationInfo(
                bibcode=bibcode,
                year=year,
                text_hash=text_hash,
                first_seen_year=stored_info.first_seen_year,
                first_seen_bibcode=stored_info.first_seen_bibcode
            )
        
        return None
    
    def add_record(self, bibcode: str, text: str, year: int) -> str:
        """
        Add a record to the deduplication database.
        
        Args:
            bibcode: Bibcode of the record
            text: Text content
            year: Year of the record
            
        Returns:
            SHA256 hash of the text
        """
        text_hash = self.compute_text_hash(text)
        
        if self.enable_rocksdb:
            self._add_record_rocksdb(bibcode, text_hash, year)
        else:
            self._add_record_memory(bibcode, text_hash, year)
        
        return text_hash
    
    def _add_record_rocksdb(self, bibcode: str, text_hash: str, year: int) -> None:
        """Add record using RocksDB."""
        try:
            key = text_hash.encode('utf-8')
            value = pickle.dumps({
                'bibcode': bibcode,
                'year': year
            })
            self.db.put(key, value)
            
        except Exception as e:
            self.logger.error(f"Error adding record to RocksDB: {e}")
    
    def _add_record_memory(self, bibcode: str, text_hash: str, year: int) -> None:
        """Add record using in-memory storage."""
        self.memory_hashes[text_hash] = DuplicationInfo(
            bibcode=bibcode,
            year=year,
            text_hash=text_hash,
            first_seen_year=year,
            first_seen_bibcode=bibcode
        )
    
    def process_batch(
        self, 
        records: List[Tuple[str, str, int]]
    ) -> Tuple[List[Tuple[str, str, int]], List[DuplicationInfo]]:
        """
        Process a batch of records for deduplication.
        
        Args:
            records: List of (bibcode, text, year) tuples
            
        Returns:
            Tuple of (unique_records, duplicates)
        """
        unique_records = []
        duplicates = []
        
        for bibcode, text, year in records:
            duplicate_info = self.check_duplicate(bibcode, text, year)
            
            if duplicate_info is not None:
                duplicates.append(duplicate_info)
            else:
                # Add to database and keep for processing
                self.add_record(bibcode, text, year)
                unique_records.append((bibcode, text, year))
        
        return unique_records, duplicates
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get deduplication statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.enable_rocksdb:
            try:
                # RocksDB doesn't have a direct count method, so we estimate
                # based on iterator sampling
                count = 0
                iterator = self.db.iterkeys()
                iterator.seek_to_first()
                
                # Sample every 1000th key to estimate total
                sample_count = 0
                for i, key in enumerate(iterator):
                    if i % 1000 == 0:
                        sample_count += 1
                    if sample_count >= 100:  # Limit sampling
                        break
                
                estimated_total = sample_count * 1000
                return {
                    'storage_type': 'rocksdb',
                    'estimated_unique_hashes': estimated_total,
                    'database_path': str(self.db_path)
                }
                
            except Exception as e:
                self.logger.error(f"Error getting RocksDB stats: {e}")
                return {
                    'storage_type': 'rocksdb',
                    'estimated_unique_hashes': -1,
                    'error': str(e)
                }
        else:
            return {
                'storage_type': 'memory',
                'unique_hashes': len(self.memory_hashes)
            }
    
    def close(self) -> None:
        """Close the deduplication database."""
        if self.enable_rocksdb and hasattr(self, 'db'):
            try:
                del self.db
                self.logger.info("Closed RocksDB deduplication database")
            except Exception as e:
                self.logger.error(f"Error closing RocksDB: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class DeduplicationConfig:
    """Configuration for deduplication system."""
    
    def __init__(
        self,
        enabled: bool = True,
        use_rocksdb: bool = True,
        db_path: Optional[Path] = None,
        cross_year: bool = True,
        log_duplicates: bool = True
    ):
        """
        Initialize deduplication configuration.
        
        Args:
            enabled: Whether deduplication is enabled
            use_rocksdb: Whether to use RocksDB (fallback to memory)
            db_path: Path to RocksDB database
            cross_year: Whether to deduplicate across years
            log_duplicates: Whether to log duplicate records
        """
        self.enabled = enabled
        self.use_rocksdb = use_rocksdb
        self.db_path = db_path
        self.cross_year = cross_year
        self.log_duplicates = log_duplicates
    
    def get_db_path(self, output_dir: Path) -> Path:
        """Get the database path, with fallback to output directory."""
        if self.db_path is not None:
            return self.db_path
        return output_dir / "deduplication.db"
