"""Text preparation component for configurable field selection and prompting."""

from typing import Dict, Any, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import hashlib


@dataclass
class PreparerConfig:
    """Configuration for text preparation."""
    fields: List[str]
    prefix: str = ""
    suffix: str = ""
    delimiter: str = "\n\n"
    lowercase: bool = True
    truncate: int = 3000
    
    def hash(self) -> str:
        """Generate hash of configuration for reproducibility tracking."""
        config_str = f"{self.fields}{self.prefix}{self.suffix}{self.delimiter}{self.lowercase}{self.truncate}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class Preparer:
    """Prepares text snippets from record fields according to configuration."""
    
    def __init__(self, config: PreparerConfig):
        """
        Initialize preparer with configuration.
        
        Args:
            config: Configuration specifying how to prepare text
        """
        self.config = config
    
    def prepare_record(self, record: Dict[str, Any]) -> Tuple[str, str]:
        """
        Prepare text from a single record.
        
        Args:
            record: Dictionary containing record fields
            
        Returns:
            Tuple of (bibcode, prepared_text)
            
        Raises:
            KeyError: If bibcode field is missing
            ValueError: If no specified fields are found in record
        """
        if "bibcode" not in record:
            raise KeyError("Record missing required 'bibcode' field")
        
        bibcode = record["bibcode"]
        
        # Extract field values, skipping missing fields
        field_values = []
        
        # Add prefix if specified
        if self.config.prefix:
            field_values.append(self.config.prefix)
        
        # Extract requested fields
        for field in self.config.fields:
            if field in record and record[field]:
                value = str(record[field])
                if self.config.lowercase:
                    value = value.lower()
                field_values.append(value)
        
        # Add suffix if specified
        if self.config.suffix:
            field_values.append(self.config.suffix)
        
        # Check if we only have prefix/suffix without any actual field content
        content_fields = [v for v in field_values if v not in [self.config.prefix, self.config.suffix]]
        if not content_fields:
            raise ValueError(f"No content found in specified fields {self.config.fields} for record {bibcode}")
        
        # Join with delimiter and truncate
        text = self.config.delimiter.join(field_values)
        if self.config.truncate > 0:
            text = text[:self.config.truncate]
        
        return bibcode, text
    
    def prepare_batch(self, records: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Prepare text from a batch of records.
        
        Args:
            records: List of record dictionaries
            
        Returns:
            List of (bibcode, prepared_text) tuples
        """
        results = []
        for record in records:
            try:
                bibcode, text = self.prepare_record(record)
                results.append((bibcode, text))
            except (KeyError, ValueError) as e:
                # Log warning but continue processing
                print(f"Warning: Skipping record due to error: {e}")
                continue
        
        return results
    
    def prepare_stream(self, record_stream: Iterator[Dict[str, Any]]) -> Iterator[Tuple[str, str]]:
        """
        Prepare text from a stream of records.
        
        Args:
            record_stream: Iterator yielding record dictionaries
            
        Yields:
            Tuples of (bibcode, prepared_text)
        """
        for record in record_stream:
            try:
                bibcode, text = self.prepare_record(record)
                yield bibcode, text
            except (KeyError, ValueError) as e:
                # Log warning but continue processing
                print(f"Warning: Skipping record due to error: {e}")
                continue
    
    def get_text_hash(self, text: str) -> str:
        """
        Generate hash of prepared text for deduplication.
        
        Args:
            text: Prepared text string
            
        Returns:
            SHA256 hash of the text
        """
        return hashlib.sha256(text.encode()).hexdigest()
