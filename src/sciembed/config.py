"""Configuration schema and validation for the embedding pipeline."""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import json

from .components.preparer import PreparerConfig
from .components.embedder import EmbedderConfig
from .components.deduplicator import DeduplicationConfig


@dataclass
class Config:
    """Main configuration for the embedding pipeline."""
    
    # Input/Output paths
    input_dir: Union[str, Path]
    output_dir: Union[str, Path]
    
    # Data selection
    years: List[int]
    fields: List[str] = field(default_factory=lambda: ["title", "abstract"])
    
    # Text preparation
    prefix: str = ""
    suffix: str = ""
    delimiter: str = "\n\n"
    lowercase: bool = True
    truncate: int = 3000
    
    # Model configuration
    model: str = "hf://sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_tokens: int = 32000
    api_key: Optional[str] = None
    device: str = "auto"
    
    # Performance settings
    num_workers: int = 4
    use_float16: bool = True
    show_progress: bool = True
    use_async: bool = False
    
    # Index settings
    create_faiss_index: bool = True
    faiss_index_type: str = "flat"
    
    # Resumability
    resume: bool = True
    deduplicate: bool = True
    
    # Deduplication settings
    dedup_use_rocksdb: bool = True
    dedup_cross_year: bool = True
    dedup_log_duplicates: bool = True
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
    
    @property
    def preparer_config(self) -> PreparerConfig:
        """Get preparer configuration."""
        return PreparerConfig(
            fields=self.fields,
            prefix=self.prefix,
            suffix=self.suffix,
            delimiter=self.delimiter,
            lowercase=self.lowercase,
            truncate=self.truncate
        )
    
    @property
    def embedder_config(self) -> EmbedderConfig:
        """Get embedder configuration."""
        return EmbedderConfig(
            model=self.model,
            batch_size=self.batch_size,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            device=self.device
        )
    
    @property
    def deduplication_config(self) -> DeduplicationConfig:
        """Get deduplication configuration."""
        return DeduplicationConfig(
            enabled=self.deduplicate,
            use_rocksdb=self.dedup_use_rocksdb,
            db_path=None,  # Will use output_dir/deduplication.db
            cross_year=self.dedup_cross_year,
            log_duplicates=self.dedup_log_duplicates
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []
        
        # Check required paths
        if not self.input_dir.exists():
            errors.append(f"Input directory does not exist: {self.input_dir}")
        
        # Check years
        if not self.years:
            errors.append("At least one year must be specified")
        
        # Check fields
        if not self.fields:
            errors.append("At least one field must be specified")
        
        # Check model format
        if not self.model:
            errors.append("Model must be specified")
        
        # Check batch size
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Check truncate length
        if self.truncate <= 0:
            errors.append("Truncate length must be positive")
        
        # Check num_workers
        if self.num_workers <= 0:
            errors.append("Number of workers must be positive")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))


def create_default_config() -> Config:
    """Create a default configuration."""
    return Config(
        input_dir="./data",
        output_dir="./embeddings",
        years=[2020],
        fields=["title", "abstract"],
        model="hf://sentence-transformers/all-MiniLM-L6-v2"
    )


def load_config(path: Optional[Union[str, Path]] = None, **kwargs) -> Config:
    """
    Load configuration with optional overrides.
    
    Args:
        path: Optional path to config file (YAML or JSON)
        **kwargs: Override parameters
        
    Returns:
        Configuration object
    """
    if path is None:
        config = create_default_config()
    else:
        path = Path(path)
        if path.suffix in ['.yaml', '.yml']:
            config = Config.from_yaml(path)
        elif path.suffix == '.json':
            config = Config.from_json(path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    # Validate final configuration
    config.validate()
    
    return config
