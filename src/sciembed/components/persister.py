"""Vector and metadata persistence using Arrow IPC and Faiss indexing."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import json
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class Manifest:
    """Metadata manifest for a year's embeddings."""
    year: int
    model: str
    dim: int
    count: int
    fields_hash: str
    prompt_hash: str
    created_at: str
    vector_file: str
    bibcode_file: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        """Create manifest from dictionary."""
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class Persister:
    """Persists embeddings and metadata in efficient columnar format."""
    
    def __init__(self, output_dir: Path, use_float16: bool = True):
        """
        Initialize persister.
        
        Args:
            output_dir: Directory to store output files
            use_float16: Whether to use float16 for space efficiency
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_float16 = use_float16
        self.dtype = np.float16 if use_float16 else np.float32
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        bibcodes: List[str],
        year: int,
        model: str,
        fields_hash: str,
        prompt_hash: str
    ) -> Manifest:
        """
        Save embeddings and create manifest.
        
        Args:
            embeddings: Array of shape (n, dim) containing embeddings
            bibcodes: List of bibcodes aligned with embeddings
            year: Year of the data
            model: Model name used for embeddings
            fields_hash: Hash of field configuration
            prompt_hash: Hash of prompt configuration
            
        Returns:
            Manifest object describing saved data
        """
        if len(embeddings) != len(bibcodes):
            raise ValueError("Embeddings and bibcodes must have same length")
        
        # Convert to specified dtype
        if embeddings.dtype != self.dtype:
            embeddings = embeddings.astype(self.dtype)
        
        # Generate file names
        model_safe = model.replace("/", "_").replace(":", "_")
        vector_file = f"embeddings_{model_safe}_{year}.arrow"
        bibcode_file = f"bibcodes_{year}.txt"
        manifest_file = f"manifest_{year}.json"
        
        # Save embeddings using Arrow IPC format
        vector_path = self.output_dir / vector_file
        self._save_vectors_arrow(embeddings, vector_path)
        
        # Save bibcodes as plain text
        bibcode_path = self.output_dir / bibcode_file
        self._save_bibcodes(bibcodes, bibcode_path)
        
        # Create and save manifest
        manifest = Manifest(
            year=year,
            model=model,
            dim=embeddings.shape[1],
            count=len(embeddings),
            fields_hash=fields_hash,
            prompt_hash=prompt_hash,
            created_at=datetime.now().isoformat(),
            vector_file=vector_file,
            bibcode_file=bibcode_file
        )
        
        manifest_path = self.output_dir / manifest_file
        manifest.save(manifest_path)
        
        return manifest
    
    def _save_vectors_arrow(self, embeddings: np.ndarray, path: Path) -> None:
        """Save embeddings using Arrow IPC format."""
        # Create Arrow table with embeddings as a list column
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        schema = pa.schema([
            pa.field("embedding", pa.list_(pa.float16() if self.use_float16 else pa.float32()))
        ])
        
        table = pa.table({"embedding": embeddings_list}, schema=schema)
        
        # Write as Arrow IPC file
        with pa.OSFile(str(path), 'wb') as sink:
            with ipc.new_file(sink, schema) as writer:
                writer.write_table(table)
    
    def _save_bibcodes(self, bibcodes: List[str], path: Path) -> None:
        """Save bibcodes as newline-delimited text file."""
        with open(path, 'w') as f:
            for bibcode in bibcodes:
                f.write(f"{bibcode}\n")
    
    def load_embeddings(self, manifest: Manifest) -> tuple[np.ndarray, List[str]]:
        """
        Load embeddings and bibcodes from manifest.
        
        Args:
            manifest: Manifest describing the data to load
            
        Returns:
            Tuple of (embeddings, bibcodes)
        """
        # Load embeddings
        vector_path = self.output_dir / manifest.vector_file
        embeddings = self._load_vectors_arrow(vector_path)
        
        # Load bibcodes
        bibcode_path = self.output_dir / manifest.bibcode_file
        bibcodes = self._load_bibcodes(bibcode_path)
        
        if len(embeddings) != len(bibcodes):
            raise ValueError("Embeddings and bibcodes have mismatched lengths")
        
        return embeddings, bibcodes
    
    def _load_vectors_arrow(self, path: Path) -> np.ndarray:
        """Load embeddings from Arrow IPC file."""
        with pa.memory_map(str(path), 'r') as source:
            with ipc.open_file(source) as reader:
                table = reader.read_all()
        
        # Convert list column back to numpy array
        embeddings_list = table["embedding"].to_pylist()
        return np.array(embeddings_list, dtype=self.dtype)
    
    def _load_bibcodes(self, path: Path) -> List[str]:
        """Load bibcodes from text file."""
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def append_embeddings(
        self,
        embeddings: np.ndarray,
        bibcodes: List[str],
        manifest: Manifest
    ) -> Manifest:
        """
        Append new embeddings to existing files.
        
        Args:
            embeddings: New embeddings to append
            bibcodes: New bibcodes to append
            manifest: Existing manifest to update
            
        Returns:
            Updated manifest
        """
        # Load existing data
        existing_embeddings, existing_bibcodes = self.load_embeddings(manifest)
        
        # Combine with new data
        combined_embeddings = np.vstack([existing_embeddings, embeddings.astype(self.dtype)])
        combined_bibcodes = existing_bibcodes + bibcodes
        
        # Save combined data
        updated_manifest = self.save_embeddings(
            combined_embeddings,
            combined_bibcodes,
            manifest.year,
            manifest.model,
            manifest.fields_hash,
            manifest.prompt_hash
        )
        
        return updated_manifest
