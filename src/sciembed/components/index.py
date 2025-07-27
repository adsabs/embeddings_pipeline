"""Lightweight indexing for fast bibcode lookup and vector search."""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import sqlite3
import json
from dataclasses import dataclass
from .persister import Manifest


@dataclass
class IndexEntry:
    """Entry in the bibcode index."""
    bibcode: str
    year: int
    row_id: int
    model: str
    vector_file: str


class Index:
    """Lightweight index for bibcode lookup and metadata management."""
    
    def __init__(self, index_path: Path):
        """
        Initialize index.
        
        Args:
            index_path: Path to SQLite index database
        """
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.index_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    bibcode TEXT,
                    year INTEGER,
                    row_id INTEGER,
                    model TEXT,
                    vector_file TEXT,
                    fields_hash TEXT,
                    prompt_hash TEXT,
                    created_at TEXT,
                    PRIMARY KEY (bibcode, model)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bibcode ON embeddings(bibcode)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_year_model ON embeddings(year, model)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manifests (
                    year INTEGER,
                    model TEXT,
                    manifest_data TEXT,
                    PRIMARY KEY (year, model)
                )
            """)
    
    def add_manifest(self, manifest: Manifest, bibcodes: List[str]) -> None:
        """
        Add a manifest and its bibcodes to the index.
        
        Args:
            manifest: Manifest describing the embeddings
            bibcodes: List of bibcodes in order
        """
        with sqlite3.connect(self.index_path) as conn:
            # Store manifest metadata
            conn.execute("""
                INSERT OR REPLACE INTO manifests (year, model, manifest_data)
                VALUES (?, ?, ?)
            """, (manifest.year, manifest.model, json.dumps(manifest.to_dict())))
            
            # Store individual bibcode entries
            entries = [
                (
                    bibcode,
                    manifest.year,
                    row_id,
                    manifest.model,
                    manifest.vector_file,
                    manifest.fields_hash,
                    manifest.prompt_hash,
                    manifest.created_at
                )
                for row_id, bibcode in enumerate(bibcodes)
            ]
            
            conn.executemany("""
                INSERT OR REPLACE INTO embeddings 
                (bibcode, year, row_id, model, vector_file, fields_hash, prompt_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, entries)
    
    def lookup_bibcode(self, bibcode: str, model: Optional[str] = None) -> List[IndexEntry]:
        """
        Look up embeddings for a bibcode.
        
        Args:
            bibcode: Bibcode to search for
            model: Optional model filter
            
        Returns:
            List of index entries matching the bibcode
        """
        with sqlite3.connect(self.index_path) as conn:
            if model:
                cursor = conn.execute("""
                    SELECT bibcode, year, row_id, model, vector_file
                    FROM embeddings 
                    WHERE bibcode = ? AND model = ?
                """, (bibcode, model))
            else:
                cursor = conn.execute("""
                    SELECT bibcode, year, row_id, model, vector_file
                    FROM embeddings 
                    WHERE bibcode = ?
                """, (bibcode,))
            
            return [
                IndexEntry(
                    bibcode=row[0],
                    year=row[1],
                    row_id=row[2],
                    model=row[3],
                    vector_file=row[4]
                )
                for row in cursor.fetchall()
            ]
    
    def lookup_year_model(self, year: int, model: str) -> Optional[Manifest]:
        """
        Look up manifest for a specific year and model.
        
        Args:
            year: Year to search for
            model: Model to search for
            
        Returns:
            Manifest if found, None otherwise
        """
        with sqlite3.connect(self.index_path) as conn:
            cursor = conn.execute("""
                SELECT manifest_data 
                FROM manifests 
                WHERE year = ? AND model = ?
            """, (year, model))
            
            row = cursor.fetchone()
            if row:
                manifest_data = json.loads(row[0])
                return Manifest.from_dict(manifest_data)
            
            return None
    
    def list_years(self) -> List[int]:
        """
        List all years available in the index.
        
        Returns:
            Sorted list of years
        """
        with sqlite3.connect(self.index_path) as conn:
            cursor = conn.execute("SELECT DISTINCT year FROM embeddings ORDER BY year")
            return [row[0] for row in cursor.fetchall()]
    
    def list_models(self) -> List[str]:
        """
        List all models available in the index.
        
        Returns:
            List of model names
        """
        with sqlite3.connect(self.index_path) as conn:
            cursor = conn.execute("SELECT DISTINCT model FROM embeddings")
            return [row[0] for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        with sqlite3.connect(self.index_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT bibcode) FROM embeddings")
            unique_bibcodes = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM manifests")
            total_manifests = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT year, model, COUNT(*) 
                FROM embeddings 
                GROUP BY year, model 
                ORDER BY year, model
            """)
            year_model_counts = {f"{row[0]}_{row[1]}": row[2] for row in cursor.fetchall()}
            
            return {
                "total_embeddings": total_embeddings,
                "unique_bibcodes": unique_bibcodes,
                "total_manifests": total_manifests,
                "year_model_counts": year_model_counts
            }
    
    def remove_year_model(self, year: int, model: str) -> bool:
        """
        Remove all entries for a specific year and model.
        
        Args:
            year: Year to remove
            model: Model to remove
            
        Returns:
            True if entries were removed, False if none found
        """
        with sqlite3.connect(self.index_path) as conn:
            cursor = conn.execute("""
                DELETE FROM embeddings 
                WHERE year = ? AND model = ?
            """, (year, model))
            
            conn.execute("""
                DELETE FROM manifests 
                WHERE year = ? AND model = ?
            """, (year, model))
            
            return cursor.rowcount > 0


class VectorIndex:
    """Vector similarity search index using Faiss."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize vector index.
        
        Args:
            output_dir: Directory to store Faiss indexes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._faiss = None
        self._init_faiss()
    
    def _init_faiss(self) -> None:
        """Initialize Faiss library if available."""
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            print("Warning: Faiss not available. Vector search will be disabled.")
            self._faiss = None
    
    def create_index(
        self,
        embeddings: np.ndarray,
        year: int,
        model: str,
        index_type: str = "flat"
    ) -> Optional[Path]:
        """
        Create Faiss index for embeddings.
        
        Args:
            embeddings: Array of embeddings to index
            year: Year of the data
            model: Model name
            index_type: Type of Faiss index ("flat", "hnsw")
            
        Returns:
            Path to created index file, None if Faiss unavailable
        """
        if self._faiss is None:
            return None
        
        dim = embeddings.shape[1]
        
        # Create appropriate index type
        if index_type == "flat":
            index = self._faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        elif index_type == "hnsw":
            index = self._faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add vectors to index
        # Normalize for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        index.add(normalized_embeddings.astype(np.float32))
        
        # Save index
        model_safe = model.replace("/", "_").replace(":", "_")
        index_file = f"index_{model_safe}_{year}_{index_type}.faiss"
        index_path = self.output_dir / index_file
        
        self._faiss.write_index(index, str(index_path))
        
        return index_path
    
    def search(
        self,
        index_path: Path,
        query_vector: np.ndarray,
        k: int = 10
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Search for similar vectors.
        
        Args:
            index_path: Path to Faiss index file
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Tuple of (scores, indices) or None if Faiss unavailable
        """
        if self._faiss is None or not index_path.exists():
            return None
        
        # Load index
        index = self._faiss.read_index(str(index_path))
        
        # Normalize query vector
        query_normalized = query_vector / np.linalg.norm(query_vector)
        query_normalized = query_normalized.reshape(1, -1).astype(np.float32)
        
        # Search
        scores, indices = index.search(query_normalized, k)
        
        return scores[0], indices[0]
