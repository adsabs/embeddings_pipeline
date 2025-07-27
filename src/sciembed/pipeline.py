"""Main pipeline class orchestrating the embedding process."""

from typing import List, Dict, Any, Optional, Iterator, Tuple
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dataclasses import dataclass
import logging
from tqdm import tqdm

from .config import Config
from .components import Loader, Preparer, Embedder, Persister, Index
from .components.loader import JSONLoader, DirectoryLoader
from .components.preparer import Preparer as TextPreparer
from .components.embedder import create_embedder
from .components.persister import Persister as VectorPersister, Manifest
from .components.index import Index as BibcodeIndex, VectorIndex
from .components.deduplicator import Deduplicator


@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""
    total_records: int = 0
    processed_records: int = 0
    skipped_records: int = 0
    failed_records: int = 0
    duplicate_records: int = 0
    total_batches: int = 0
    processing_time: float = 0.0
    embedding_time: float = 0.0


class Pipeline:
    """Main pipeline for processing scientific papers into embeddings."""
    
    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stats = PipelineStats()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize components
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Data loader
        json_loader = JSONLoader(show_progress=self.config.show_progress)
        self.loader = DirectoryLoader(json_loader)
        
        # Text preparer
        self.preparer = TextPreparer(self.config.preparer_config)
        
        # Embedder
        self.embedder = create_embedder(self.config.embedder_config)
        
        # Persister
        self.persister = VectorPersister(
            self.config.output_dir,
            use_float16=self.config.use_float16
        )
        
        # Indexes
        index_path = self.config.output_dir / "index.db"
        self.bibcode_index = BibcodeIndex(index_path)
        
        if self.config.create_faiss_index:
            self.vector_index = VectorIndex(self.config.output_dir)
        else:
            self.vector_index = None
        
        # Deduplicator
        dedup_config = self.config.deduplication_config
        if dedup_config.enabled:
            db_path = dedup_config.get_db_path(self.config.output_dir)
            self.deduplicator = Deduplicator(db_path, dedup_config.use_rocksdb)
        else:
            self.deduplicator = None
        
        self.logger.info(f"Initialized pipeline with model: {self.embedder.name}")
        self.logger.info(f"Embedding dimension: {self.embedder.dim}")
        if self.deduplicator:
            self.logger.info("Deduplication enabled")
    
    def run(self) -> PipelineStats:
        """
        Run the complete embedding pipeline.
        
        Returns:
            Pipeline execution statistics
        """
        import time
        start_time = time.time()
        
        self.logger.info("Starting embedding pipeline")
        self.logger.info(f"Processing years: {self.config.years}")
        self.logger.info(f"Fields: {self.config.fields}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        
        try:
            for year in self.config.years:
                self._process_year(year)
            
            self.stats.processing_time = time.time() - start_time
            
            self.logger.info("Pipeline completed successfully")
            self.logger.info(f"Total records processed: {self.stats.processed_records}")
            self.logger.info(f"Total time: {self.stats.processing_time:.2f}s")
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _process_year(self, year: int) -> None:
        """
        Process embeddings for a single year.
        
        Args:
            year: Year to process
        """
        self.logger.info(f"Processing year {year}")
        
        # Check if already processed (resumability)
        if self.config.resume:
            existing_manifest = self.bibcode_index.lookup_year_model(year, self.embedder.name)
            if existing_manifest:
                self.logger.info(f"Year {year} already processed for model {self.embedder.name}, skipping")
                return
        
        # Load data for the year
        try:
            records = self.loader.load_years(
                self.config.input_dir,
                [year],
                fields=self.config.fields + ["bibcode"]  # Always include bibcode
            )
            
            # Process in batches
            embeddings, bibcodes = self._process_records_batched(records, year)
            
            if not embeddings:
                self.logger.warning(f"No embeddings generated for year {year}")
                return
            
            # Save embeddings and metadata
            manifest = self.persister.save_embeddings(
                embeddings=embeddings,
                bibcodes=bibcodes,
                year=year,
                model=self.embedder.name,
                fields_hash=self.config.preparer_config.hash(),
                prompt_hash=self.config.preparer_config.hash()  # Same for now
            )
            
            # Update index
            self.bibcode_index.add_manifest(manifest, bibcodes)
            
            # Create vector index if requested
            if self.vector_index:
                index_path = self.vector_index.create_index(
                    embeddings, year, self.embedder.name, self.config.faiss_index_type
                )
                if index_path:
                    self.logger.info(f"Created vector index: {index_path}")
            
            self.logger.info(f"Completed year {year}: {len(embeddings)} embeddings")
            
        except FileNotFoundError as e:
            self.logger.error(f"Data file not found for year {year}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to process year {year}: {e}")
            raise
    
    def _process_records_batched(
        self, 
        records: Iterator[Dict[str, Any]], 
        year: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process records in batches for efficient embedding.
        
        Args:
            records: Iterator of record dictionaries
            year: Year being processed
            
        Returns:
            Tuple of (embeddings array, bibcodes list)
        """
        all_embeddings = []
        all_bibcodes = []
        
        # Prepare texts
        prepared_pairs = list(self.preparer.prepare_stream(records))
        
        if not prepared_pairs:
            return np.array([]), []
        
        self.stats.total_records += len(prepared_pairs)
        
        # Apply deduplication if enabled
        if self.deduplicator:
            # Convert to format expected by deduplicator
            dedup_records = [(bibcode, text, year) for bibcode, text in prepared_pairs]
            unique_records, duplicates = self.deduplicator.process_batch(dedup_records)
            
            # Update stats
            self.stats.duplicate_records += len(duplicates)
            
            # Log duplicates if configured
            if self.config.deduplication_config.log_duplicates and duplicates:
                for dup in duplicates:
                    self.logger.info(
                        f"Duplicate found: {dup.bibcode} (year {dup.year}) "
                        f"matches {dup.first_seen_bibcode} (year {dup.first_seen_year})"
                    )
            
            # Use only unique records for embedding
            prepared_pairs = [(bibcode, text) for bibcode, text, _ in unique_records]
            
            if not prepared_pairs:
                self.logger.info(f"All {len(duplicates)} records for year {year} were duplicates")
                return np.array([]), []
        
        # Calculate optimal batch size
        avg_text_len = sum(len(text) for _, text in prepared_pairs) / len(prepared_pairs)
        optimal_batch_size = self.embedder.batch_size(int(avg_text_len))
        
        self.logger.info(f"Processing {len(prepared_pairs)} records in batches of {optimal_batch_size}")
        
        # Process in batches
        progress_bar = None
        if self.config.show_progress:
            progress_bar = tqdm(
                total=len(prepared_pairs),
                desc=f"Embedding {year}",
                unit="records"
            )
        
        try:
            for i in range(0, len(prepared_pairs), optimal_batch_size):
                batch_pairs = prepared_pairs[i:i + optimal_batch_size]
                
                # Extract texts and bibcodes
                batch_bibcodes = [pair[0] for pair in batch_pairs]
                batch_texts = [pair[1] for pair in batch_pairs]
                
                # Generate embeddings
                try:
                    import time
                    embed_start = time.time()
                    batch_embeddings = self.embedder.embed_batch(batch_texts)
                    self.stats.embedding_time += time.time() - embed_start
                    
                    # Accumulate results
                    all_embeddings.append(batch_embeddings)
                    all_bibcodes.extend(batch_bibcodes)
                    
                    self.stats.processed_records += len(batch_texts)
                    self.stats.total_batches += 1
                    
                    if progress_bar:
                        progress_bar.update(len(batch_texts))
                        
                except Exception as e:
                    self.logger.error(f"Failed to embed batch: {e}")
                    self.stats.failed_records += len(batch_texts)
                    if progress_bar:
                        progress_bar.update(len(batch_texts))
                    continue
        
        finally:
            if progress_bar:
                progress_bar.close()
        
        # Combine all embeddings
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            return combined_embeddings, all_bibcodes
        else:
            return np.array([]), []
    
    def search_similar(
        self,
        query_text: str,
        year: int,
        k: int = 10
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Search for similar papers using vector similarity.
        
        Args:
            query_text: Text to search for
            year: Year to search in
            k: Number of results to return
            
        Returns:
            List of (bibcode, similarity_score) tuples, or None if no index
        """
        if not self.vector_index:
            self.logger.warning("Vector index not available")
            return None
        
        # Generate query embedding
        query_embedding = self.embedder.embed_single(query_text)
        
        # Find vector index file
        model_safe = self.embedder.name.replace("/", "_").replace(":", "_")
        index_file = f"index_{model_safe}_{year}_{self.config.faiss_index_type}.faiss"
        index_path = self.config.output_dir / index_file
        
        if not index_path.exists():
            self.logger.warning(f"Vector index not found for year {year}")
            return None
        
        # Search
        search_result = self.vector_index.search(index_path, query_embedding, k)
        if search_result is None:
            return None
        
        scores, indices = search_result
        
        # Get corresponding bibcodes
        manifest = self.bibcode_index.lookup_year_model(year, self.embedder.name)
        if not manifest:
            self.logger.warning(f"Manifest not found for year {year}")
            return None
        
        _, bibcodes = self.persister.load_embeddings(manifest)
        
        # Return results
        results = []
        for score, idx in zip(scores, indices):
            if 0 <= idx < len(bibcodes):
                results.append((bibcodes[idx], float(score)))
        
        return results
    
    def get_embedding(self, bibcode: str, model: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Retrieve embedding for a specific bibcode.
        
        Args:
            bibcode: Bibcode to look up
            model: Model name (defaults to current embedder)
            
        Returns:
            Embedding vector or None if not found
        """
        if model is None:
            model = self.embedder.name
        
        # Look up in index
        entries = self.bibcode_index.lookup_bibcode(bibcode, model)
        if not entries:
            return None
        
        # Get the first entry (there should only be one per model)
        entry = entries[0]
        
        # Load manifest and embeddings
        manifest = self.bibcode_index.lookup_year_model(entry.year, entry.model)
        if not manifest:
            return None
        
        embeddings, bibcodes = self.persister.load_embeddings(manifest)
        
        # Find the specific embedding
        if entry.row_id < len(embeddings):
            return embeddings[entry.row_id]
        
        return None
