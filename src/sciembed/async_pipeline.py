"""Async pipeline with producer/consumer queues for high-throughput processing."""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Iterator, Tuple, AsyncIterator
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from queue import Queue, Empty
import threading

from .config import Config
from .pipeline import PipelineStats
from .components import Loader, Preparer, Embedder, Persister, Index
from .components.loader import JSONLoader, DirectoryLoader
from .components.preparer import Preparer as TextPreparer
from .components.embedder import create_embedder
from .components.persister import Persister as VectorPersister, Manifest
from .components.index import Index as BibcodeIndex, VectorIndex
from .components.deduplicator import Deduplicator


@dataclass
class QueueItem:
    """Item passed between async pipeline stages."""
    bibcode: str
    text: str
    year: int
    batch_id: int = 0


@dataclass
class EmbeddingResult:
    """Result from embedding stage."""
    bibcodes: List[str]
    embeddings: np.ndarray
    year: int
    batch_id: int


class AsyncPipeline:
    """High-throughput async pipeline with producer/consumer queues."""
    
    def __init__(self, config: Config, max_queue_size: int = 1000):
        """
        Initialize async pipeline.
        
        Args:
            config: Pipeline configuration
            max_queue_size: Maximum items in each queue
        """
        self.config = config
        self.max_queue_size = max_queue_size
        self.stats = PipelineStats()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Async queues
        self.load_queue: asyncio.Queue = None
        self.prep_queue: asyncio.Queue = None
        self.embed_queue: asyncio.Queue = None
        self.persist_queue: asyncio.Queue = None
        
        # Thread-safe completion tracking
        self._completion_event = threading.Event()
        self._error_event = threading.Event()
        self._errors = []
        
        # Initialize components
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize pipeline components."""
        # Data loader
        json_loader = JSONLoader(show_progress=False)  # Progress handled centrally
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
        
        self.logger.info(f"Initialized async pipeline with model: {self.embedder.name}")
        self.logger.info(f"Embedding dimension: {self.embedder.dim}")
        if self.deduplicator:
            self.logger.info("Deduplication enabled")
    
    async def run(self) -> PipelineStats:
        """
        Run the complete async embedding pipeline.
        
        Returns:
            Pipeline execution statistics
        """
        start_time = time.time()
        
        self.logger.info("Starting async embedding pipeline")
        self.logger.info(f"Processing years: {self.config.years}")
        self.logger.info(f"Fields: {self.config.fields}")
        self.logger.info(f"Max queue size: {self.max_queue_size}")
        
        try:
            # Initialize queues
            self.load_queue = asyncio.Queue(maxsize=self.max_queue_size)
            self.prep_queue = asyncio.Queue(maxsize=self.max_queue_size)
            self.embed_queue = asyncio.Queue(maxsize=self.max_queue_size // 4)  # Smaller for batched items
            self.persist_queue = asyncio.Queue(maxsize=self.max_queue_size // 8)  # Even smaller
            
            # Process each year
            for year in self.config.years:
                await self._process_year_async(year)
            
            self.stats.processing_time = time.time() - start_time
            
            self.logger.info("Async pipeline completed successfully")
            self.logger.info(f"Total records processed: {self.stats.processed_records}")
            self.logger.info(f"Total time: {self.stats.processing_time:.2f}s")
            self.logger.info(f"Throughput: {self.stats.processed_records / self.stats.processing_time:.1f} records/sec")
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"Async pipeline failed: {e}")
            raise
    
    async def _process_year_async(self, year: int) -> None:
        """
        Process embeddings for a single year using async workers.
        
        Args:
            year: Year to process
        """
        self.logger.info(f"Processing year {year} with async pipeline")
        
        # Check if already processed (resumability)
        if self.config.resume:
            existing_manifest = self.bibcode_index.lookup_year_model(year, self.embedder.name)
            if existing_manifest:
                self.logger.info(f"Year {year} already processed for model {self.embedder.name}, skipping")
                return
        
        # Clear completion event
        self._completion_event.clear()
        self._error_event.clear()
        self._errors.clear()
        
        # Start async workers
        tasks = []
        
        # Data loading worker (I/O bound)
        tasks.append(asyncio.create_task(self._load_worker(year)))
        
        # Text preparation workers (CPU bound)
        num_prep_workers = min(4, self.config.num_workers or 4)
        for i in range(num_prep_workers):
            tasks.append(asyncio.create_task(self._prep_worker(f"prep-{i}")))
        
        # Embedding workers (GPU/API bound) 
        num_embed_workers = 1 if self.embedder.config.model_type == "openai" else 2
        for i in range(num_embed_workers):
            tasks.append(asyncio.create_task(self._embed_worker(f"embed-{i}")))
        
        # Persistence worker (I/O bound)
        tasks.append(asyncio.create_task(self._persist_worker(year)))
        
        # Progress monitoring
        if self.config.show_progress:
            tasks.append(asyncio.create_task(self._progress_monitor(year)))
        
        try:
            # Wait for completion or error
            await asyncio.gather(*tasks)
            
            if self._error_event.is_set():
                raise Exception(f"Pipeline errors: {self._errors}")
                
        except asyncio.CancelledError:
            self.logger.info("Pipeline tasks cancelled")
            raise
        except Exception as e:
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            raise
    
    async def _load_worker(self, year: int) -> None:
        """Load data and put into preparation queue."""
        try:
            self.logger.info(f"Starting loader worker for year {year}")
            
            # Load records for the year
            records = self.loader.load_years(
                self.config.input_dir,
                [year],
                fields=self.config.fields + ["bibcode"]
            )
            
            count = 0
            for record in records:
                if self._error_event.is_set():
                    break
                    
                if "bibcode" not in record:
                    continue
                    
                # Create queue item
                item = QueueItem(
                    bibcode=record["bibcode"],
                    text="",  # Will be prepared later
                    year=year
                )
                
                # Add raw record data
                setattr(item, '_record', record)
                
                await self.load_queue.put(item)
                count += 1
            
            self.stats.total_records += count
            self.logger.info(f"Loader finished: {count} records queued for year {year}")
            
            # Signal end of loading
            await self.load_queue.put(None)
            
        except Exception as e:
            self.logger.error(f"Load worker failed: {e}")
            self._errors.append(f"Load: {e}")
            self._error_event.set()
    
    async def _prep_worker(self, worker_id: str) -> None:
        """Prepare text from loaded records."""
        try:
            self.logger.debug(f"Starting prep worker {worker_id}")
            
            while True:
                if self._error_event.is_set():
                    break
                    
                try:
                    # Get item from load queue
                    item = await asyncio.wait_for(self.load_queue.get(), timeout=1.0)
                    
                    if item is None:  # End signal
                        await self.prep_queue.put(None)
                        break
                    
                    # Prepare text
                    record = getattr(item, '_record')
                    prepared_pairs = list(self.preparer.prepare_stream([record]))
                    
                    if prepared_pairs:
                        bibcode, text = prepared_pairs[0]
                        
                        # Apply deduplication if enabled
                        if self.deduplicator:
                            duplicate_info = self.deduplicator.check_duplicate(bibcode, text, item.year)
                            if duplicate_info:
                                # Log duplicate if configured
                                if self.config.deduplication_config.log_duplicates:
                                    self.logger.info(
                                        f"Duplicate found: {duplicate_info.bibcode} (year {duplicate_info.year}) "
                                        f"matches {duplicate_info.first_seen_bibcode} (year {duplicate_info.first_seen_year})"
                                    )
                                # Skip this record
                                continue
                            else:
                                # Add to deduplication database
                                self.deduplicator.add_record(bibcode, text, item.year)
                        
                        item.text = text
                        await self.prep_queue.put(item)
                    
                    self.load_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Prep worker {worker_id} error: {e}")
                    self._errors.append(f"Prep {worker_id}: {e}")
                    self._error_event.set()
                    break
            
            self.logger.debug(f"Prep worker {worker_id} finished")
            
        except Exception as e:
            self.logger.error(f"Prep worker {worker_id} failed: {e}")
            self._errors.append(f"Prep {worker_id}: {e}")
            self._error_event.set()
    
    async def _embed_worker(self, worker_id: str) -> None:
        """Batch and embed prepared texts."""
        try:
            self.logger.debug(f"Starting embed worker {worker_id}")
            
            batch = []
            batch_id = 0
            end_signals = 0
            num_prep_workers = min(4, self.config.num_workers or 4)
            
            while True:
                if self._error_event.is_set():
                    break
                    
                try:
                    # Get item from prep queue
                    item = await asyncio.wait_for(self.prep_queue.get(), timeout=1.0)
                    
                    if item is None:  # End signal
                        end_signals += 1
                        if end_signals >= num_prep_workers:
                            # Process final batch
                            if batch:
                                await self._process_embedding_batch(batch, batch_id)
                            await self.embed_queue.put(None)
                            break
                        continue
                    
                    batch.append(item)
                    
                    # Calculate optimal batch size
                    if not batch:
                        continue
                        
                    avg_len = sum(len(item.text) for item in batch) / len(batch)
                    optimal_size = self.embedder.batch_size(int(avg_len))
                    
                    # Process batch when full
                    if len(batch) >= optimal_size:
                        await self._process_embedding_batch(batch, batch_id)
                        batch = []
                        batch_id += 1
                    
                    self.prep_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Process partial batch if timeout
                    if batch:
                        await self._process_embedding_batch(batch, batch_id)
                        batch = []
                        batch_id += 1
                    continue
                except Exception as e:
                    self.logger.error(f"Embed worker {worker_id} error: {e}")
                    self._errors.append(f"Embed {worker_id}: {e}")
                    self._error_event.set()
                    break
            
            self.logger.debug(f"Embed worker {worker_id} finished")
            
        except Exception as e:
            self.logger.error(f"Embed worker {worker_id} failed: {e}")
            self._errors.append(f"Embed {worker_id}: {e}")
            self._error_event.set()
    
    async def _process_embedding_batch(self, batch: List[QueueItem], batch_id: int) -> None:
        """Process a batch of items for embedding."""
        if not batch:
            return
            
        try:
            # Extract texts
            texts = [item.text for item in batch]
            bibcodes = [item.bibcode for item in batch]
            year = batch[0].year
            
            # Generate embeddings in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                embed_start = time.time()
                embeddings = await loop.run_in_executor(
                    executor, 
                    self.embedder.embed_batch, 
                    texts
                )
                self.stats.embedding_time += time.time() - embed_start
            
            # Create result
            result = EmbeddingResult(
                bibcodes=bibcodes,
                embeddings=embeddings,
                year=year,
                batch_id=batch_id
            )
            
            await self.embed_queue.put(result)
            
            self.stats.processed_records += len(texts)
            self.stats.total_batches += 1
            
        except Exception as e:
            self.logger.error(f"Embedding batch failed: {e}")
            self.stats.failed_records += len(batch)
            raise
    
    async def _persist_worker(self, year: int) -> None:
        """Persist embeddings and update indexes."""
        try:
            self.logger.debug(f"Starting persist worker for year {year}")
            
            all_embeddings = []
            all_bibcodes = []
            end_signals = 0
            num_embed_workers = 1 if self.embedder.config.model_type == "openai" else 2
            
            while True:
                if self._error_event.is_set():
                    break
                    
                try:
                    # Get result from embed queue
                    result = await asyncio.wait_for(self.embed_queue.get(), timeout=2.0)
                    
                    if result is None:  # End signal
                        end_signals += 1
                        if end_signals >= num_embed_workers:
                            break
                        continue
                    
                    # Accumulate results
                    all_embeddings.append(result.embeddings)
                    all_bibcodes.extend(result.bibcodes)
                    
                    self.embed_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Persist worker error: {e}")
                    self._errors.append(f"Persist: {e}")
                    self._error_event.set()
                    break
            
            # Save all embeddings
            if all_embeddings:
                combined_embeddings = np.vstack(all_embeddings)
                
                # Save in thread pool (I/O bound)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    manifest = await loop.run_in_executor(
                        executor,
                        self.persister.save_embeddings,
                        combined_embeddings,
                        all_bibcodes,
                        year,
                        self.embedder.name,
                        self.config.preparer_config.hash(),
                        self.config.preparer_config.hash()
                    )
                
                # Update index
                self.bibcode_index.add_manifest(manifest, all_bibcodes)
                
                # Create vector index if requested
                if self.vector_index:
                    index_path = await loop.run_in_executor(
                        executor,
                        self.vector_index.create_index,
                        combined_embeddings,
                        year,
                        self.embedder.name,
                        self.config.faiss_index_type
                    )
                    if index_path:
                        self.logger.info(f"Created vector index: {index_path}")
                
                self.logger.info(f"Persisted {len(combined_embeddings)} embeddings for year {year}")
            
            # Signal completion
            self._completion_event.set()
            self.logger.debug(f"Persist worker finished for year {year}")
            
        except Exception as e:
            self.logger.error(f"Persist worker failed: {e}")
            self._errors.append(f"Persist: {e}")
            self._error_event.set()
    
    async def _progress_monitor(self, year: int) -> None:
        """Monitor and display progress."""
        try:
            from tqdm.asyncio import tqdm
            
            # Wait a bit for stats to accumulate
            await asyncio.sleep(2)
            
            with tqdm(desc=f"Embedding {year}", unit="records") as pbar:
                last_processed = 0
                
                while not self._completion_event.is_set() and not self._error_event.is_set():
                    current_processed = self.stats.processed_records
                    delta = current_processed - last_processed
                    
                    if delta > 0:
                        pbar.update(delta)
                        last_processed = current_processed
                    
                    await asyncio.sleep(1)
                
                # Final update
                final_delta = self.stats.processed_records - last_processed
                if final_delta > 0:
                    pbar.update(final_delta)
            
        except ImportError:
            # Fallback to simple logging
            while not self._completion_event.is_set() and not self._error_event.is_set():
                await asyncio.sleep(10)
                self.logger.info(f"Progress: {self.stats.processed_records} records processed")
        except Exception as e:
            self.logger.error(f"Progress monitor failed: {e}")
