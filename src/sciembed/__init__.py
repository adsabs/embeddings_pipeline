"""
Scientific embeddings pipeline for astronomical literature.

A high-performance, configurable pipeline for embedding scientific papers
from the Astrophysics Data System (ADS) using various embedding models.
"""

from .pipeline import Pipeline
from .async_pipeline import AsyncPipeline
from .config import Config
from .components import Loader, Preparer, Embedder, Persister, Index

__version__ = "0.1.0"
__all__ = ["Pipeline", "AsyncPipeline", "Config", "Loader", "Preparer", "Embedder", "Persister", "Index"]
