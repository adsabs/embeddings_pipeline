"""Pipeline runner that chooses between sync and async implementations."""

import asyncio
from typing import Union

from .config import Config
from .pipeline import Pipeline, PipelineStats
from .async_pipeline import AsyncPipeline


def run_pipeline(config: Config) -> PipelineStats:
    """
    Run the embedding pipeline using sync or async implementation.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Pipeline execution statistics
    """
    if config.use_async:
        return asyncio.run(run_async_pipeline(config))
    else:
        return run_sync_pipeline(config)


def run_sync_pipeline(config: Config) -> PipelineStats:
    """
    Run the synchronous pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Pipeline execution statistics
    """
    pipeline = Pipeline(config)
    return pipeline.run()


async def run_async_pipeline(config: Config) -> PipelineStats:
    """
    Run the asynchronous pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Pipeline execution statistics
    """
    pipeline = AsyncPipeline(config)
    return await pipeline.run()


def create_pipeline(config: Config) -> Union[Pipeline, AsyncPipeline]:
    """
    Create a pipeline instance based on configuration.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Pipeline instance (sync or async)
    """
    if config.use_async:
        return AsyncPipeline(config)
    else:
        return Pipeline(config)
