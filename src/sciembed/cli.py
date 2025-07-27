"""Command-line interface for the sciembed pipeline."""

import click
from pathlib import Path
from typing import List, Optional

from .config import Config, load_config
from .runner import run_pipeline


@click.group()
@click.version_option()
def main():
    """Scientific embeddings pipeline for astronomical literature."""
    pass


@main.command()
@click.option(
    "--input", "-i",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory containing yearly JSON files"
)
@click.option(
    "--output", "-o", 
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for embeddings and indexes"
)
@click.option(
    "--years",
    type=str,
    required=True,
    help="Years to process (e.g., '2020' or '2015:2020' or '2018,2019,2020')"
)
@click.option(
    "--fields",
    type=str,
    default="title,abstract",
    help="Comma-separated list of fields to embed"
)
@click.option(
    "--model", "-m",
    type=str,
    default="hf://sentence-transformers/all-MiniLM-L6-v2",
    help="Embedding model (e.g., 'openai://text-embedding-3-small' or 'hf://model-name')"
)
@click.option(
    "--prefix",
    type=str,
    default="",
    help="Prefix to add to each text before embedding"
)
@click.option(
    "--suffix", 
    type=str,
    default="",
    help="Suffix to add to each text after embedding"
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=32,
    help="Batch size for embedding"
)
@click.option(
    "--config", "-c",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file (YAML or JSON)"
)
@click.option(
    "--api-key",
    type=str,
    help="API key for remote embedding services"
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device for local models ('cpu', 'cuda', 'auto')"
)
@click.option(
    "--no-faiss",
    is_flag=True,
    help="Skip creating Faiss vector index"
)
@click.option(
    "--no-progress",
    is_flag=True, 
    help="Disable progress bars"
)
@click.option(
    "--async",
    "use_async",
    is_flag=True,
    help="Use async pipeline for higher throughput"
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Number of worker threads for async pipeline"
)
def ingest(
    input_dir: Path,
    output_dir: Path,
    years: str,
    fields: str,
    model: str,
    prefix: str,
    suffix: str,
    batch_size: int,
    config_file: Optional[Path],
    api_key: Optional[str],
    device: str,
    no_faiss: bool,
    no_progress: bool,
    use_async: bool,
    workers: int
):
    """Ingest and embed scientific papers."""
    
    # Parse years
    parsed_years = _parse_years(years)
    
    # Parse fields
    parsed_fields = [f.strip() for f in fields.split(",")]
    
    # Load configuration
    config_overrides = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "years": parsed_years,
        "fields": parsed_fields,
        "model": model,
        "prefix": prefix,
        "suffix": suffix,
        "batch_size": batch_size,
        "create_faiss_index": not no_faiss,
        "show_progress": not no_progress,
        "device": device,
        "use_async": use_async,
        "num_workers": workers,
    }
    
    if api_key:
        config_overrides["api_key"] = api_key
    
    config = load_config(config_file, **config_overrides)
    
    # Run pipeline
    stats = run_pipeline(config)
    
    # Print summary
    click.echo("\n" + "="*50)
    click.echo("EMBEDDING PIPELINE COMPLETED")
    click.echo("="*50)
    click.echo(f"Total records processed: {stats.processed_records:,}")
    click.echo(f"Duplicate records skipped: {stats.duplicate_records:,}")
    click.echo(f"Total batches: {stats.total_batches:,}")
    click.echo(f"Processing time: {stats.processing_time:.2f}s")
    click.echo(f"Embedding time: {stats.embedding_time:.2f}s")
    click.echo(f"Output directory: {output_dir}")


@main.command()
@click.option(
    "--output", "-o",
    "output_dir", 
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing embeddings"
)
@click.option(
    "--query", "-q",
    type=str,
    required=True,
    help="Text query to search for"
)
@click.option(
    "--year", "-y",
    type=int,
    required=True,
    help="Year to search in"
)
@click.option(
    "--model", "-m",
    type=str,
    help="Model name (if not specified, uses first available)"
)
@click.option(
    "--limit", "-k",
    type=int,
    default=10,
    help="Number of results to return"
)
@click.option(
    "--config", "-c",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file for embedding model setup"
)
def search(
    output_dir: Path,
    query: str,
    year: int,
    model: Optional[str],
    limit: int,
    config_file: Optional[Path]
):
    """Search for similar papers using vector similarity."""
    
    # Load configuration for model setup
    config = load_config(config_file, output_dir=output_dir) if config_file else Config(
        input_dir=".",  # Not used for search
        output_dir=output_dir,
        years=[year]
    )
    
    # Override model if specified
    if model:
        config.model = model
    
    # Create pipeline for search
    pipeline = Pipeline(config)
    
    # Perform search
    results = pipeline.search_similar(query, year, limit)
    
    if not results:
        click.echo("No results found or vector index not available.")
        return
    
    # Display results
    click.echo(f"\nSearch results for: '{query}' (Year: {year})")
    click.echo("="*60)
    
    for i, (bibcode, score) in enumerate(results, 1):
        click.echo(f"{i:2d}. {bibcode} (similarity: {score:.4f})")


@main.command()
@click.option(
    "--output", "-o",
    "output_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path), 
    required=True,
    help="Directory containing embeddings"
)
def info(output_dir: Path):
    """Show information about processed embeddings."""
    
    from .components.index import Index
    
    # Load index
    index_path = output_dir / "index.db"
    if not index_path.exists():
        click.echo("No index found. Run 'sciembed ingest' first.")
        return
    
    index = Index(index_path)
    stats = index.get_stats()
    years = index.list_years()
    models = index.list_models()
    
    # Display information
    click.echo("EMBEDDINGS INFORMATION")
    click.echo("="*40)
    click.echo(f"Total embeddings: {stats['total_embeddings']:,}")
    click.echo(f"Unique bibcodes: {stats['unique_bibcodes']:,}")
    click.echo(f"Total manifests: {stats['total_manifests']}")
    click.echo(f"Years available: {', '.join(map(str, years))}")
    click.echo(f"Models available: {', '.join(models)}")
    
    if stats['year_model_counts']:
        click.echo("\nBreakdown by year and model:")
        for key, count in stats['year_model_counts'].items():
            click.echo(f"  {key}: {count:,} embeddings")


def _parse_years(years_str: str) -> List[int]:
    """Parse years string into list of integers."""
    if ":" in years_str:
        # Range format: "2015:2020"
        start, end = years_str.split(":", 1)
        return list(range(int(start), int(end) + 1))
    elif "," in years_str:
        # Comma-separated: "2018,2019,2020"
        return [int(y.strip()) for y in years_str.split(",")]
    else:
        # Single year: "2020"
        return [int(years_str)]


if __name__ == "__main__":
    main()
