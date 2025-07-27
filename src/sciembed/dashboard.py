"""Streamlit dashboard for scientific embeddings pipeline experimentation."""

import streamlit as st
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml

import sys

# Add the src directory to the path for absolute imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Core imports only - heavy imports moved to functions
from sciembed.config import Config, load_config
from sciembed.components.embedder import create_embedder, EmbedderConfig
from sciembed.components.loader import JSONLoader
from sciembed.components.preparer import Preparer, PreparerConfig


def main():
    """Main Streamlit dashboard application."""
    st.set_page_config(
        page_title="SciX Experimental Embeddings Pipeline",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("SciX Experimental Embeddings Pipeline")
    st.markdown("*Experimental testing interface for SciX corpus embeddings*")
    
    # Initialize session state
    if 'embeddings_cache' not in st.session_state:
        st.session_state.embeddings_cache = {}
    
    # Initialize processing control state
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = 'stopped'
    if 'processing_stop' not in st.session_state:
        st.session_state.processing_stop = False
    if 'processing_pause' not in st.session_state:
        st.session_state.processing_pause = False
    if 'processing_progress' not in st.session_state:
        st.session_state.processing_progress = {
            'current_year_idx': 0,
            'processed_count': 0,
            'chunk_count': 0,
            'skipped_count': 0,
            'chunk_embeddings': [],
            'chunk_bibcodes': [],
            'chunk_texts': [],
            'output_path': None,
            'timestamp': None,
            'subfolder_name': None
        }
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = [
            "hf://nasa-impact/nasa-smd-ibm-st-v2",
            "hf://sentence-transformers/all-MiniLM-L6-v2",
            "hf://sentence-transformers/all-mpnet-base-v2",
            "openai://text-embedding-3-small",
            "openai://text-embedding-3-large",
            "openai://text-embedding-ada-002"
        ]
        
        selected_model = st.selectbox(
            "Embedding Model",
            model_options,
            help="Choose the embedding model for text processing"
        )
        
        # Device selection
        device = st.selectbox(
            "Device",
            ["cuda", "auto", "cpu"],
            help="Processing device for local models"
        )
        
        # API key input for OpenAI models
        api_key = None
        if selected_model.startswith("openai://"):
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Required for OpenAI models"
            )
        
        # Batch size is now automatically optimized based on corpus size selection
        
        # Load existing embeddings
        st.header("📊 Existing Data")
        embeddings_dir = st.text_input(
            "Embeddings Directory",
            value="./embeddings",
            help="Path to existing embeddings for search/comparison",
            key="sidebar_embeddings_dir"
        )
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Embedding Playground",
        "Corpus Processing",
        "Similarity Search", 
        "Model Comparison",
        "Hybrid Search Testing"
    ])
    
    with tab1:
        embedding_playground(selected_model, api_key, device)
    
    with tab2:
        corpus_processing(selected_model, api_key, device)
    
    with tab3:
        similarity_search(embeddings_dir)
    
    with tab4:
        model_comparison(api_key, device)
    
    with tab5:
        hybrid_search_testing(embeddings_dir, selected_model, api_key, device)


def launch_background_job(
    input_dir: str,
    years: str,
    fields: List[str],
    model: str,
    api_key: Optional[str],
    device: str,
    batch_size: int,
    prefix: str,
    suffix: str,
    output_dir: str,
    limit: Optional[int] = None
):
    """Launch background CLI processing job."""
    import subprocess
    import os
    from datetime import datetime
    
    # Create timestamped output directory for background job
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if limit is None:
        subset_desc = "full_corpus"
    else:
        subset_desc = f"{limit}_papers"
    
    bg_output_dir = f"{output_dir}/background_{timestamp}_{subset_desc}"
    
    # Build CLI command
    cmd_parts = [
        "sciembed", "ingest",
        "--input", input_dir,
        "--output", bg_output_dir,
        "--years", years,
        "--fields", ",".join(fields),
        "--model", model,
        "--batch-size", str(batch_size),
        "--device", device
    ]
    
    # Add optional parameters
    if api_key:
        cmd_parts.extend(["--api-key", api_key])
    if prefix:
        cmd_parts.extend(["--prefix", prefix])
    if suffix:
        cmd_parts.extend(["--suffix", suffix])
    
    # Create log file
    log_file = f"{bg_output_dir}_process.log"
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Launch background process
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd_parts,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
        
        # Store job info in session state
        if 'background_jobs' not in st.session_state:
            st.session_state.background_jobs = []
        
        job_info = {
            'pid': process.pid,
            'timestamp': timestamp,
            'output_dir': bg_output_dir,
            'log_file': log_file,
            'command': ' '.join(cmd_parts),
            'status': 'running'
        }
        st.session_state.background_jobs.append(job_info)
        
        st.success(f"🚀 **Background job launched successfully!**")
        st.info(f"**Process ID:** {process.pid}")
        st.info(f"**Output Directory:** {bg_output_dir}")
        st.info(f"**Log File:** {log_file}")
        st.warning("⚠️ **Important:** This job will continue running even if you close the browser or lose connection!")
        
        # Show command that was run
        with st.expander("🔍 Command Details"):
            st.code(' '.join(cmd_parts), language='bash')
        
        # Instructions for monitoring
        st.markdown("### 📊 Monitoring Your Background Job:")
        st.code(f"""
# Check if process is still running:
ps aux | grep {process.pid}

# Monitor log file:
tail -f {log_file}

# Stop the job if needed:
kill {process.pid}
        """, language='bash')
        
    except Exception as e:
        st.error(f"❌ Failed to launch background job: {e}")


def get_optimal_batch_size(sample_size: str, custom_size: Optional[int] = None) -> int:
    """Determine optimal batch size based on corpus size selection."""
    if sample_size == "Full Corpus":
        return 512  # Maximum throughput for full corpus
    elif sample_size == "10000 papers":
        return 256  # High throughput for larger samples
    elif sample_size == "5000 papers":
        return 128  # Moderate throughput
    elif sample_size == "1000 papers":
        return 64   # Conservative for smaller samples
    elif sample_size == "Custom":
        if custom_size:
            if custom_size >= 10000:
                return 256
            elif custom_size >= 5000:
                return 128
            elif custom_size >= 1000:
                return 64
            else:
                return 32  # Very small samples
        return 64  # Default for custom
    else:
        return 64  # Default fallback


def embedding_playground(model: str, api_key: Optional[str], device: str):
    """Interactive embedding generation and visualization."""
    st.header("Embedding Playground")
    st.markdown("Generate embeddings for custom text and visualize the results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input options
        input_method = st.radio(
            "Input Method",
            ["Single Text", "Multiple Texts", "Sample Papers"],
            horizontal=True
        )
        
        texts = []
        if input_method == "Single Text":
            text = st.text_area(
                "Text to Embed",
                placeholder="Enter your text here...",
                height=150,
                key="playground_single_text"
            )
            if text.strip():
                texts = [text.strip()]
        
        elif input_method == "Multiple Texts":
            text_input = st.text_area(
                "Texts to Embed (one per line)",
                placeholder="Text 1\nText 2\nText 3...",
                height=150,
                key="playground_multiple_texts"
            )
            if text_input.strip():
                texts = [line.strip() for line in text_input.split('\n') if line.strip()]
        
        else:  # Sample Papers
            sample_papers = get_sample_papers()
            selected_samples = st.multiselect(
                "Select Sample Papers",
                options=list(sample_papers.keys()),
                default=list(sample_papers.keys())[:3]
            )
            texts = [sample_papers[key] for key in selected_samples]
        
        # Text preprocessing options
        st.subheader("Text Preprocessing")
        col_prefix, col_suffix = st.columns(2)
        
        with col_prefix:
            prefix = st.text_input("Prefix", placeholder="Optional prefix...", key="playground_prefix")
        
        with col_suffix:
            suffix = st.text_input("Suffix", placeholder="Optional suffix...", key="playground_suffix")
        
        # Apply preprocessing
        if texts:
            processed_texts = [f"{prefix}{text}{suffix}" for text in texts]
            
            # Show processed texts
            with st.expander("📝 Processed Texts", expanded=False):
                for i, text in enumerate(processed_texts, 1):
                    st.text_area(f"Text {i}", value=text, height=100, disabled=True)
    
    with col2:
        # Model info
        st.subheader("Model Info")
        playground_batch_size = 32  # Fixed optimal size for playground
        config = EmbedderConfig(
            model=model,
            batch_size=playground_batch_size,
            api_key=api_key,
            device=device
        )
        
        st.info(f"""
        **Model:** {model}
        **Type:** {config.model_type}
        **Batch Size:** {playground_batch_size} (optimized for playground)
        **Device:** {device}
        """)
        
        st.caption("Model will be loaded when embeddings are generated")
    
    # Generate embeddings
    if texts and st.button("Generate Embeddings", type="primary"):
        # Create embedder only when needed
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            status_text.text("Loading model...")
            config = EmbedderConfig(
                model=model,
                batch_size=playground_batch_size,
                api_key=api_key,
                device=device
            )
            embedder = create_embedder(config)
            
            status_text.text(f"Generating embeddings for {len(processed_texts)} texts...")
            embeddings = []
            
            for i in range(0, len(processed_texts), playground_batch_size):
                batch = processed_texts[i:i+playground_batch_size]
                batch_embeddings = embedder.embed_batch(batch)
                embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = (i + len(batch)) / len(processed_texts)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i + len(batch)}/{len(processed_texts)} texts...")
            
            embeddings = np.array(embeddings)
            status_text.text("Embeddings complete!")
            progress_bar.progress(1.0)
            
            # Store in session state
            st.session_state.embeddings_cache[model] = {
                'texts': texts,
                'processed_texts': processed_texts,
                'embeddings': embeddings
            }
            
            # Display results
            display_embedding_results(texts, processed_texts, embeddings)
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")


def corpus_processing(model: str, api_key: Optional[str], device: str):
    """Process SciX corpus and generate embeddings with metadata."""
    st.header("Corpus Processing")
    st.markdown("Process real SciX corpus data and generate embeddings with full metadata tracking")
    
    # Background job monitoring section
    if 'background_jobs' in st.session_state and st.session_state.background_jobs:
        st.subheader("🔍 Background Job Monitor")
        
        for i, job in enumerate(st.session_state.background_jobs):
            with st.expander(f"Job {i+1}: PID {job['pid']} ({job['timestamp']})", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Check if process is still running
                    try:
                        import psutil
                        process = psutil.Process(job['pid'])
                        if process.is_running():
                            st.success(f"🟢 **Status:** Running")
                            try:
                                st.info(f"**CPU:** {process.cpu_percent():.1f}%")
                                st.info(f"**Memory:** {process.memory_info().rss / 1024 / 1024:.1f} MB")
                            except:
                                pass  # Sometimes these fail on first call
                        else:
                            st.error(f"🔴 **Status:** Stopped")
                    except ImportError:
                        # Fallback without psutil
                        import os
                        try:
                            os.kill(job['pid'], 0)  # Test if process exists
                            st.success(f"🟢 **Status:** Running")
                        except OSError:
                            st.error(f"🔴 **Status:** Process not found")
                    except Exception as e:
                        st.warning(f"⚠️ **Status:** Unknown ({e})")
                
                with col2:
                    st.info(f"**Output:** {job['output_dir']}")
                    st.info(f"**Log:** {job['log_file']}")
                    
                    # Show recent log lines
                    if st.button(f"📄 Show Recent Logs", key=f"logs_{job['pid']}"):
                        try:
                            with open(job['log_file'], 'r') as f:
                                lines = f.readlines()
                                recent_lines = lines[-20:] if len(lines) > 20 else lines
                                st.text_area("Recent Log Output", ''.join(recent_lines), height=200, key=f"log_content_{job['pid']}")
                        except Exception as e:
                            st.error(f"Could not read log file: {e}")
                
                with col3:
                    if st.button(f"⏹️ Kill Job", key=f"kill_{job['pid']}", help="Stop this background job"):
                        try:
                            import os
                            os.kill(job['pid'], 9)  # SIGKILL
                            st.success(f"Job {job['pid']} terminated")
                            # Remove from session state
                            st.session_state.background_jobs.remove(job)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to kill job: {e}")
                    
                    st.code(f"tail -f {job['log_file']}", language='bash')
        
        st.markdown("---")
    
    # Configuration section
    st.subheader("Corpus Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Input directory
        input_dir = st.text_input(
            "Corpus Directory",
            value="/home/scixmuse/scix_data/ads_metadata_by_year_full/",
            help="Directory containing yearly JSONL files (e.g., ads_metadata_2020_full.jsonl)",
            key="corpus_input_dir"
        )
        
        # Sample size first to determine if we need year input
        sample_size = st.selectbox(
            "Sample Size",
            ["1000 papers", "5000 papers", "10000 papers", "Full Corpus", "Custom"],
            help="Number of papers to process (Full Corpus = all available papers)"
        )
        
        if sample_size == "Custom":
            custom_size = st.number_input(
                "Custom Sample Size",
                min_value=1,
                max_value=1000000,
                value=1000,
                help="Number of papers to sample"
            )
        else:
            custom_size = None
        
        # Determine and display optimal batch size
        optimal_batch_size = get_optimal_batch_size(sample_size, custom_size)
        st.info(f"🔧 **Optimal Batch Size:** {optimal_batch_size} (auto-selected for {sample_size.lower()})")
        
        # Processing mode selection
        processing_mode = st.radio(
            "Processing Mode",
            ["Interactive (in browser)", "Background (persistent)"],
            help="Interactive: runs in browser (stops if connection lost). Background: runs as persistent background job."
        )
        
        # Year selection (disabled for full corpus)
        if sample_size == "Full Corpus":
            st.info("🚀 Full Corpus mode: Will process ALL available papers with incremental saving every 10,000 papers")
            st.warning("⚡ For high-performance processing: Use batch size 256+, ensure adequate GPU memory")
            st.info("⏸️ Use Pause/Stop controls during processing to manage long-running tasks")
            year_input = st.text_input(
                "Years to Process",
                value="Auto-discover all years",
                help="Full corpus mode will find and process all available year files",
                key="corpus_years",
                disabled=True
            )
        else:
            year_input = st.text_input(
                "Years to Process",
                value="2023",
                help="Single year (2023), range (2020:2023), or list (2020,2021,2023)",
                key="corpus_years"
            )
    
    with col2:
        # Field selection
        field_config = st.radio(
            "Text Fields to Embed",
            ["Title + Abstract", "Abstract Only", "Title Only", "Custom Fields"],
            help="Which fields to use for embedding generation"
        )
        
        if field_config == "Custom Fields":
            custom_fields = st.multiselect(
                "Select Fields",
                ["title", "abstract", "keywords", "author", "bibcode"],
                default=["title", "abstract"],
                help="Choose which fields to combine for embeddings"
            )
        else:
            field_mapping = {
                "Title + Abstract": ["title", "abstract"],
                "Abstract Only": ["abstract"],
                "Title Only": ["title"]
            }
            custom_fields = field_mapping[field_config]
        
        # Text preprocessing
        st.markdown("**Text Preprocessing**")
        prefix = st.text_input("Prefix", placeholder="Optional prefix...", key="corpus_prefix")
        suffix = st.text_input("Suffix", placeholder="Optional suffix...", key="corpus_suffix")
        delimiter = st.selectbox("Field Delimiter", ["\n\n", " | ", " ", "\n"], index=0, key="corpus_delimiter")
        truncate_length = st.number_input("Max Text Length", min_value=100, max_value=32000, value=8192, key="corpus_truncate")
    
    # Output configuration
    st.subheader("Output Configuration")
    
    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        output_dir = st.text_input(
            "Output Directory",
            value="./embeddings_dashboard_output",
            help="Directory to save embeddings and metadata",
            key="corpus_output_dir"
        )
        
        export_format = st.selectbox(
            "Export Format",
            ["JSON + Metadata", "CSV + Embeddings", "Solr-ready Format"],
            help="Format for output files",
            key="corpus_export_format"
        )
    
    with col_out2:
        include_metadata = st.multiselect(
            "Include Metadata",
            ["Bibcodes", "Field Configuration", "Model Info", "Processing Stats", "Raw Paper Data"],
            default=["Bibcodes", "Field Configuration", "Model Info"],
            help="What metadata to include in output",
            key="corpus_metadata"
        )
    
    # Preview section
    if st.button("Preview Data", help="Load a small sample to preview"):
        # Determine preview size based on subset selection
        if sample_size == "Full Corpus":
            preview_limit = 10  # Reasonable default for full corpus preview
            available_years = discover_available_years(input_dir)
            if available_years:
                st.info(f"Full Corpus mode: Found {len(available_years)} years ({min(available_years)}-{max(available_years)})")
                preview_corpus_data(input_dir, str(available_years[0]), custom_fields, preview_limit)
            else:
                st.error(f"No year files found in {input_dir}")
        else:
            # Use subset size for preview, but cap at 20 for reasonable preview
            if sample_size == "Custom":
                preview_limit = min(custom_size, 20)
            else:
                requested_size = int(sample_size.split()[0])
                preview_limit = min(requested_size, 20)
            
            preview_corpus_data(input_dir, year_input, custom_fields, preview_limit)
    
    # Show current progress status if any
    has_saved_progress = (st.session_state.processing_progress['processed_count'] > 0 or 
                         len(st.session_state.processing_progress['chunk_embeddings']) > 0)
    
    if has_saved_progress:
        if st.session_state.processing_state == 'paused':
            st.warning(f"⏸️ **Processing Paused** - {st.session_state.processing_progress['processed_count']:,} papers processed. Use Resume button below or reset to start over.")
        else:
            st.info(f"📊 **Previous Progress Found** - {st.session_state.processing_progress['processed_count']:,} papers processed. Will continue from where left off.")
    
    # Main processing button - changes based on state  
    if processing_mode == "Background (persistent)":
        button_text = "🚀 Launch Background Job"
        button_type = "primary"
        button_help = "Launches persistent background processing that survives browser disconnection"
    elif has_saved_progress and st.session_state.processing_state == 'paused':
        button_text = "▶️ Resume Processing"
        button_type = "secondary"
        button_help = "Resume paused interactive processing"
    else:
        button_text = "🚀 Process Corpus" 
        button_type = "primary"
        button_help = "Start interactive processing in browser"
    
    if st.button(button_text, type=button_type, help=button_help):
        if not Path(input_dir).exists():
            st.error(f"Input directory not found: {input_dir}")
            return
        
        # Determine years and sample size
        if sample_size == "Full Corpus":
            available_years = discover_available_years(input_dir)
            if not available_years:
                st.error(f"No year files found in {input_dir}")
                return
            years_to_process = ",".join(map(str, available_years))
            limit = None
            st.info(f"Processing Full Corpus: {len(available_years)} years ({min(available_years)}-{max(available_years)})")
        else:
            years_to_process = year_input
            if sample_size == "Custom":
                limit = custom_size
            else:
                limit = int(sample_size.split()[0])
        
        if processing_mode == "Background (persistent)":
            # Launch background CLI job
            launch_background_job(
                input_dir=input_dir,
                years=years_to_process,
                fields=custom_fields,
                model=model,
                api_key=api_key,
                device=device,
                batch_size=optimal_batch_size,
                prefix=prefix,
                suffix=suffix,
                output_dir=output_dir,
                limit=limit
            )
        else:
            # Run interactive processing
            process_corpus_data(
                input_dir=input_dir,
                years=years_to_process,
                fields=custom_fields,
                model=model,
                api_key=api_key,
                device=device,
                batch_size=optimal_batch_size,
                prefix=prefix,
                suffix=suffix,
                delimiter=delimiter,
                truncate_length=truncate_length,
                output_dir=output_dir,
                export_format=export_format,
                include_metadata=include_metadata,
                limit=limit
            )


def preview_corpus_data(input_dir: str, years: str, fields: List[str], limit: int = 5):
    """Preview a small sample of corpus data."""
    try:
        # Parse years
        year_list = parse_years(years)
        
        st.subheader("Data Preview")
        
        # Show available files first
        input_path = Path(input_dir)
        if not input_path.exists():
            st.error(f"Directory does not exist: {input_dir}")
            return
        
        # Look for ADS files first, then any JSON files
        ads_files = list(input_path.glob("ads_metadata_*_full.jsonl"))
        if ads_files:
            st.info(f"Found {len(ads_files)} SciX metadata files:")
            for file_path in sorted(ads_files)[:10]:  # Show first 10 files
                st.text(f"  • {file_path.name}")
            if len(ads_files) > 10:
                st.text(f"  ... and {len(ads_files) - 10} more")
        else:
            available_files = list(input_path.glob("*.json*"))
            if available_files:
                st.info(f"📁 Found {len(available_files)} JSON files:")
                for file_path in sorted(available_files)[:10]:  # Show first 10 files
                    st.text(f"  • {file_path.name}")
                if len(available_files) > 10:
                    st.text(f"  ... and {len(available_files) - 10} more")
            else:
                st.error(f"No JSON/JSONL files found in {input_dir}")
                return
        
        for year in year_list[:1]:  # Just show first year for preview
            year_file = find_year_file(input_dir, year)
            
            if not year_file:
                st.warning(f"No data file found for year {year}. Looking for: ads_metadata_{year}_full.jsonl, {year}.json, {year}.jsonl, {year}.json.gz, {year}.jsonl.gz")
                continue
            
            # Load sample records
            loader = JSONLoader(show_progress=False)
            records = []
            
            for i, record in enumerate(loader.load(year_file, fields + ["bibcode"])):
                if i >= limit:
                    break
                records.append(record)
            
            if records:
                st.success(f"Found {len(records)} sample records from {year}")
                
                # Show sample records
                for i, record in enumerate(records):
                    with st.expander(f"Sample {i+1}: {record.get('bibcode', 'No bibcode')}"):
                        for field in fields:
                            if field in record:
                                value = str(record[field])
                                if len(value) > 200:
                                    value = value[:200] + "..."
                                st.text_area(f"{field.title()}", value, height=100, disabled=True)
            else:
                st.warning(f"No records found in {year_file}")
                
    except Exception as e:
        st.error(f"Error previewing data: {e}")


def process_corpus_data(
    input_dir: str,
    years: str,
    fields: List[str],
    model: str,
    api_key: Optional[str],
    device: str,
    batch_size: int,
    prefix: str,
    suffix: str,
    delimiter: str,
    truncate_length: int,
    output_dir: str,
    export_format: str,
    include_metadata: List[str],
    limit: Optional[int] = None
):
    """Process corpus data and generate embeddings with metadata, including pause/stop and incremental saving."""
    
    # Session state is already initialized in main()
    
    # Create or reuse timestamped subfolder with subset info
    from datetime import datetime
    
    # Check if we're resuming and already have an output path
    if (st.session_state.processing_progress['output_path'] and 
        Path(st.session_state.processing_progress['output_path']).exists()):
        # Resuming - reuse existing path
        output_path = Path(st.session_state.processing_progress['output_path'])
        timestamp = st.session_state.processing_progress['timestamp']
        subfolder_name = st.session_state.processing_progress['subfolder_name']
        st.success(f"📁 Resuming with existing output folder: {subfolder_name}")
    else:
        # Starting fresh - create new timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine subset description for folder name
        if limit is None:
            subset_desc = "full_corpus"
        else:
            subset_desc = f"{limit}_papers"
        
        # Create subfolder: output_dir/YYYYMMDD_HHMMSS_subset_desc/
        subfolder_name = f"{timestamp}_{subset_desc}"
        output_path = Path(output_dir) / subfolder_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save to session state for future resume
        st.session_state.processing_progress.update({
            'output_path': str(output_path),
            'timestamp': timestamp,
            'subfolder_name': subfolder_name
        })
        
        st.info(f"📁 Created new output folder: {subfolder_name}")
    
    # Initialize skipped bibcodes tracking
    skipped_bibcodes_file = output_path / "skipped_bibcodes.txt"
    
    # Parse years
    year_list = parse_years(years)
    
    # Control buttons - always available during processing
    control_container = st.container()
    with control_container:
        st.markdown("**🎛️ Processing Controls** (Available during processing)")
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)
        
        with control_col1:
            if st.button("⏸️ Pause", key="pause_btn", help="Pause processing after current batch"):
                st.session_state.processing_pause = True
                st.session_state.processing_state = 'paused'
                st.warning("⏸️ Processing will pause after current batch...")
        
        with control_col2:
            if st.button("▶️ Resume", key="resume_btn", help="Resume paused processing"):
                st.session_state.processing_pause = False
                st.session_state.processing_state = 'running'
                st.info("▶️ Resuming processing...")
        
        with control_col3:
            if st.button("⏹️ Stop", key="stop_btn", help="Stop processing and save current progress"):
                st.session_state.processing_stop = True
                st.session_state.processing_state = 'stopping'
                st.error("🛑 Processing will stop after current batch...")
        
        with control_col4:
            if st.button("🔄 Reset", key="reset_btn", help="Reset all processing state"):
                st.session_state.processing_stop = False
                st.session_state.processing_pause = False
                st.session_state.processing_state = 'stopped'
                st.session_state.processing_progress = {
                    'current_year_idx': 0,
                    'processed_count': 0,
                    'chunk_count': 0,
                    'skipped_count': 0,
                    'chunk_embeddings': [],
                    'chunk_bibcodes': [],
                    'chunk_texts': [],
                    'output_path': None,
                    'timestamp': None,
                    'subfolder_name': None
                }
                st.success("🔄 All controls reset")
    
    # Status display
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.processing_state == 'running':
            st.success("🟢 **Status:** Processing Active")
        elif st.session_state.processing_state == 'paused':
            st.warning("🟡 **Status:** Paused (click Resume to continue)")
        elif st.session_state.processing_state == 'stopping':
            st.error("🔴 **Status:** Stopping...")
        else:
            st.info("⚪ **Status:** Ready to Process")
    
    with status_col2:
        if st.session_state.processing_progress['processed_count'] > 0:
            st.info(f"📊 **Progress:** {st.session_state.processing_progress['processed_count']:,} papers processed")
    
    # Create embedder
    model_progress = st.progress(0)
    model_status = st.empty()
    
    try:
        model_status.text("Loading embedding model...")
        model_progress.progress(0.3)
        
        config = EmbedderConfig(
            model=model,
            batch_size=batch_size,
            api_key=api_key,
            device=device
        )
        embedder = create_embedder(config)
        
        model_progress.progress(1.0)
        model_status.text(f"Model loaded: {embedder.name} (dim: {embedder.dim})")
        
    except Exception as e:
        model_status.text("")
        model_progress.empty()
        st.error(f"Error loading model: {e}")
        return
    
    # Create preparer
    preparer_config = PreparerConfig(
        fields=fields,
        prefix=prefix,
        suffix=suffix,
        delimiter=delimiter,
        lowercase=True,
        truncate=truncate_length
    )
    preparer = Preparer(preparer_config)
    
    # Initialize tracking - restore from saved state if resuming
    has_saved_progress = (st.session_state.processing_progress['processed_count'] > 0 or 
                         len(st.session_state.processing_progress['chunk_embeddings']) > 0)
    
    if has_saved_progress and st.session_state.processing_state != 'stopped':
        # Resuming from pause - restore state
        processed_count = st.session_state.processing_progress['processed_count']
        chunk_count = st.session_state.processing_progress['chunk_count']
        skipped_count = st.session_state.processing_progress['skipped_count']
        chunk_embeddings = st.session_state.processing_progress['chunk_embeddings']
        chunk_bibcodes = st.session_state.processing_progress['chunk_bibcodes']
        chunk_texts = st.session_state.processing_progress['chunk_texts']
        start_year_idx = st.session_state.processing_progress['current_year_idx']
        st.success(f"📈 Resuming from: {processed_count:,} papers processed, starting from year index {start_year_idx}")
    else:
        # Starting fresh
        processed_count = 0
        chunk_count = 0
        skipped_count = 0
        chunk_embeddings = []
        chunk_bibcodes = []
        chunk_texts = []
        start_year_idx = 0
        # Reset progress tracking
        st.session_state.processing_progress = {
            'current_year_idx': 0,
            'processed_count': 0,
            'chunk_count': 0,
            'skipped_count': 0,
            'chunk_embeddings': [],
            'chunk_bibcodes': [],
            'chunk_texts': [],
            'output_path': None,
            'timestamp': None,
            'subfolder_name': None
        }
        st.info("🚀 Starting fresh processing")
    
    CHUNK_SIZE = 10000  # Save every 10k papers
    
    # Progress containers
    progress_container = st.container()
    status_container = st.container()
    
    with progress_container:
        year_progress = st.progress(0)
        batch_progress = st.progress(0)
        overall_status = st.empty()
        timing_status = st.empty()
        control_status = st.empty()
    
    import time
    start_time = time.time()
    
    # Set processing state
    st.session_state.processing_state = 'running'
    
    for year_idx, year in enumerate(year_list):
        # Skip years we've already processed if resuming
        if year_idx < start_year_idx:
            continue
            
        if st.session_state.processing_stop:
            control_status.warning("Processing stopped by user")
            break
            
        # Handle pause
        if st.session_state.processing_pause:
            # Save current state
            st.session_state.processing_progress.update({
                'current_year_idx': year_idx,
                'processed_count': processed_count,
                'chunk_count': chunk_count,
                'skipped_count': skipped_count,
                'chunk_embeddings': chunk_embeddings,
                'chunk_bibcodes': chunk_bibcodes,
                'chunk_texts': chunk_texts
            })
            control_status.warning("⏸️ Processing paused. Click Resume to continue from this point.")
            return
            
        with status_container:
            st.info(f"Processing year {year}... (Year {year_idx + 1}/{len(year_list)})")
        
        # Find year file
        year_file = find_year_file(input_dir, year)
        if not year_file:
            st.warning(f"No data file found for year {year}")
            continue
        
        try:
            # Load and process data
            loader = JSONLoader(show_progress=False)
            
            batch_texts = []
            batch_bibcodes = []
            
            for record in loader.load(year_file, fields + ["bibcode"]):
                # Check for stop/pause - more responsive control
                if st.session_state.processing_stop:
                    control_status.warning("🛑 Stopping processing...")
                    break
                    
                if st.session_state.processing_pause:
                    # Save current state and pause
                    st.session_state.processing_progress.update({
                        'current_year_idx': year_idx,
                        'processed_count': processed_count,
                        'chunk_count': chunk_count,
                        'skipped_count': skipped_count,
                        'chunk_embeddings': chunk_embeddings,
                        'chunk_bibcodes': chunk_bibcodes,
                        'chunk_texts': chunk_texts
                    })
                    control_status.warning("⏸️ Processing paused. Click Resume to continue.")
                    return
                
                if limit and processed_count >= limit:
                    break
                
                try:
                    bibcode, text = preparer.prepare_record(record)
                    batch_texts.append(text)
                    batch_bibcodes.append(bibcode)
                    
                    # Process batch when full OR when we've reached the limit
                    should_process_batch = (
                        len(batch_texts) >= batch_size or 
                        (limit and len(batch_texts) + processed_count >= limit)
                    )
                    
                    if should_process_batch:
                        # If we have a limit, only process up to the limit
                        if limit and processed_count + len(batch_texts) > limit:
                            # Truncate batch to exactly reach the limit
                            remaining = limit - processed_count
                            batch_texts = batch_texts[:remaining]
                            batch_bibcodes = batch_bibcodes[:remaining]
                        
                        if batch_texts:  # Only process if we have texts
                            batch_start = time.time()
                            embeddings = embedder.embed_batch(batch_texts)
                            batch_time = time.time() - batch_start
                            
                            # Add to current chunk
                            chunk_embeddings.extend(embeddings)
                            chunk_bibcodes.extend(batch_bibcodes)
                            chunk_texts.extend(batch_texts)
                            processed_count += len(batch_texts)
                            
                            # Calculate performance metrics
                            elapsed_time = time.time() - start_time
                            papers_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                            
                            # Update progress
                            overall_status.text(f"Processed {processed_count:,} papers | {papers_per_second:.1f} papers/sec | Skipped: {skipped_count}")
                            timing_status.text(f"Batch: {batch_time:.2f}s | Total: {elapsed_time:.1f}s | ETA: {((limit or 25000000) - processed_count) / papers_per_second / 3600:.1f}h" if papers_per_second > 0 else "Calculating...")
                            
                            # Update session state progress for real-time tracking
                            st.session_state.processing_progress.update({
                                'processed_count': processed_count,
                                'skipped_count': skipped_count
                            })
                            
                            # Update batch progress
                            if limit:
                                batch_progress.progress(min(processed_count / limit, 1.0))
                            
                            # Check if we need to save chunk (every 10k papers)
                            if len(chunk_embeddings) >= CHUNK_SIZE:
                                # Determine subset description for metadata
                                if limit is None:
                                    subset_desc = "full_corpus"
                                else:
                                    subset_desc = f"{limit}_papers"
                                    
                                save_chunk(output_path, chunk_embeddings, chunk_bibcodes, chunk_texts, 
                                          chunk_count, export_format, include_metadata, embedder, config, 
                                          fields, prefix, suffix, delimiter, truncate_length, year_list, timestamp, subset_desc, limit)
                                chunk_count += 1
                                
                                # Update session state with new chunk count
                                st.session_state.processing_progress['chunk_count'] = chunk_count
                                
                                # Clear chunk data
                                chunk_embeddings = []
                                chunk_bibcodes = []
                                chunk_texts = []
                                
                                st.info(f"💾 Saved chunk {chunk_count} ({CHUNK_SIZE:,} papers)")
                            
                            # Clear batch
                            batch_texts = []
                            batch_bibcodes = []
                            
                            # Break if we've reached the limit
                            if limit and processed_count >= limit:
                                break
                
                except Exception as e:
                    # Log skipped bibcode instead of showing warning
                    skipped_bibcode = record.get('bibcode', f'unknown_{processed_count}')
                    with open(skipped_bibcodes_file, 'a') as f:
                        f.write(f"{skipped_bibcode}\t{str(e)}\n")
                    skipped_count += 1
                    continue
            
            # Process remaining batch
            if batch_texts and not st.session_state.processing_stop:
                embeddings = embedder.embed_batch(batch_texts)
                chunk_embeddings.extend(embeddings)
                chunk_bibcodes.extend(batch_bibcodes)
                chunk_texts.extend(batch_texts)
                processed_count += len(batch_texts)
        
        except Exception as e:
            st.error(f"Error processing year {year}: {e}")
            continue
        
        # Update year progress
        year_progress.progress((year_idx + 1) / len(year_list))
        
        if st.session_state.processing_stop:
            break
    
    # Save final chunk if there's remaining data
    if chunk_embeddings:
        save_chunk(output_path, chunk_embeddings, chunk_bibcodes, chunk_texts, 
                  chunk_count, export_format, include_metadata, embedder, config, 
                  fields, prefix, suffix, delimiter, truncate_length, year_list, timestamp, subset_desc, limit)
    
    # Create a comprehensive summary file
    create_processing_summary(output_path, processed_count, skipped_count, chunk_count, 
                             optimal_batch_size, fields, embedder, config, 
                             timestamp, subset_desc, year_list, export_format)
    
    # Reset processing state
    st.session_state.processing_state = 'stopped'
    st.session_state.processing_stop = False
    st.session_state.processing_pause = False
    # Clear progress tracking since processing is complete
    st.session_state.processing_progress = {
        'current_year_idx': 0,
        'processed_count': 0,
        'chunk_count': 0,
        'skipped_count': 0,
        'chunk_embeddings': [],
        'chunk_bibcodes': [],
        'chunk_texts': [],
        'output_path': None,
        'timestamp': None,
        'subfolder_name': None
    }
    
    if processed_count == 0:
        st.error("No embeddings were generated")
        return
    
    # Display success message
    status_msg = "Processing complete!" if not st.session_state.processing_stop else "Processing stopped by user"
    st.success(status_msg)
    st.info(f"""
    **Summary:**
    - Papers processed: {processed_count:,}
    - Papers skipped: {skipped_count}
    - Chunks saved: {chunk_count + (1 if chunk_embeddings else 0)}
    - Subset: {subset_desc.replace('_', ' ').title()}
    - Output folder: {subfolder_name}
    - Full path: {output_path}
    """)
    
    # Show download links
    st.subheader("Download Results")
    
    # List output files
    output_files = list(output_path.glob("*"))
    for file_path in sorted(output_files):
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                st.download_button(
                    label=f"Download {file_path.name}",
                    data=f.read(),
                    file_name=file_path.name,
                    mime="application/octet-stream"
                )


def save_chunk(output_path: Path, embeddings: list, bibcodes: list, texts: list, 
               chunk_num: int, export_format: str, include_metadata: list, 
               embedder, config, fields: list, prefix: str, suffix: str, 
               delimiter: str, truncate_length: int, year_list: list, 
               timestamp: str, subset_desc: str, limit: Optional[int]):
    """Save a chunk of embeddings incrementally."""
    
    embeddings_array = np.array(embeddings)
    
    # Generate metadata for this chunk
    import time
    processing_time = time.time()
    
    metadata = {
        "processing_info": {
            "timestamp": timestamp,
            "processing_time_unix": processing_time,
            "processing_time_iso": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(processing_time)),
            "chunk_number": chunk_num,
            "subset_description": subset_desc,
            "subset_limit": limit,
            "embedding_dimension": embeddings_array.shape[1],
            "papers_in_chunk": len(embeddings),
            "fields_used": fields,
            "field_combination": f"{'|'.join(fields)}",
            "preprocessing": {
                "prefix": prefix,
                "suffix": suffix,
                "delimiter": delimiter,
                "truncate_length": truncate_length
            },
            "years_processed": year_list
        },
        "model_info": {
            "model_name": embedder.name,
            "model_type": config.model_type,
            "embedding_dim": embedder.dim,
            "batch_size": config.batch_size,
            "device": config.device
        },
        "data_mapping": {
            "bibcode_to_index": {bibcode: idx for idx, bibcode in enumerate(bibcodes)},
            "total_papers_in_chunk": len(bibcodes),
            "first_bibcode": bibcodes[0] if bibcodes else None,
            "last_bibcode": bibcodes[-1] if bibcodes else None
        }
    }
    
    # Create chunk-specific filename
    chunk_suffix = f"_chunk_{chunk_num:04d}"
    
    # Save based on format
    if export_format == "JSON + Metadata":
        save_json_format_chunk(output_path, embeddings_array, bibcodes, texts, metadata, include_metadata, chunk_suffix)
    elif export_format == "CSV + Embeddings":
        save_csv_format_chunk(output_path, embeddings_array, bibcodes, texts, metadata, include_metadata, chunk_suffix)
    else:  # Solr-ready Format
        save_solr_format_chunk(output_path, embeddings_array, bibcodes, texts, metadata, include_metadata, chunk_suffix)


def save_json_format_chunk(output_path: Path, embeddings: np.ndarray, bibcodes: List[str], 
                          texts: List[str], metadata: Dict, include_metadata: List[str], chunk_suffix: str):
    """Save chunk in JSON format."""
    data = {
        "embeddings": embeddings.tolist(),
        "bibcodes": bibcodes,
        "metadata": metadata
    }
    
    if "Raw Paper Data" in include_metadata:
        data["texts"] = texts
    
    with open(output_path / f"embeddings{chunk_suffix}.json", "w") as f:
        json.dump(data, f, indent=2)


def save_csv_format_chunk(output_path: Path, embeddings: np.ndarray, bibcodes: List[str],
                         texts: List[str], metadata: Dict, include_metadata: List[str], chunk_suffix: str):
    """Save chunk in CSV format."""
    import pandas as pd
    df_data = {"bibcode": bibcodes}
    
    # Add embedding dimensions as columns
    for i in range(embeddings.shape[1]):
        df_data[f"dim_{i}"] = embeddings[:, i]
    
    if "Raw Paper Data" in include_metadata:
        df_data["text"] = texts
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path / f"embeddings{chunk_suffix}.csv", index=False)


def save_solr_format_chunk(output_path: Path, embeddings: np.ndarray, bibcodes: List[str],
                          texts: List[str], metadata: Dict, include_metadata: List[str], chunk_suffix: str):
    """Save chunk in Solr-ready format."""
    solr_docs = []
    
    for i, (bibcode, embedding) in enumerate(zip(bibcodes, embeddings)):
        doc = {
            "bibcode": bibcode,
            "embedding_vector": embedding.tolist(),
            "embedding_model": metadata["model_info"]["model_name"],
            "embedding_dim": len(embedding),
            "fields_used": metadata["processing_info"]["fields_used"]
        }
        
        if "Raw Paper Data" in include_metadata:
            doc["embedded_text"] = texts[i]
        
        solr_docs.append(doc)
    
    # Save as JSONL for Solr ingestion
    with open(output_path / f"solr_embeddings{chunk_suffix}.jsonl", "w") as f:
        for doc in solr_docs:
            f.write(json.dumps(doc) + "\n")


def create_processing_summary(output_path: Path, processed_count: int, skipped_count: int, 
                             chunk_count: int, batch_size: int, fields: List[str], 
                             embedder, config, timestamp: str, subset_desc: str, 
                             year_list: List[int], export_format: str):
    """Create a comprehensive summary of the processing run."""
    import time
    
    summary = {
        "processing_summary": {
            "run_id": timestamp,
            "completion_time": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            "total_papers_processed": processed_count,
            "total_papers_skipped": skipped_count,
            "success_rate": f"{processed_count / (processed_count + skipped_count) * 100:.2f}%" if (processed_count + skipped_count) > 0 else "N/A",
            "chunks_created": chunk_count + (1 if chunk_count == 0 else 0),
            "subset_description": subset_desc
        },
        "model_configuration": {
            "model_name": embedder.name,
            "model_type": config.model_type,
            "embedding_dimension": embedder.dim,
            "batch_size": batch_size,
            "device": config.device
        },
        "text_processing": {
            "fields_used": fields,
            "field_combination": " + ".join(fields),
            "preprocessing_applied": True
        },
        "data_sources": {
            "years_processed": year_list,
            "year_range": f"{min(year_list)}-{max(year_list)}" if len(year_list) > 1 else str(year_list[0]) if year_list else "None"
        },
        "output_format": {
            "export_format": export_format,
            "files_created": f"Multiple chunk files in {export_format.lower().replace(' + ', '_').replace(' ', '_')}_format",
            "bibcode_mapping": "Included in each file",
            "metadata_included": "Full metadata in each chunk"
        },
        "file_description": {
            "embeddings_chunks": f"embeddings_chunk_XXXX.{export_format.split()[0].lower()}",
            "skipped_papers": "skipped_bibcodes.txt (if any)",
            "this_summary": "processing_summary.json"
        }
    }
    
    # Save summary
    with open(output_path / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def similarity_search(embeddings_dir: str):
    """Search for similar papers in existing embeddings."""
    st.header("Similarity Search")
    st.markdown("Search for similar papers using your generated embeddings")
    
    # Lazy import heavy dependencies
    try:
        from sciembed.components.index import Index
        from sciembed.pipeline import Pipeline
    except ImportError as e:
        st.error(f"Import error: {e}")
        return
    
    embeddings_path = Path(embeddings_dir)
    
    if not embeddings_path.exists():
        st.warning(f"Embeddings directory not found: {embeddings_dir}")
        st.info("Run the ingest command first or specify a valid embeddings directory")
        return
    
    # Load index
    index_path = embeddings_path / "index.db"
    if not index_path.exists():
        st.warning("No index found in embeddings directory")
        return
    
    try:
        index = Index(index_path)
        stats = index.get_stats()
        years = index.list_years()
        models = index.list_models()
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Embeddings", f"{stats['total_embeddings']:,}")
        with col2:
            st.metric("Unique Papers", f"{stats['unique_bibcodes']:,}")
        with col3:
            st.metric("Years Available", len(years))
        with col4:
            st.metric("Models Available", len(models))
        
        # Search interface
        col_search, col_params = st.columns([2, 1])
        
        with col_search:
            query = st.text_area(
                "Search Query",
                placeholder="Enter your search query...",
                height=100,
                key="similarity_search_query"
            )
        
        with col_params:
            selected_year = st.selectbox("Year", years, key="similarity_year")
            selected_model = st.selectbox("Model", models, key="similarity_model") if models else st.selectbox("Model", ["No models found"], key="similarity_model_none")
            num_results = st.slider("Number of Results", 1, 50, 10, key="similarity_num_results")
        
        if query and st.button("Search", type="primary"):
            search_and_display_results(index, query, selected_year, selected_model, num_results)
    
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")


def model_comparison(api_key: Optional[str], device: str):
    """Compare different embedding models side by side."""
    st.header("Model Comparison")
    st.markdown("Compare how different models embed the same text")
    
    # Model selection for comparison
    available_models = [
        "hf://nasa-impact/nasa-smd-ibm-st-v2",
        "hf://sentence-transformers/all-MiniLM-L6-v2",
        "hf://sentence-transformers/all-mpnet-base-v2",
    ]
    
    if api_key:
        available_models.extend([
            "openai://text-embedding-3-small",
            "openai://text-embedding-3-large"
        ])
    
    selected_models = st.multiselect(
        "Models to Compare",
        available_models,
        default=available_models[:2]
    )
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models for comparison")
        return
    
    # Test text input
    test_text = st.text_area(
        "Test Text",
        value="We present observations of the cosmic microwave background radiation using the Planck satellite telescope. Our analysis reveals new insights into the early universe and dark matter distribution.",
        height=100,
        key="comparison_test_text"
    )
    
    if test_text and st.button("Compare Models", type="primary"):
        comparison_batch_size = 32  # Fixed optimal size for comparison
        compare_models(selected_models, test_text, api_key, device, comparison_batch_size)


def hybrid_search_testing(embeddings_dir: str, model: str, api_key: Optional[str], device: str):
    """Test hybrid search combining semantic and keyword search."""
    st.header("Hybrid Search Testing")
    st.markdown("Experiment with combining semantic embeddings and keyword search for Solr integration")
    
    # Mock Solr fields for testing
    st.subheader("Search Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Semantic Search Weight**")
        semantic_weight = st.slider("Vector similarity weight", 0.0, 1.0, 0.7, 0.1)
        
        st.markdown("**Vector Search Fields**")
        vector_fields = st.multiselect(
            "Fields to use for semantic search",
            ["title_vector", "abstract_vector", "full_text_vector"],
            default=["title_vector", "abstract_vector"]
        )
    
    with col2:
        st.markdown("**Keyword Search Weight**")
        keyword_weight = st.slider("BM25 keyword weight", 0.0, 1.0, 0.3, 0.1)
        
        st.markdown("**Keyword Search Fields**")
        keyword_fields = st.multiselect(
            "Fields to use for keyword search",
            ["title", "abstract", "keywords", "author"],
            default=["title", "abstract"]
        )
    
    # Normalize weights
    total_weight = semantic_weight + keyword_weight
    if total_weight > 0:
        semantic_weight = semantic_weight / total_weight
        keyword_weight = keyword_weight / total_weight
    
    # Query input
    query = st.text_area(
        "Search Query",
        placeholder="Enter your search query...",
        value="exoplanet detection using machine learning",
        key="hybrid_search_query"
    )
    
    if query and st.button("Test Hybrid Search", type="primary"):
        test_hybrid_search(query, semantic_weight, keyword_weight, vector_fields, keyword_fields, model, api_key, device)


def display_embedding_results(texts: List[str], processed_texts: List[str], embeddings: np.ndarray):
    """Display embedding generation results with visualizations."""
    st.success(f"Generated {len(embeddings)} embeddings")
    
    # Basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Embedding Dimension", embeddings.shape[1])
    with col2:
        st.metric("Average Norm", f"{np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")
    with col3:
        st.metric("Standard Deviation", f"{np.std(embeddings):.4f}")
    
    # Similarity matrix if multiple texts
    if len(embeddings) > 1:
        st.subheader("Similarity Matrix")
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1)
        similarity_matrix = similarity_matrix / np.outer(norms, norms)
        
        # Create heatmap
        try:
            import plotly.express as px
        except ImportError:
            st.error("plotly not installed. Run: pip install plotly")
            return
            
        fig = px.imshow(
            similarity_matrix,
            title="Cosine Similarity Between Texts",
            color_continuous_scale="viridis",
            aspect="auto"
        )
        fig.update_layout(
            xaxis_title="Text Index",
            yaxis_title="Text Index"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Dimensionality reduction visualization
    if len(embeddings) > 2:
        st.subheader("Embedding Visualization (2D Projection)")
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
        except ImportError:
            st.error("scikit-learn not installed. Run: pip install scikit-learn")
            return
        
        # Choose reduction method
        reduction_method = st.radio(
            "Dimensionality Reduction",
            ["PCA", "t-SNE"],
            horizontal=True
        )
        
        if reduction_method == "PCA":
            reducer = PCA(n_components=2)
            embedding_2d = reducer.fit_transform(embeddings)
            explained_variance = reducer.explained_variance_ratio_.sum()
            st.info(f"PCA explains {explained_variance:.1%} of variance")
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embedding_2d = reducer.fit_transform(embeddings)
        
        # Create scatter plot
        try:
            import plotly.graph_objects as go
        except ImportError:
            st.error("plotly not installed. Run: pip install plotly")
            return
            
        fig = go.Figure()
        
        for i, (x, y) in enumerate(embedding_2d):
            # Truncate text for hover display
            hover_text = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[f"Text {i+1}"],
                textposition="middle center",
                hovertext=hover_text,
                hoverinfo="text",
                marker=dict(size=10),
                name=f"Text {i+1}"
            ))
        
        fig.update_layout(
            title=f"Embedding Visualization ({reduction_method})",
            xaxis_title=f"{reduction_method} Component 1",
            yaxis_title=f"{reduction_method} Component 2",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Raw embeddings download
    if st.button("Download Embeddings"):
        # Prepare data for download
        embedding_data = {
            'texts': texts,
            'processed_texts': processed_texts,
            'embeddings': embeddings.tolist()
        }
        
        json_str = json.dumps(embedding_data, indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name="embeddings.json",
            mime="application/json"
        )


def search_and_display_results(index, query: str, year: int, model: str, num_results: int):
    """Perform search and display results."""
    try:
        from sciembed.pipeline import Pipeline
        
        # Create temporary config for pipeline
        config = Config(
            input_dir=".",  # Not used for search
            output_dir=index.db_path.parent,
            years=[year],
            model=model
        )
        
        pipeline = Pipeline(config)
        results = pipeline.search_similar(query, year, num_results)
        
        if not results:
            st.warning("No results found")
            return
        
        st.success(f"Found {len(results)} results")
        
        # Display results
        for i, (bibcode, score) in enumerate(results, 1):
            with st.expander(f"📄 Result {i}: {bibcode} (similarity: {score:.4f})"):
                # Try to load paper metadata if available
                st.code(f"Bibcode: {bibcode}")
                st.progress(score)
    
    except Exception as e:
        st.error(f"Search error: {e}")


def compare_models(models: List[str], text: str, api_key: Optional[str], device: str, batch_size: int):
    """Compare embeddings from different models."""
    embeddings_by_model = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model in enumerate(models):
        status_text.text(f"Processing model {i+1}/{len(models)}: {model}")
        
        try:
            config = EmbedderConfig(
                model=model,
                batch_size=batch_size,
                api_key=api_key,
                device=device
            )
            embedder = create_embedder(config)
            embedding = embedder.embed_single(text)
            
            embeddings_by_model[model] = {
                'embedding': embedding,
                'dimension': embedder.dim,
                'norm': np.linalg.norm(embedding)
            }
            
        except Exception as e:
            st.error(f"Error with model {model}: {e}")
        
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("Comparison complete!")
    
    if len(embeddings_by_model) < 2:
        st.warning("Need at least 2 successful model runs for comparison")
        return
    
    # Display comparison results
    display_model_comparison_results(embeddings_by_model, text)


def display_model_comparison_results(embeddings_by_model: Dict[str, Dict], text: str):
    """Display model comparison results."""
    st.subheader("📊 Model Comparison Results")
    
    # Basic statistics table
    model_stats = []
    for model, data in embeddings_by_model.items():
        model_stats.append({
            'Model': model.split('/')[-1],  # Show just the model name
            'Dimensions': data['dimension'],
            'L2 Norm': f"{data['norm']:.3f}",
            'Mean': f"{np.mean(data['embedding']):.4f}",
            'Std': f"{np.std(data['embedding']):.4f}"
        })
    
    st.table(pd.DataFrame(model_stats))
    
    # Pairwise similarities
    st.subheader("🔗 Pairwise Similarities")
    models = list(embeddings_by_model.keys())
    
    similarity_data = []
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:  # Avoid duplicates
                emb1 = embeddings_by_model[model1]['embedding']
                emb2 = embeddings_by_model[model2]['embedding']
                
                # Normalize embeddings
                emb1_norm = emb1 / np.linalg.norm(emb1)
                emb2_norm = emb2 / np.linalg.norm(emb2)
                
                similarity = np.dot(emb1_norm, emb2_norm)
                
                similarity_data.append({
                    'Model 1': model1.split('/')[-1],
                    'Model 2': model2.split('/')[-1],
                    'Cosine Similarity': f"{similarity:.4f}"
                })
    
    if similarity_data:
        st.table(pd.DataFrame(similarity_data))


def test_hybrid_search(query: str, semantic_weight: float, keyword_weight: float, 
                      vector_fields: List[str], keyword_fields: List[str],
                      model: str, api_key: Optional[str], device: str):
    """Simulate hybrid search testing."""
    st.subheader("Hybrid Search Simulation")
    
    # Generate query embedding
    try:
        config = EmbedderConfig(model=model, api_key=api_key, device=device, batch_size=32)
        embedder = create_embedder(config)
        query_embedding = embedder.embed_single(query)
        
        st.success("Query embedding generated")
        
        # Mock search results
        st.markdown("**Simulated Solr Query:**")
        
        # Build mock Solr query
        solr_query_parts = []
        
        if semantic_weight > 0 and vector_fields:
            vector_query = " OR ".join([f"{field}:[vector query]" for field in vector_fields])
            solr_query_parts.append(f"({vector_query})^{semantic_weight:.2f}")
        
        if keyword_weight > 0 and keyword_fields:
            keyword_query = " OR ".join([f"{field}:{query}" for field in keyword_fields])
            solr_query_parts.append(f"({keyword_query})^{keyword_weight:.2f}")
        
        final_query = " OR ".join(solr_query_parts)
        
        st.code(final_query, language="text")
        
        # Mock results analysis
        st.markdown("**Expected Behavior:**")
        
        if semantic_weight > keyword_weight:
            st.info("**Semantic-heavy search**: Results will prioritize conceptual similarity over exact keyword matches")
        elif keyword_weight > semantic_weight:
            st.info("**Keyword-heavy search**: Results will prioritize exact term matches over conceptual similarity")
        else:
            st.info("**Balanced search**: Equal weight given to semantic and keyword matching")
        
        # Configuration export
        st.subheader("Export Configuration")
        
        config_data = {
            'hybrid_search_config': {
                'semantic_weight': semantic_weight,
                'keyword_weight': keyword_weight,
                'vector_fields': vector_fields,
                'keyword_fields': keyword_fields,
                'embedding_model': model
            }
        }
        
        yaml_config = yaml.dump(config_data, default_flow_style=False)
        
        st.code(yaml_config, language="yaml")
        
        st.download_button(
            label="Download Hybrid Search Config",
            data=yaml_config,
            file_name="hybrid_search_config.yaml",
            mime="text/yaml"
        )
    
    except Exception as e:
        st.error(f"Error in hybrid search testing: {e}")


def parse_years(years_str: str) -> List[int]:
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


@st.cache_data
def discover_available_years(input_dir: str) -> List[int]:
    """Discover all available year files in the input directory."""
    base_path = Path(input_dir)
    if not base_path.exists():
        return []
    
    years = []
    # Look for ADS naming convention: ads_metadata_YYYY_full.jsonl
    for file_path in base_path.glob("ads_metadata_*_full.jsonl"):
        if file_path.is_file():
            name = file_path.name
            # Extract year from ads_metadata_YYYY_full.jsonl
            parts = name.split("_")
            if len(parts) >= 3 and parts[0] == "ads" and parts[1] == "metadata" and parts[3] == "full.jsonl":
                year_str = parts[2]
                if year_str.isdigit() and len(year_str) == 4:
                    year = int(year_str)
                    if 1900 <= year <= 2100:  # Reasonable year range
                        years.append(year)
    
    # Fallback: Look for standard year files
    if not years:
        for file_path in base_path.glob("*"):
            if file_path.is_file():
                name = file_path.name
                # Remove extensions to get the base name
                for ext in [".json.gz", ".jsonl.gz", ".json", ".jsonl"]:
                    if name.endswith(ext):
                        base_name = name[:-len(ext)]
                        break
                else:
                    continue
                
                # Check if base name is a 4-digit year
                if base_name.isdigit() and len(base_name) == 4:
                    year = int(base_name)
                    if 1900 <= year <= 2100:  # Reasonable year range
                        years.append(year)
    
    return sorted(years)


def find_year_file(input_dir: str, year: int) -> Optional[Path]:
    """Find the data file for a given year."""
    base_path = Path(input_dir)
    
    # Try ADS naming convention first: ads_metadata_YYYY_full.jsonl
    ads_file = base_path / f"ads_metadata_{year}_full.jsonl"
    if ads_file.exists():
        return ads_file
    
    # Fallback to standard naming conventions
    for ext in [".json", ".jsonl", ".json.gz", ".jsonl.gz"]:
        file_path = base_path / f"{year}{ext}"
        if file_path.exists():
            return file_path
    
    return None


def save_json_format(output_path: Path, embeddings: np.ndarray, bibcodes: List[str], 
                    texts: List[str], metadata: Dict, include_metadata: List[str]):
    """Save in JSON format with metadata."""
    
    # Main embeddings file
    data = {
        "embeddings": embeddings.tolist(),
        "bibcodes": bibcodes,
        "metadata": metadata
    }
    
    if "Raw Paper Data" in include_metadata:
        data["texts"] = texts
    
    with open(output_path / "embeddings.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # Separate bibcode mapping for easy lookup
    if "Bibcodes" in include_metadata:
        bibcode_mapping = {bibcode: idx for idx, bibcode in enumerate(bibcodes)}
        with open(output_path / "bibcode_index.json", "w") as f:
            json.dump(bibcode_mapping, f, indent=2)
    
    # Metadata file
    if "Field Configuration" in include_metadata or "Model Info" in include_metadata:
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def save_csv_format(output_path: Path, embeddings: np.ndarray, bibcodes: List[str],
                   texts: List[str], metadata: Dict, include_metadata: List[str]):
    """Save in CSV format."""
    
    # Create DataFrame
    import pandas as pd
    df_data = {"bibcode": bibcodes}
    
    # Add embedding dimensions as columns
    for i in range(embeddings.shape[1]):
        df_data[f"dim_{i}"] = embeddings[:, i]
    
    if "Raw Paper Data" in include_metadata:
        df_data["text"] = texts
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path / "embeddings.csv", index=False)
    
    # Metadata as separate file
    if "Field Configuration" in include_metadata or "Model Info" in include_metadata:
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def save_solr_format(output_path: Path, embeddings: np.ndarray, bibcodes: List[str],
                    texts: List[str], metadata: Dict, include_metadata: List[str]):
    """Save in Solr-ready format for hybrid search."""
    
    # Create Solr update format
    solr_docs = []
    
    for i, (bibcode, embedding) in enumerate(zip(bibcodes, embeddings)):
        doc = {
            "bibcode": bibcode,
            "embedding_vector": embedding.tolist(),
            "embedding_model": metadata["model_info"]["model_name"],
            "embedding_dim": len(embedding),
            "fields_used": metadata["processing_info"]["fields_used"]
        }
        
        if "Raw Paper Data" in include_metadata:
            doc["embedded_text"] = texts[i]
        
        solr_docs.append(doc)
    
    # Save as JSONL for Solr ingestion
    with open(output_path / "solr_embeddings.jsonl", "w") as f:
        for doc in solr_docs:
            f.write(json.dumps(doc) + "\n")
    
    # Create Solr schema snippet
    schema_snippet = {
        "fields": [
            {"name": "bibcode", "type": "string", "stored": True, "indexed": True},
            {"name": "embedding_vector", "type": "pdoubles", "stored": True, "indexed": False},
            {"name": "embedding_model", "type": "string", "stored": True, "indexed": True},
            {"name": "embedding_dim", "type": "pint", "stored": True, "indexed": False},
            {"name": "fields_used", "type": "strings", "stored": True, "indexed": False}
        ]
    }
    
    if "Raw Paper Data" in include_metadata:
        schema_snippet["fields"].append({
            "name": "embedded_text", "type": "text_general", "stored": True, "indexed": True
        })
    
    with open(output_path / "solr_schema.json", "w") as f:
        json.dump(schema_snippet, f, indent=2)
    
    # Create hybrid search example query
    example_query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "more_like_this": {
                            "fields": ["title", "abstract"],
                            "like": "QUERY_TEXT_HERE",
                            "boost": metadata.get("keyword_weight", 0.3)
                        }
                    },
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                                "params": {
                                    "query_vector": "QUERY_EMBEDDING_VECTOR_HERE"
                                }
                            },
                            "boost": metadata.get("semantic_weight", 0.7)
                        }
                    }
                ]
            }
        }
    }
    
    with open(output_path / "hybrid_search_template.json", "w") as f:
        json.dump(example_query, f, indent=2)


def get_sample_papers() -> Dict[str, str]:
    """Return sample scientific paper abstracts for testing."""
    return {
        "Exoplanet Detection": "We present the discovery of a new exoplanet using the transit method with the Kepler Space Telescope. The planet, designated KOI-123b, orbits a sun-like star every 45.2 days and shows evidence of atmospheric water vapor.",
        
        "Dark Matter Study": "Analysis of galaxy cluster dynamics reveals new constraints on dark matter particle interactions. Using weak lensing observations from the Hubble Space Telescope, we measure the dark matter distribution in 15 massive clusters.",
        
        "Cosmic Microwave Background": "We present high-precision measurements of the cosmic microwave background anisotropies using data from the Planck satellite. Our results provide new insights into the geometry of the universe and the nature of dark energy.",
        
        "Gravitational Waves": "The detection of gravitational waves from a binary black hole merger provides direct confirmation of Einstein's general theory of relativity. We analyze the waveform characteristics and infer the masses and spins of the merging objects.",
        
        "Stellar Evolution": "Observations of red giant stars in globular clusters reveal insights into stellar evolution and age determination. We use asteroseismology to probe the internal structure of these evolved stars.",
        
        "Galaxy Formation": "High-redshift galaxy observations with the James Webb Space Telescope show evidence of rapid star formation in the early universe. These findings challenge current models of galaxy assembly and evolution."
    }


if __name__ == "__main__":
    main()
