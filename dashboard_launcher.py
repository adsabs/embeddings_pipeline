#!/usr/bin/env python3
"""Simple launcher script for the SciX Experimental Embeddings Pipeline."""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "src" / "sciembed" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    # Launch Streamlit on port 8502
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(dashboard_path),
        "--server.port", "8502",
        "--server.headless", "true",
        "--server.enableCORS", "false"
    ]
    
    print("Starting SciX Experimental Embeddings Pipeline...")
    print(f"Dashboard location: {dashboard_path}")
    print("Dashboard will be available at: http://localhost:8502")
    print("-" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
