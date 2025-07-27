#!/usr/bin/env python3
"""Helper script to test and interact with index.db"""

import sqlite3
import sys
from pathlib import Path
from src.sciembed.components.index import Index

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_index.py <index_db_path> [command]")
        print("Commands:")
        print("  stats - Show index statistics")
        print("  years - List available years")
        print("  models - List available models") 
        print("  lookup <bibcode> - Look up a specific bibcode")
        print("  sample [n] - Show n random entries (default 5)")
        return
    
    index_path = Path(sys.argv[1])
    if not index_path.exists():
        print(f"Error: {index_path} does not exist")
        return
    
    index = Index(index_path)
    command = sys.argv[2] if len(sys.argv) > 2 else "stats"
    
    if command == "stats":
        stats = index.get_stats()
        print("Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")
    
    elif command == "years":
        years = index.list_years()
        print(f"Available years ({len(years)}): {', '.join(map(str, years))}")
    
    elif command == "models":
        models = index.list_models()
        print(f"Available models ({len(models)}):")
        for model in models:
            print(f"  {model}")
    
    elif command == "lookup":
        if len(sys.argv) < 4:
            print("Usage: lookup <bibcode>")
            return
        bibcode = sys.argv[3]
        entries = index.lookup_bibcode(bibcode)
        if entries:
            print(f"Found {len(entries)} entries for {bibcode}:")
            for entry in entries:
                print(f"  Year: {entry.year}, Row: {entry.row_id}, Model: {entry.model}")
                print(f"  Vector file: {entry.vector_file}")
        else:
            print(f"No entries found for {bibcode}")
    
    elif command == "sample":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        with sqlite3.connect(index_path) as conn:
            cursor = conn.execute(f"""
                SELECT bibcode, year, row_id, model, vector_file 
                FROM embeddings 
                ORDER BY RANDOM() 
                LIMIT {n}
            """)
            print(f"Random sample of {n} entries:")
            for row in cursor.fetchall():
                print(f"  {row[0]} (year {row[1]}, row {row[2]}, model: {row[3]})")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
