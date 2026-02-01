"""
Main pipeline script to process Durham flow meter data."""

import os
import sys
from pathlib import Path

import pandas as pd

from preprocessing.data_loader import read_all_flow_meters

# Get absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def main():
    """Run the complete data processing pipeline."""
    
    # Define paths
    raw_data_dir = PROJECT_ROOT / "data" / "raw" / "Durham"
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_file = output_dir / "combined_flow_data.csv"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    flow_data = read_all_flow_meters(raw_data_dir, verbose=True)
    print(f"\n✓ Loaded {len(flow_data)} total records")
    print(f"✓ Meters found: {flow_data['Meter'].unique().tolist()}")

if __name__ == "__main__":
    main()
