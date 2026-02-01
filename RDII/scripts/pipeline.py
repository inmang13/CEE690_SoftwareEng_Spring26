"""
Main pipeline script to process Durham flow meter data."""

import os
import sys

import pandas as pd

from preprocessing.data_loader import read_all_flow_meters


def main():
    """Run the complete data processing pipeline."""
    
    # Define paths
    raw_data_dir = "data/raw/Durham/"
    output_dir = "data/processed/"
    output_file = os.path.join(output_dir, "combined_flow_data.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load data
    flow_data = read_all_flow_meters(raw_data_dir, verbose=True)
    print(f"\n✓ Loaded {len(flow_data)} total records")
    print(f"✓ Meters found: {flow_data['Meter'].unique().tolist()}")

if __name__ == "__main__":
    main()
