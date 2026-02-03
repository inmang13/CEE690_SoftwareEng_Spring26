"""
Main pipeline script to process Durham flow meter data."""

import os
import sys
from pathlib import Path

from rdii.data_loader import read_all_flow_meters

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
    
    # Step 2: Basic Filtering
    initial_count = len(flow_data)
    flow_data = flow_data[flow_data['Flow_MGD'].notna()]
    filtered_count = len(flow_data)
    
    print(f"✓ Removed {initial_count - filtered_count} rows with "
          f"missing Flow_MGD")
    print(f"✓ {filtered_count} records remaining")
    
    # Step 3: Saving File
    flow_data.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()
