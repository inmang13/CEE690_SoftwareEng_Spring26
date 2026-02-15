# scripts/pipeline.py
"""Main pipeline script to process Durham flow meter data."""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rdii.data_loader import read_all_flow_meters
from rdii.data_cleaner import clean_sewer_timeseries


def main():
    """Run the complete data processing pipeline."""
    
    print("=" * 70)
    print("DURHAM FLOW METER DATA PIPELINE")
    print("=" * 70)
    
    # Define paths
    raw_data_dir = PROJECT_ROOT / "data" / "raw" / "Durham"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    combined_file = processed_dir / "combined_flow_data.csv"
    cleaned_file = processed_dir / "cleaned_flow_data.csv"
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data (or skip if already done)
    print("\nStep 1: Loading flow meter data...")
    print("-" * 70)
    
    if combined_file.exists():
        print(f"Found existing file: {combined_file}")
        print("Loading from file...")
        flow_data = pd.read_csv(combined_file, parse_dates=['DateTime'])
        print(f"✓ Loaded {len(flow_data)} records")
    else:
        print("Loading from raw CSV files...")
        try:
            flow_data = read_all_flow_meters(str(raw_data_dir), verbose=True)
            flow_data.to_csv(combined_file, index=False)
            print(f"✓ Saved combined data to: {combined_file}")
        except Exception as e:
            print(f"\n✗ Failed to load data: {e}")
            sys.exit(1)
    
    # Step 2: Clean data
    print("\nStep 2: Cleaning data...")
    print("-" * 70)
    try:
        cleaned_data = clean_sewer_timeseries(
            flow_data,
            flow_col='Flow_MGD',
            freq='15min',
            interp_limit=4
        )
        
        # Save cleaned data
        cleaned_data.to_csv(cleaned_file, index=False)
        print(f"\n✓ Saved cleaned data to: {cleaned_file}")
        
        
    except Exception as e:
        print(f"\n✗ Failed to clean data: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
