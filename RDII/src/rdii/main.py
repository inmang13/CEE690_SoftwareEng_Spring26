# scripts/main.py
"""Main pipeline script to process Durham flow meter data."""

import json
import sys
from pathlib import Path

import pandas as pd

from rdii.data_loader import read_all_flow_meters
from rdii.data_cleaner import clean_sewer_timeseries
from rdii.remove_BWI import calculate_BWI_minflow
from rdii.plots import plot_all_meters



def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_or_combine_data(raw_dir: Path, combined_file: Path, verbose: bool = True) -> pd.DataFrame:
    """Load existing combined data or create from raw files."""
    if combined_file.exists():
        if verbose:
            print(f"Found existing file: {combined_file.name}")
            print("Loading from file...")
        flow_data = pd.read_csv(combined_file, parse_dates=['DateTime'])
        if verbose:
            print(f"✓ Loaded {len(flow_data):,} records")
    else:
        if verbose:
            print("Loading from raw CSV files...")
        flow_data = read_all_flow_meters(str(raw_dir), verbose=verbose)
        flow_data.to_csv(combined_file, index=False)
        if verbose:
            print(f"✓ Saved combined data to: {combined_file}")
    
    return flow_data


def clean_and_save(data: pd.DataFrame, config: dict, output_file: Path, verbose: bool = True) -> pd.DataFrame:
    """Clean data and save results."""
    if verbose:
        print("\nCleaning data...")
    
    cleaned_data = clean_sewer_timeseries(
        data,
        flow_col=config['flow_column'],
        freq=config['frequency'],
        interp_limit=config['interpolation_limit']
    )
    
    cleaned_data.to_csv(output_file, index=False)
    
    if verbose:
        print(f"✓ Saved cleaned data to: {output_file}")
        print("\nCleaning Summary:")
        print(f"  Total records: {len(cleaned_data):,}")
        print(f"  QC Flag counts:")
        print(cleaned_data['QC_flag'].value_counts().to_string())
    
    return cleaned_data


def main(config_path: str = 'config.json'):
    """Run the complete data processing pipeline."""
    
    print("=" * 70)
    print("DURHAM FLOW METER DATA PIPELINE")
    print("=" * 70)
    
    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in config file: {e}")
        sys.exit(1)
    
    # Setup paths
    project_root = Path(config['project_root']) if 'project_root' in config else Path(__file__).parent.parent.parent
    raw_data_dir = project_root / config['paths']['raw_data']
    processed_dir = project_root / config['paths']['processed_data']
    plot_dir = project_root / config['paths']['plots_dir']

    combined_file = processed_dir / config['paths']['combined_filename']
    cleaned_file = processed_dir / config['paths']['cleaned_filename']
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\nStep 1: Loading flow meter data...")
    print("-" * 70)
    try:
        flow_data = load_or_combine_data(raw_data_dir, combined_file, verbose=True)
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        sys.exit(1)
    
    # Step 2: Clean data
    print("\nStep 2: Cleaning data...")
    print("-" * 70)
    try:
        cleaned_data = clean_and_save(
            flow_data,
            config['cleaning'],
            cleaned_file,
            verbose=True
        )
    except Exception as e:
        print(f"✗ Failed to clean data: {e}")
        sys.exit(1)
    
    #Step 3: Create plots (optional, can be added later)
    print("\nStep 3: Creating QC plots...")
    print("-" * 70)
    try:
        # Create QC plots for all meters
        plot_all_meters(cleaned_data, plot_type='qc')
        # Create BWI estimate plots for all meters
        plot_all_meters(cleaned_data, plot_type='bwi')

    except Exception as e:
        print(f"✗ Failed to create plots: {e}")
        sys.exit(1)


    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    main(config_file)
