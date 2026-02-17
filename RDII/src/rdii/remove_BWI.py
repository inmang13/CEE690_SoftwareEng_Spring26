# src/rdii/remove_BWI.py
"""Module for calculating and removing Base Wastewater Infiltration (BWI) from flow data."""

import sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import json

def process_all_meters_BWI(
    df,
    fraction_min=0.85,
    rolling_window=30,
    night_start=1,
    night_end=7,
    n_workers=None
):
    """
    Apply BWI calculation and removal to all meters in parallel.
    """

    if len(df) == 0:
        result = df.copy()
        return result

    if n_workers is None:
        n_workers = os.cpu_count()

    # Split DataFrame by Meter
    groups = [
        (group.copy(), fraction_min, rolling_window, night_start, night_end)
        for _, group in df.groupby("Meter")
    ]

    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_process_meter_BWI, groups))

    # Combine all meters
    return pd.concat(results, ignore_index=True)

def _process_meter_BWI(args):
    """
    Worker function for a single meter.
    """
    meter_df, fraction_min, rolling_window, night_start, night_end = args
    
    bwi_estimate = calculate_BWI_minflow(
        meter_df,
        fraction_min=fraction_min,
        rolling_window=rolling_window,
        night_start=night_start,
        night_end=night_end
    )
    
    corrected_df = remove_BWI(meter_df, bwi_estimate)
    return corrected_df


def calculate_BWI_minflow(df, fraction_min=0.85, rolling_window=30, night_start=1, night_end=7):
    """
    Calculate Base Wastewater Infiltration (BWI) minimum flow using nighttime flows.

    """
    
    df_night = df.copy()
    
    # Ensure DateTime is index
    if 'DateTime' in df_night.columns:
        df_night['DateTime'] = pd.to_datetime(df_night['DateTime'], errors='coerce')
        df_night = df_night.set_index('DateTime')
    
    df_night['hour'] = df_night.index.hour
    
    # Filter to nighttime hours (typical minimum-use period)
    night_window = df_night[(df_night['hour'] >= night_start) & (df_night['hour'] <= night_end)]
    
    # Calculate daily minimum nighttime flow
    mnf_daily = night_window['Flow_MGD'].resample('D').min()
    
    # Remove outliers using Tukey method (IQR-based)
    Q1 = mnf_daily.quantile(0.25)
    Q3 = mnf_daily.quantile(0.75)
    IQR = Q3 - Q1
    
    # Upper bound for typical nighttime flows
    upper_bound = Q3 + (1.5 * IQR)
    mnf_daily_cleaned = mnf_daily[mnf_daily < upper_bound]
    
    # Apply rolling mean smoothing
    mnf_smoothed = mnf_daily_cleaned.rolling(
        window=rolling_window,
        center=True,
        min_periods=max(1, rolling_window // 3)  # Require at least 1/3 of window
    ).mean()
    
    # Align smoothed daily MNF to original 15-min resolution
    # Forward-fill to propagate each day's value across all its 15-min intervals
    mnf_15min = mnf_smoothed.reindex(df_night.index, method='ffill')
    
    # Apply fraction to get BWI estimate
    bwi_estimate = mnf_15min * fraction_min

    return bwi_estimate

def remove_BWI(df, bwi_estimate):
    """
    Remove BWI estimate from original flow data.
    """
    
    df_corrected = df.copy()
    
    # Ensure DateTime is index
    if 'DateTime' in df_corrected.columns:
        df_corrected = df_corrected.set_index('DateTime')
    
    # Subtract BWI estimate from original flow
    df_corrected['Flow_MGD_BWI_Corrected'] = df_corrected['Flow_MGD'] - bwi_estimate
    
    return df_corrected.reset_index()

def load_config(config_path ):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
        

def main(config_path: str = 'config.json'):
        
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
        combined_file = processed_dir / config['paths']['combined_filename']        
        processed_dir.mkdir(parents=True, exist_ok=True)

        combined_file = processed_dir / config['paths']['combined_filename']
        cleaned_file = processed_dir / config['paths']['cleaned_filename']
        bwi_corrected_file = processed_dir / config['paths']['bwi_corrected_filename']

        # Load cleaned data
        try:
            cleaned_data = pd.read_csv(cleaned_file)
            print(f"✓ Loaded cleaned data with {len(cleaned_data)} rows and {len(cleaned_data['Meter'].unique())} meters")
        except Exception as e:
            print(f"✗ Failed to load cleaned data: {e}")
            sys.exit(1)

        try:
            n_workers = int(config.get('parallel', {}).get('n_workers', os.cpu_count()))
            
            cleaned_bwi = process_all_meters_BWI(
                cleaned_data,
                fraction_min=config['bwi']['fraction_min'],
                rolling_window=config['bwi']['rolling_window'],
                night_start=config['bwi']['night_start'],
                night_end=config['bwi']['night_end'],
                n_workers=n_workers
            )
            
            cleaned_bwi.to_csv(bwi_corrected_file, index=False)
            print(f"✓ Saved clean_bwi data to: {bwi_corrected_file}")

        except Exception as e:
            print(f"✗ Failed to calculate/remove BWI: {e}")
            sys.exit(1)

        
if __name__ == "__main__":
        config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
        main(config_file)