# src/rdii/remove_GWI.py
"""Module for calculating and removing Base Wastewater Infiltration (GWI) from flow data."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
from rdii.plots import plot_GWI_estimate

def process_all_meters_GWI(
    df,
    fraction_min=0.85,
    rolling_window=30,
    night_start=1,
    night_end=7,
):
    """
    Apply GWI calculation and removal to all meters in parallel.
    """

    if len(df) == 0:
        result = df.copy()
        return result

    results = []

    df['DateTime'] = pd.to_datetime(df['DateTime'])

    for _, group in df.groupby("Meter"):
        GWI_estimate = calculate_GWI_minflow(
            group,
            fraction_min=fraction_min,
            rolling_window=rolling_window,
            night_start=night_start,
            night_end=night_end
        )

        corrected_df = remove_GWI(group, gwi_estimate=GWI_estimate) 
        results.append(corrected_df)

    return pd.concat(results, ignore_index=True)


def calculate_GWI_minflow(df, fraction_min=0.85, rolling_window=30, night_start=1, night_end=7):
    """
    Calculate Base Wastewater Infiltration (GWI) minimum flow using nighttime flows.

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
    
    # Expand daily MNF to 15‑minute resolution
    mnf_15min = (
        mnf_smoothed
        .resample('15min')
        .ffill()
    )

    # Align exactly to original data index
    mnf_15min = (
        mnf_15min
        .reindex(df_night.index)
        .ffill()
        .bfill()
    )
    
    # Apply fraction to get GWI estimate
    gwi_estimate = mnf_15min * fraction_min

    return gwi_estimate

def remove_GWI(df, gwi_estimate):
    """
    Remove GWI estimate from original flow data.
    """
    
    df_corrected = df.copy()
    
    # Ensure DateTime is index
    if 'DateTime' in df_corrected.columns:
        df_corrected = df_corrected.set_index('DateTime')
    
    # Subtract GWI estimate from original flow
    df_corrected['GWI_estimate'] = gwi_estimate
    df_corrected['Flow_MGD_GWI_Corrected'] = df_corrected['Flow_MGD'] - gwi_estimate
    
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
        plots_dir= project_root / config['paths']['plots_dir']
        combined_file = processed_dir / config['paths']['combined_filename']  
        processed_dir.mkdir(parents=True, exist_ok=True)

        combined_file = processed_dir / config['paths']['combined_filename']
        cleaned_file = processed_dir / config['paths']['cleaned_filename']
        gwi_corrected_file = processed_dir / config['paths']['gwi_removed_filename']

        # Load cleaned data
        try:
            cleaned_data = pd.read_csv(cleaned_file)
            print(cleaned_data.head())
            print(f"✓ Loaded cleaned data with {len(cleaned_data)} rows and {len(cleaned_data['Meter'].unique())} meters")
        except Exception as e:
            print(f"✗ Failed to load cleaned data: {e}")
            sys.exit(1)


        try:   
            cleaned_gwi = process_all_meters_GWI(
                cleaned_data,
                fraction_min=config['gwi']['fraction_min'],
                rolling_window=config['gwi']['rolling_window'],
                night_start=config['gwi']['night_start'],
                night_end=config['gwi']['night_end']
                                        )

            for meter_name, meter_df in cleaned_gwi.groupby('Meter'):
                plot_GWI_estimate(meter_df, meter_name, output_dir=plots_dir)

            cleaned_gwi.to_csv(gwi_corrected_file, index=False)
            print(f"✓ Saved clean_gwi data to: {gwi_corrected_file}")

        except Exception as e:
            print(f"✗ Failed to calculate/remove GWI: {e}")
            sys.exit(1)

        
if __name__ == "__main__":
        config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
        main(config_file)