# src/rdii/data_cleaner.py
"""Module for cleaning sewer flow timeseries data."""

import sys
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from rdii.data_loader import read_all_flow_meters
import json

def _clean_meter_wrapper(args):
    group, flow_col, freq, interp_limit = args
    return _clean_single_meter(
        group,
        flow_col=flow_col,
        freq=freq,
        interp_limit=interp_limit
    )

def clean_sewer_timeseries(
    df,
    flow_col='Flow_MGD',
    freq='15min',
    interp_limit=4,
    n_workers=None
):
    """
    Clean sewer flow timeseries meter-by-meter.
    """

    # Handle empty dataframe
    if len(df) == 0:
        # Return empty df with QC_flag column
        result = df.copy()
        result['QC_flag'] = pd.Series(dtype='object')
        return result

    # Default = all available cores
    if n_workers is None:
        n_workers = os.cpu_count()

    # Split by meter
    groups = [
        (group, flow_col, freq, interp_limit)
        for _, group in df.groupby('Meter')
    ]

    # Run meters in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        cleaned_all = list(executor.map(_clean_meter_wrapper, groups))

    return pd.concat(cleaned_all, ignore_index=True)


def _clean_single_meter(df, flow_col, freq, interp_limit):

    """
    Clean data for a single meter.    
    """

    # Sort by time
    df = df.sort_values('DateTime').copy()
    df=df[['DateTime', flow_col,'Meter','Source_File']]
    
    # Enforce regular timestep
    df = enforce_regular_timestep(df, freq)
    
    
    # Apply cleaning steps and track what happened
    df, negative_mask = remove_negative_flows(df, flow_col, threshold=0.0)
    df, flatline_mask = remove_flatlines(df, flow_col,)
    df, outlier_mask = remove_low_outliers(df, flow_col, window=30, threshold_multiplier=3)
    df, interpolated_mask = interpolate_gaps(df, flow_col, interp_limit)
    df = add_qc_flags(df, flow_col, interpolated_mask, negative_mask, flatline_mask,outlier_mask)  

    return df


def enforce_regular_timestep(df, freq):

    """
    Enforce regular time intervals in the data.
    """

    # Set DateTime as index
    df = df.set_index('DateTime')
    
    # Create regular frequency
    original_len = len(df)

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    # Resample numeric columns (take mean)
    df_numeric = df[numeric_cols].resample(freq, label='left', closed='left').mean()
    
    # Resample non-numeric columns (take first value)
    df_non_numeric = df[non_numeric_cols].resample(freq, label='left', closed='left').first()
    
    # Combine back together
    df = pd.concat([df_numeric, df_non_numeric], axis=1)
    
    new_len = len(df)
    
    added = new_len - original_len
    if added > 0:
        print(f"  Added {added} timestamps to regularize frequency")
    
    # Reset index (DateTime becomes column again)
    df = df.reset_index()
    
    return df

def remove_negative_flows(df,flow_col,threshold,verbose=False):
    """
    Remove negative or physically impossible flow values.
    """
    df = df.copy()
    
    # Detect negative flows
    negative_mask = df[flow_col] < threshold
    negative_count = negative_mask.sum()
    
    if negative_count > 0:
        df.loc[negative_mask, flow_col] = np.nan
    
    return df, negative_mask


def remove_flatlines(df, flow_col, window=48):

    """
    Remove flatline periods (constant values).
    """

    df = df.copy()
    
    # Detect flatlines (zero standard deviation)
    flatlines = df[flow_col].rolling(window).std() == 0
    flat_count = flatlines.sum()
    
    df.loc[flatlines, flow_col] = np.nan
    
    return df, flatlines


def interpolate_gaps(df, flow_col, interp_limit):

    """
    Interpolate short gaps in data.
    """

    df = df.copy()
    
    # Track which values were NaN before interpolation
    was_nan = df[flow_col].isna()
    
    # Find gap sizes (in terms of consecutive NaNs)
    # Create groups of consecutive NaNs
    nan_groups = (was_nan != was_nan.shift()).cumsum()
    gap_sizes = was_nan.groupby(nan_groups).transform('sum')

    # Only interpolate small gaps
    should_interpolate = was_nan & (gap_sizes <= 4)

    # Find gap sizes (in terms of consecutive NaNs)
    # Create groups of consecutive NaNs
    nan_groups = (was_nan != was_nan.shift()).cumsum()
    gap_sizes = was_nan.groupby(nan_groups).transform('sum')

    # Only interpolate small gaps
    should_interpolate = was_nan & (gap_sizes <= 4)

    # Temporarily mark large gaps so they won't be interpolated
    df.loc[~should_interpolate & was_nan, flow_col] = -999999

    # Linear interpolation for small gaps only
    df[flow_col] = df[flow_col].interpolate(
        method='linear',
        limit=interp_limit,
        limit_direction='both'
    )

    # Restore large gaps as NaN
    df.loc[df[flow_col] == -999999, flow_col] = np.nan

    # Identify which NaNs were filled
    interpolated_mask = should_interpolate & df[flow_col].notna()
        
    return df, interpolated_mask


def remove_low_outliers(df, flow_col, window=14, threshold_multiplier=3,verbose=False):
    """
    Remove low outliers based on daily minimums using robust MAD detection.
    """

    df = df.copy()

    # Extract date from DateTime
    df['Date'] = pd.to_datetime(df['DateTime']).dt.date
    
    # Calculate daily minimum for each date
    daily_min = (
        df.groupby('Date')[flow_col]
        .min()
        .reset_index()
        .rename(columns={flow_col: 'min_flow'})
    )
    
    flow = daily_min['min_flow']

    # Rolling baseline (median is robust to outliers)
    rolling_median = flow.rolling(window, center=True).median()

    # difference from local baseline
    deviation = flow - rolling_median

    # estimate spread using MAD (robust)
    mad = deviation.abs().rolling(window, center=True).median()

    threshold = -threshold_multiplier * 1.4826 * mad
    
    # Identify negative spikes (days with suspiciously low minimums)
    neg_spikes = deviation < threshold
    outlier_count = neg_spikes.sum()

    # Mark those days' min_flow as NaN
    daily_min.loc[neg_spikes, 'min_flow'] = np.nan
    
    # Get the dates that were flagged as outliers
    flagged_dates = daily_min.loc[neg_spikes, 'Date'].values
    
    # Create mask for all records from flagged days
    outlier_mask = df['Date'].isin(flagged_dates)
    outlier_record_count = outlier_mask.sum()
    
    if outlier_record_count > 0:
        df.loc[outlier_mask, flow_col] = np.nan
    
    # Clean up temporary column
    df = df.drop('Date', axis=1)
    
    return df, outlier_mask


def add_qc_flags(df, flow_col,interpolated_mask, negative_mask, flatline_mask,outlier_mask):

    """
    Add quality control flags.    
    """

    df = df.copy()
    
    df['QC_flag'] = 'OK'

    # Mark interpolated values
    df.loc[interpolated_mask, 'QC_flag'] = 'INTERPOLATED'

    # Mark low outliers values
    df.loc[outlier_mask, 'QC_flag'] = 'LOW_OUTLIER'

    # Mark negative values
    df.loc[negative_mask, 'QC_flag'] = 'NEGATIVE'

    # Mark removed flatlines (may overwrite INTERPOLATED if both)
    df.loc[flatline_mask, 'QC_flag'] = 'FLATLINE_REMOVED'
    
    # Mark missing data (highest priority - overwrites everything)
    df.loc[df[flow_col].isna(), 'QC_flag'] = 'MISSING' 

    return df

def load_or_combine_data(raw_dir, combined_file, verbose = False):
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

        # Load data
        df = load_or_combine_data(raw_data_dir, combined_file)
        clean_data = clean_sewer_timeseries(
            df,
            flow_col=config['cleaning']['flow_column'],
            freq=config['cleaning']['frequency'],
            interp_limit=config['cleaning']['interpolation_limit']
        )

        print(f"✓ Loaded cleaned data with {len(clean_data)} rows and {len(clean_data['Meter'].unique())} meters")
        
        clean_data.to_csv(cleaned_file, index=False)
        print(f"✓ Saved cleaned data to: {cleaned_file}")


if __name__ == "__main__":
        config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
        main(config_file)
        
