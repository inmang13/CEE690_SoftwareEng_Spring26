# src/rdii/data_cleaner.py
"""Module for cleaning sewer flow timeseries data."""

import pandas as pd
import numpy as np


def clean_sewer_timeseries(
    df,
    flow_col='Flow_MGD',
    freq='15min',
    interp_limit=4
):
    """
    Clean sewer flow timeseries meter-by-meter.
    """

    cleaned_all = []
    
    for meter, group in df.groupby('Meter'):
        print(f"\nCleaning meter: {meter}")
        
        # Clean individual meter
        cleaned = _clean_single_meter(
            group,
            flow_col=flow_col,
            freq=freq,
            interp_limit=interp_limit
        )
        
        cleaned_all.append(cleaned)
    
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
    df, interpolated_mask = interpolate_gaps(df, flow_col, interp_limit)
    df = add_qc_flags(df, flow_col, interpolated_mask, negative_mask, flatline_mask)  

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

def remove_negative_flows(df,flow_col,threshold,verbose=True):
    """
    Remove negative or physically impossible flow values.
    """
    df = df.copy()
    
    # Detect negative flows
    negative_mask = df[flow_col] < threshold
    negative_count = negative_mask.sum()
    
    if negative_count > 0:
        if verbose:
            print(f"  Removed {negative_count} negative/invalid flow values")
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
    
    if flat_count > 0:
        print(f"  Removed {flat_count} flatline points")
    
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


def add_qc_flags(df, flow_col,interpolated_mask, negative_mask, flatline_mask):

    """
    Add quality control flags.    
    """

    df = df.copy()
    
    df['QC_flag'] = 'OK'

    # Mark interpolated values
    df.loc[interpolated_mask, 'QC_flag'] = 'INTERPOLATED'

    # Mark negative values
    df.loc[negative_mask, 'QC_flag'] = 'NEGATIVE'

    # Mark removed flatlines (may overwrite INTERPOLATED if both)
    df.loc[flatline_mask, 'QC_flag'] = 'FLATLINE_REMOVED'
    
    # Mark missing data (highest priority - overwrites everything)
    df.loc[df[flow_col].isna(), 'QC_flag'] = 'MISSING' 

    return df
