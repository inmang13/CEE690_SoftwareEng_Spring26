# src/rdii/remove_BWI.py
"""Module for calculating and removing Base Wastewater Infiltration (BWI) from flow data."""

import pandas as pd
import numpy as np

def calculate_BWI_minflow(df, fraction_min=0.85, rolling_window=30, night_start=1, night_end=7):
    """
    Calculate Base Wastewater Infiltration (BWI) minimum flow using nighttime flows.

    """
    
    df_night = df.copy()
    
    # Ensure DateTime is index
    if 'DateTime' in df_night.columns:
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