# tests/test_data_cleaner.py
"""Unit tests for data_cleaner module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from rdii.data_cleaner import (
    clean_sewer_timeseries,
    enforce_regular_timestep,
    remove_negative_flows,
    remove_flatlines,
    interpolate_gaps,
    add_qc_flags
)


# Fixtures
@pytest.fixture
def sample_flow_data():
    """Create sample flow data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    data = {
        'DateTime': dates,
        'Flow_MGD': np.random.uniform(0.5, 2.0, 100),
        'Meter': ['TEST'] * 100,
        'Source_File': ['test.csv'] * 100
    }
    return pd.DataFrame(data)


@pytest.fixture
def flow_data_with_gaps():
    """Create flow data with missing timestamps."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    # Remove some timestamps to create gaps
    dates = dates.delete([10, 11, 12, 50, 51])
    data = {
        'DateTime': dates,
        'Flow_MGD': np.random.uniform(0.5, 2.0, len(dates)),
        'Meter': ['TEST'] * len(dates),
        'Source_File': ['test.csv'] * len(dates)
    }
    return pd.DataFrame(data)


@pytest.fixture
def flow_data_with_negatives():
    """Create flow data with negative values."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    flows = np.random.uniform(0.5, 2.0, 100)
    flows[10:15] = -0.5  # Add negative values
    data = {
        'DateTime': dates,
        'Flow_MGD': flows,
        'Meter': ['TEST'] * 100,
        'Source_File': ['test.csv'] * 100
    }
    return pd.DataFrame(data)


@pytest.fixture
def flow_data_with_flatline():
    """Create flow data with flatline period."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    flows = np.random.uniform(0.5, 2.0, 100)
    flows[30:80] = 1.0  # Create flatline (50+ points)
    data = {
        'DateTime': dates,
        'Flow_MGD': flows,
        'Meter': ['TEST'] * 100,
        'Source_File': ['test.csv'] * 100
    }
    return pd.DataFrame(data)


@pytest.fixture
def flow_data_with_nans():
    """Create flow data with NaN values."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    flows = np.random.uniform(0.5, 2.0, 100)
    flows[20:23] = np.nan  # Small gap (3 points)
    flows[50:60] = np.nan  # Large gap (10 points)
    data = {
        'DateTime': dates,
        'Flow_MGD': flows,
        'Meter': ['TEST'] * 100,
        'Source_File': ['test.csv'] * 100
    }
    return pd.DataFrame(data)


@pytest.fixture
def multi_meter_data():
    """Create data with multiple meters."""
    dates = pd.date_range('2023-01-01', periods=50, freq='15min')
    data = {
        'DateTime': list(dates) * 3,
        'Flow_MGD': np.random.uniform(0.5, 2.0, 150),
        'Meter': ['METER1'] * 50 + ['METER2'] * 50 + ['METER3'] * 50,
        'Source_File': ['test.csv'] * 150
    }
    return pd.DataFrame(data)


# Test enforce_regular_timestep
def test_enforce_regular_timestep_no_gaps(sample_flow_data):
    """Test that regular data stays unchanged."""
    result = enforce_regular_timestep(sample_flow_data, '15min')
    assert len(result) == len(sample_flow_data)
    assert 'DateTime' in result.columns


def test_enforce_regular_timestep_with_gaps(flow_data_with_gaps):
    """Test that gaps are filled with regular timesteps."""
    result = enforce_regular_timestep(flow_data_with_gaps, '15min')
    # Should have 100 timestamps (original full range)
    assert len(result) == 100
    # Check that timestamps are regular
    time_diffs = result['DateTime'].diff().dropna()
    assert all(time_diffs == pd.Timedelta('15min'))


def test_enforce_regular_timestep_preserves_data(sample_flow_data):
    """Test that existing data is preserved."""
    original_flow_sum = sample_flow_data['Flow_MGD'].sum()
    result = enforce_regular_timestep(sample_flow_data, '15min')
    # Sum should be approximately the same (allowing for floating point errors)
    assert abs(result['Flow_MGD'].sum() - original_flow_sum) < 0.01


# Test remove_negative_flows
def test_remove_negative_flows_clean_data(sample_flow_data):
    """Test with data that has no negative values."""
    result, mask = remove_negative_flows(sample_flow_data, 'Flow_MGD', threshold=0.0, verbose=False)
    assert mask.sum() == 0  # No negatives detected
    assert not result['Flow_MGD'].isna().any()


def test_remove_negative_flows_with_negatives(flow_data_with_negatives):
    """Test that negative values are removed."""
    result, mask = remove_negative_flows(flow_data_with_negatives, 'Flow_MGD', threshold=0.0, verbose=False)
    assert mask.sum() == 5  # 5 negative values
    assert result.loc[mask, 'Flow_MGD'].isna().all()


def test_remove_negative_flows_threshold(sample_flow_data):
    """Test custom threshold."""
    result, mask = remove_negative_flows(sample_flow_data, 'Flow_MGD', threshold=1.0, verbose=False)
    # Some values should be below 1.0 and marked
    assert mask.sum() > 0


# Test remove_flatlines
def test_remove_flatlines_no_flatline(sample_flow_data):
    """Test with data that has no flatlines."""
    result, mask = remove_flatlines(sample_flow_data, 'Flow_MGD', window=48)
    # Random data should have very few or no flatlines
    assert mask.sum() < 5


def test_remove_flatlines_with_flatline(flow_data_with_flatline):
    """Test that flatlines are detected and removed."""
    result, mask = remove_flatlines(flow_data_with_flatline, 'Flow_MGD', window=48)
    assert mask.sum() > 0  # Flatline detected
    assert result.loc[mask, 'Flow_MGD'].isna().all()


def test_remove_flatlines_window_size(flow_data_with_flatline):
    """Test different window sizes."""
    result_small, mask_small = remove_flatlines(flow_data_with_flatline, 'Flow_MGD', window=10)
    result_large, mask_large = remove_flatlines(flow_data_with_flatline, 'Flow_MGD', window=100)
    # Smaller window should detect more flatlines
    assert mask_small.sum() >= mask_large.sum()


# Test interpolate_gaps
def test_interpolate_gaps_no_gaps(sample_flow_data):
    """Test with data that has no gaps."""
    result, mask = interpolate_gaps(sample_flow_data, 'Flow_MGD', interp_limit=4)
    assert mask.sum() == 0  # Nothing interpolated
    assert not result['Flow_MGD'].isna().any()


def test_interpolate_gaps_small_gap(flow_data_with_nans):
    """Test that small gaps are interpolated."""
    result, mask = interpolate_gaps(flow_data_with_nans, 'Flow_MGD', interp_limit=4)
    # Small gap (3 points) should be interpolated
    assert mask.sum() == 3
    # Check that small gap is filled
    assert not result['Flow_MGD'].iloc[20:23].isna().any()


def test_interpolate_gaps_large_gap(flow_data_with_nans):
    """Test that large gaps are NOT interpolated."""
    result, mask = interpolate_gaps(flow_data_with_nans, 'Flow_MGD', interp_limit=4)
    # Large gap (10 points) should NOT be interpolated
    # Check that large gap still has NaNs
    assert result['Flow_MGD'].iloc[50:60].isna().any()


def test_interpolate_gaps_limit(flow_data_with_nans):
    """Test interpolation limit parameter."""
    # With limit=1, only gaps of 1 should be filled
    result, mask = interpolate_gaps(flow_data_with_nans, 'Flow_MGD', interp_limit=1)
    # Should not fill 3-point gap
    assert result['Flow_MGD'].iloc[20:23].isna().any()


# Test add_qc_flags
def test_add_qc_flags_all_ok():
    """Test QC flags when everything is OK."""
    df = pd.DataFrame({
        'DateTime': pd.date_range('2023-01-01', periods=10, freq='15min'),
        'Flow_MGD': [1.0] * 10
    })
    interpolated = pd.Series([False] * 10)
    negative = pd.Series([False] * 10)
    flatline = pd.Series([False] * 10)
    outlier = pd.Series([False] * len(df))
    
    result = add_qc_flags(df, 'Flow_MGD', interpolated, negative, flatline,outlier)
    assert all(result['QC_flag'] == 'OK')


def test_add_qc_flags_interpolated():
    """Test QC flags for interpolated values."""
    df = pd.DataFrame({
        'DateTime': pd.date_range('2023-01-01', periods=10, freq='15min'),
        'Flow_MGD': [1.0] * 10
    })
    interpolated = pd.Series([False, False, True, True, False, False, False, False, False, False])
    negative = pd.Series([False] * 10)
    flatline = pd.Series([False] * 10)
    outlier = pd.Series([False] * len(df))
    
    result = add_qc_flags(df, 'Flow_MGD', interpolated, negative, flatline,outlier)
    assert result.loc[2:3, 'QC_flag'].eq('INTERPOLATED').all()


def test_add_qc_flags_missing():
    """Test QC flags for missing values."""
    df = pd.DataFrame({
        'DateTime': pd.date_range('2023-01-01', periods=10, freq='15min'),
        'Flow_MGD': [1.0, 1.0, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    interpolated = pd.Series([False] * 10)
    negative = pd.Series([False] * 10)
    flatline = pd.Series([False] * 10)
    outlier = pd.Series([False] * len(df))
    
    result = add_qc_flags(df, 'Flow_MGD', interpolated, negative, flatline,outlier)
    assert result.loc[2:3, 'QC_flag'].eq('MISSING').all()


def test_add_qc_flags_priority():
    """Test that MISSING flag has highest priority."""
    df = pd.DataFrame({
        'DateTime': pd.date_range('2023-01-01', periods=10, freq='15min'),
        'Flow_MGD': [1.0, 1.0, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    # Mark as both interpolated and negative, but it's actually NaN
    interpolated = pd.Series([False, False, True, True, False, False, False, False, False, False])
    negative = pd.Series([False, False, True, True, False, False, False, False, False, False])
    flatline = pd.Series([False] * 10)
    outlier = pd.Series([False] * len(df))
    
    result = add_qc_flags(df, 'Flow_MGD', interpolated, negative, flatline,outlier)
    # MISSING should override everything
    assert result.loc[2:3, 'QC_flag'].eq('MISSING').all()


# Test clean_sewer_timeseries (integration tests)
def test_clean_sewer_timeseries_single_meter(sample_flow_data, capsys):
    """Test full cleaning pipeline for single meter."""
    result = clean_sewer_timeseries(sample_flow_data, flow_col='Flow_MGD', freq='15min', interp_limit=4)
    
    # Check output structure
    assert 'QC_flag' in result.columns
    assert len(result) >= len(sample_flow_data)
    assert result['Meter'].nunique() == 1


def test_clean_sewer_timeseries_multiple_meters(multi_meter_data):
    """Test full cleaning pipeline for multiple meters."""
    result = clean_sewer_timeseries(multi_meter_data, flow_col='Flow_MGD', freq='15min', interp_limit=4)
    
    # Check that all meters are present
    assert result['Meter'].nunique() == 3
    assert set(result['Meter'].unique()) == {'METER1', 'METER2', 'METER3'}
    
    # Check QC flags exist
    assert 'QC_flag' in result.columns


def test_clean_sewer_timeseries_with_all_issues():
    """Test cleaning with data containing all types of issues."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    flows = np.random.uniform(0.5, 2.0, 100)
    
    # Add various issues
    flows[10:15] = -0.5  # Negative values
    flows[30:80] = 1.0   # Flatline
    flows[20:23] = np.nan  # Small gap
    
    data = pd.DataFrame({
        'DateTime': dates,
        'Flow_MGD': flows,
        'Meter': ['TEST'] * 100,
        'Source_File': ['test.csv'] * 100
    })
    
    result = clean_sewer_timeseries(data, flow_col='Flow_MGD', freq='15min', interp_limit=4)
    
    # Check that all QC flag types are present
    qc_flags = result['QC_flag'].unique()
    assert 'OK' in qc_flags
    assert len(qc_flags) > 1  # Should have some non-OK flags


def test_clean_sewer_timeseries_preserves_columns():
    """Test that original columns are preserved."""
    dates = pd.date_range('2023-01-01', periods=50, freq='15min')
    data = pd.DataFrame({
        'DateTime': dates,
        'Flow_MGD': np.random.uniform(0.5, 2.0, 50),
        'Meter': ['TEST'] * 50,
        'Source_File': ['test.csv'] * 50
    })
    
    result = clean_sewer_timeseries(data, flow_col='Flow_MGD', freq='15min', interp_limit=4)
    
    # Check required columns exist
    required_cols = ['DateTime', 'Flow_MGD', 'Meter', 'Source_File', 'QC_flag']
    for col in required_cols:
        assert col in result.columns


# Edge cases
def test_empty_dataframe():
    """Test handling of empty dataframe."""
    df = pd.DataFrame(columns=['DateTime', 'Flow_MGD', 'Meter', 'Source_File'])
    result = clean_sewer_timeseries(df, flow_col='Flow_MGD', freq='15min', interp_limit=4)
    assert len(result) == 0


def test_single_row_dataframe():
    """Test handling of single row."""
    df = pd.DataFrame({
        'DateTime': [pd.Timestamp('2023-01-01')],
        'Flow_MGD': [1.0],
        'Meter': ['TEST'],
        'Source_File': ['test.csv']
    })
    result = clean_sewer_timeseries(df, flow_col='Flow_MGD', freq='15min', interp_limit=4)
    assert len(result) >= 1
    assert 'QC_flag' in result.columns


def test_all_nan_values():
    """Test handling when all values are NaN."""
    dates = pd.date_range('2023-01-01', periods=50, freq='15min')
    df = pd.DataFrame({
        'DateTime': dates,
        'Flow_MGD': [np.nan] * 50,
        'Meter': ['TEST'] * 50,
        'Source_File': ['test.csv'] * 50
    })
    result = clean_sewer_timeseries(df, flow_col='Flow_MGD', freq='15min', interp_limit=4)
    assert all(result['QC_flag'] == 'MISSING')

def test_clean_sewer_timeseries_preserves_meter_groups(multi_meter_data):
    """
    Cleaning should not create or drop meter groups.
    """
    original_meters = set(multi_meter_data['Meter'].unique())
    original_count = multi_meter_data['Meter'].nunique()

    result = clean_sewer_timeseries(
        multi_meter_data,
        flow_col='Flow_MGD',
        freq='15min',
        interp_limit=4
    )

    cleaned_meters = set(result['Meter'].unique())
    cleaned_count = result['Meter'].nunique()

    #  Same number of meters
    assert cleaned_count == original_count

    # Same meter identities (no None, no extras)
    assert cleaned_meters == original_meters

    #  No missing meter values introduced
    assert not result['Meter'].isna().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])