"""
Unit tests for event_stats.py
Run with: pytest tests/test_event_stats.py
"""

import numpy as np
import pandas as pd
import pytest

from rdii.event_stats import (
    axis_features,
    compute_event_stats,
    magnitude_stats,
    normalized_stats,
    quality_flags,
    shape_stats,
    slice_event_window,
    timestep_hours_from_frequency,
    timing_stats,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TIMESTEP_HOURS = 0.25  # 15-min
N = 200
TIMESTAMPS = pd.date_range("2023-01-01", periods=N, freq="15min")


def flat_baseline(n, val=1.0):
    return np.full(n, val)


def triangular_pulse(n, onset, duration, peak=0.5):
    signal = np.zeros(n)
    half = duration // 2
    for i in range(duration):
        signal[onset + i] = peak * (1 - abs(i - half) / half)
    return signal


def make_event(start, end):
    return {
        'start_idx':  start,
        'end_idx':    end,
        'start_time': TIMESTAMPS[start],
        'end_time':   TIMESTAMPS[end],
        'Meter':      'TEST',
    }


# ---------------------------------------------------------------------------
# timestep_hours_from_frequency
# ---------------------------------------------------------------------------

def test_timestep_15min():
    assert timestep_hours_from_frequency('15min') == pytest.approx(0.25)

def test_timestep_30min():
    assert timestep_hours_from_frequency('30min') == pytest.approx(0.5)

def test_timestep_1h():
    assert timestep_hours_from_frequency('1h') == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# slice_event_window
# ---------------------------------------------------------------------------

def test_slice_event_window_lengths():
    flow     = flat_baseline(N)
    baseline = flat_baseline(N)
    residual = np.zeros(N)
    ev       = make_event(10, 30)

    flow_e, baseline_e, residual_e, time_e = slice_event_window(
        ev, flow, baseline, residual, TIMESTAMPS
    )

    expected_len = 30 - 10 + 1  # end_idx inclusive
    assert len(flow_e)     == expected_len
    assert len(baseline_e) == expected_len
    assert len(residual_e) == expected_len
    assert len(time_e)     == expected_len


def test_slice_event_window_no_timestamps():
    flow     = flat_baseline(N)
    baseline = flat_baseline(N)
    residual = np.zeros(N)
    ev       = make_event(5, 15)

    _, _, _, time_e = slice_event_window(ev, flow, baseline, residual, timestamps=None)

    assert isinstance(time_e, np.ndarray)
    assert time_e[0] == 5
    assert time_e[-1] == 15


# ---------------------------------------------------------------------------
# timing_stats
# ---------------------------------------------------------------------------

def test_timing_stats_duration():
    residual = np.zeros(40)
    residual[20] = 1.0   # peak at index 20
    time_e   = TIMESTAMPS[:40]

    stats = timing_stats(time_e, residual, TIMESTEP_HOURS)

    assert stats['duration_hrs']       == pytest.approx(40 * 0.25)
    assert stats['time_to_peak_hrs']   == pytest.approx(20 * 0.25)
    assert stats['rise_time_hrs']      == pytest.approx(20 * 0.25)
    assert stats['recession_time_hrs'] == pytest.approx(19 * 0.25)
    assert stats['peak_timestep']      == 20


def test_timing_stats_peak_at_start():
    residual    = np.zeros(20)
    residual[0] = 1.0
    stats       = timing_stats(TIMESTAMPS[:20], residual, TIMESTEP_HOURS)

    assert stats['time_to_peak_hrs']   == pytest.approx(0.0)
    assert stats['recession_time_hrs'] == pytest.approx(19 * 0.25)


def test_timing_stats_peak_at_end():
    residual     = np.zeros(20)
    residual[-1] = 1.0
    stats        = timing_stats(TIMESTAMPS[:20], residual, TIMESTEP_HOURS)

    assert stats['time_to_peak_hrs']   == pytest.approx(19 * 0.25)
    assert stats['recession_time_hrs'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# magnitude_stats
# ---------------------------------------------------------------------------

def test_magnitude_stats_basic():
    residual = np.full(40, 0.2)
    baseline = flat_baseline(40, 1.0)
    flow     = baseline + residual

    stats = magnitude_stats(flow, baseline, residual, TIMESTEP_HOURS)

    assert stats['peak_flow']       == pytest.approx(1.2)
    assert stats['peak_excess']     == pytest.approx(0.2)
    assert stats['mean_excess']     == pytest.approx(0.2)
    assert stats['total_volume_MG'] == pytest.approx(40 * 0.2 * 0.25)
    assert stats['baseline_mean']   == pytest.approx(1.0)


def test_magnitude_stats_clips_negatives():
    """Negative residuals should not contribute to volume or mean_excess."""
    residual = np.array([-0.1] * 20 + [0.3] * 20)
    baseline = flat_baseline(40, 1.0)
    flow     = baseline + residual

    stats = magnitude_stats(flow, baseline, residual, TIMESTEP_HOURS)

    assert stats['mean_excess']     == pytest.approx(0.3 * 20 / 40)
    assert stats['total_volume_MG'] == pytest.approx(20 * 0.3 * 0.25)
    assert stats['peak_excess']     == pytest.approx(0.3)  # raw, not clipped


# ---------------------------------------------------------------------------
# shape_stats
# ---------------------------------------------------------------------------

def test_shape_stats_single_peak():
    residual = triangular_pulse(60, 0, 60, peak=0.5)
    stats    = shape_stats(residual, TIMESTEP_HOURS)

    assert stats['peak_count'] >= 1
    assert not np.isnan(stats['rise_slope'])
    assert not np.isnan(stats['recession_slope'])


def test_shape_stats_rise_slope():
    """Peak at index 10 with value 1.0 -> rise_slope = 1.0 / (10 * 0.25) = 0.4 MGD/hr."""
    residual     = np.zeros(40)
    residual[10] = 1.0
    stats        = shape_stats(residual, TIMESTEP_HOURS)

    assert stats['rise_slope'] == pytest.approx(1.0 / (10 * 0.25))


def test_shape_stats_peak_at_start_rise_slope_nan():
    """Peak at index 0 means no rise time -- rise_slope should be NaN."""
    residual    = np.zeros(20)
    residual[0] = 1.0
    stats       = shape_stats(residual, TIMESTEP_HOURS)

    assert np.isnan(stats['rise_slope'])


def test_shape_stats_multi_peak():
    residual        = np.zeros(80)
    residual[10]    = 0.4
    residual[11:19] = 0.1
    residual[20]    = 0.5
    stats           = shape_stats(residual, TIMESTEP_HOURS)

    assert stats['peak_count'] >= 2


# ---------------------------------------------------------------------------
# normalized_stats
# ---------------------------------------------------------------------------

def test_normalized_stats_peak_to_base():
    residual = np.full(40, 0.5)
    baseline = flat_baseline(40, 1.0)
    flow     = baseline + residual

    stats = normalized_stats(flow, baseline, residual, TIMESTEP_HOURS)

    assert stats['peak_to_base_ratio'] == pytest.approx(0.5 / 1.0)


def test_normalized_stats_zero_baseline():
    """Zero baseline should return NaN for ratio-based metrics."""
    residual = np.full(20, 0.3)
    baseline = flat_baseline(20, 0.0)
    flow     = residual.copy()

    stats = normalized_stats(flow, baseline, residual, TIMESTEP_HOURS)

    assert np.isnan(stats['peak_to_base_ratio'])
    assert np.isnan(stats['rdii_ratio'])


def test_normalized_stats_volume_to_peak():
    """Flat residual of 0.2 for 40 steps -> volume = 40*0.2*0.25 = 2.0 MG, peak = 0.2 -> ratio = 10.0."""
    residual = np.full(40, 0.2)
    baseline = flat_baseline(40, 1.0)
    flow     = baseline + residual

    stats = normalized_stats(flow, baseline, residual, TIMESTEP_HOURS)

    assert stats['volume_to_peak_ratio'] == pytest.approx(2.0 / 0.2)


# ---------------------------------------------------------------------------
# axis_features
# ---------------------------------------------------------------------------

def test_axis_features_pulls_from_stats_row():
    row = {
        'rise_slope':           0.8,
        'volume_to_peak_ratio': 5.0,
        'peak_to_base_ratio':   0.4,
    }
    axes = axis_features(row)

    assert axes['axis_speed']       == pytest.approx(0.8)
    assert axes['axis_persistence'] == pytest.approx(5.0)
    assert axes['axis_intensity']   == pytest.approx(0.4)


def test_axis_features_missing_keys_return_nan():
    axes = axis_features({})

    assert np.isnan(axes['axis_speed'])
    assert np.isnan(axes['axis_persistence'])
    assert np.isnan(axes['axis_intensity'])


# ---------------------------------------------------------------------------
# quality_flags
# ---------------------------------------------------------------------------

def test_quality_flags_no_negatives():
    residual = np.full(40, 0.2)
    baseline = flat_baseline(40, 1.0)
    flow     = baseline + residual

    flags = quality_flags(flow, baseline, residual)

    assert flags['negative_residual_fraction'] == pytest.approx(0.0)


def test_quality_flags_all_negative():
    residual = np.full(40, -0.1)
    baseline = flat_baseline(40, 1.0)
    flow     = baseline + residual

    flags = quality_flags(flow, baseline, residual)

    assert flags['negative_residual_fraction'] == pytest.approx(1.0)


def test_quality_flags_stable_baseline():
    """Flat baseline -> drift indicator should be 0."""
    residual = np.full(40, 0.2)
    baseline = flat_baseline(40, 1.0)
    flow     = baseline + residual

    flags = quality_flags(flow, baseline, residual)

    assert flags['baseline_drift_indicator'] == pytest.approx(0.0)


def test_quality_flags_flat_event():
    """Perfectly flat residual -> std = 0 -> flat_event_indicator = 0."""
    residual = np.full(40, 0.3)
    baseline = flat_baseline(40, 1.0)
    flow     = baseline + residual

    flags = quality_flags(flow, baseline, residual)

    assert flags['flat_event_indicator'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_event_stats (end-to-end)
# ---------------------------------------------------------------------------

def test_compute_event_stats_returns_dataframe():
    baseline = flat_baseline(N)
    signal   = triangular_pulse(N, 20, 60, peak=0.4)
    flow     = baseline + signal
    residual = flow - baseline

    events = [make_event(20, 79)]

    df = compute_event_stats(events, flow, baseline, residual, TIMESTAMPS, TIMESTEP_HOURS)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_compute_event_stats_expected_columns():
    baseline = flat_baseline(N)
    signal   = triangular_pulse(N, 20, 60, peak=0.4)
    flow     = baseline + signal
    residual = flow - baseline

    events = [make_event(20, 79)]
    df     = compute_event_stats(events, flow, baseline, residual, TIMESTAMPS, TIMESTEP_HOURS)

    expected = [
        'start_time', 'end_time', 'Meter',
        'duration_hrs', 'peak_excess', 'total_volume_MG',
        'rise_slope', 'recession_slope', 'skewness',
        'peak_to_base_ratio', 'rdii_ratio',
        'axis_speed', 'axis_persistence', 'axis_intensity',
        'negative_residual_fraction',
    ]
    for col in expected:
        assert col in df.columns, f"Missing column: {col}"


def test_compute_event_stats_multiple_meters():
    baseline = flat_baseline(N)
    signal   = triangular_pulse(N, 20, 40, peak=0.3)
    flow     = baseline + signal
    residual = flow - baseline

    events = [
        {**make_event(20, 59), 'Meter': 'M1'},
        {**make_event(20, 59), 'Meter': 'M2'},
    ]

    df = compute_event_stats(events, flow, baseline, residual, TIMESTAMPS, TIMESTEP_HOURS)

    assert len(df) == 2
    assert set(df['Meter']) == {'M1', 'M2'}


def test_compute_event_stats_residual_computed_if_missing():
    """If residual is not passed, it should be computed internally."""
    baseline = flat_baseline(N)
    signal   = triangular_pulse(N, 10, 40, peak=0.3)
    flow     = baseline + signal

    events = [make_event(10, 49)]
    df     = compute_event_stats(events, flow, baseline, residual=None,
                                  timestamps=TIMESTAMPS, timestep_hours=TIMESTEP_HOURS)

    assert len(df) == 1
    assert df['peak_excess'].iloc[0] > 0