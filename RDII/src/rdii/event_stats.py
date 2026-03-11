"""
event_stats.py

Purpose:
    Compute response statistics for pre-defined flow events.
    These statistics support:
        - event classification
        - RDII estimation
        - cross-meter comparison

Assumptions:
    - events is a list of dicts with start_idx, end_idx, start_time, end_time
    - flow, baseline, residual are full-length numpy arrays aligned to the same index
    - timestep_hours is derived from config['cleaning']['frequency'] via
      timestep_hours_from_frequency(), not hardcoded
    - residual = flow - baseline (computed here if not provided)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew

from rdii.utils import load_config

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def timestep_hours_from_frequency(frequency: str) -> float:
    """
    Convert a pandas frequency string to decimal hours.

    Examples
    --------
    '15min' -> 0.25
    '30min' -> 0.5
    '1h'    -> 1.0
    '1T'    -> 1/60
    """
    return pd.tseries.frequencies.to_offset(frequency).nanos / 3.6e12


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_event_stats(
    events: list[dict],
    flow: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray = None,
    timestamps: pd.DatetimeIndex = None,
    timestep_hours: float = 0.25,
) -> pd.DataFrame:
    """
    Compute statistics for each event and return as a DataFrame.

    Parameters
    ----------
    events : list[dict]
        Event catalog from define_events(). Each dict must have
        start_idx, end_idx, start_time, end_time.
    flow : np.ndarray
        Raw observed flow (full series).
    baseline : np.ndarray
        Baseline flow (full series).
    residual : np.ndarray, optional
        Precomputed residual. Computed as flow - baseline if not provided.
    timestamps : pd.DatetimeIndex, optional
        Full datetime index aligned to flow/baseline/residual.
    timestep_hours : float
        Hours per timestep. Derive from config via
        timestep_hours_from_frequency(config['cleaning']['frequency']).

    Returns
    -------
    pd.DataFrame
        One row per event with all computed statistics.
    """
    if residual is None:
        residual = flow - baseline

    rows = []

    for ev in events:
        flow_e, baseline_e, residual_e, time_e = slice_event_window(
            ev, flow, baseline, residual, timestamps
        )

        row = {
            'start_time': ev['start_time'],
            'end_time':   ev['end_time'],
            'start_idx':  ev['start_idx'],
            'end_idx':    ev['end_idx'],
            'Meter':      ev.get('Meter', None),
        }

        row.update(timing_stats(time_e, residual_e, timestep_hours))
        row.update(magnitude_stats(flow_e, baseline_e, residual_e, timestep_hours))
        row.update(shape_stats(residual_e, timestep_hours))
        row.update(normalized_stats(flow_e, baseline_e, residual_e, timestep_hours))
        row.update(quality_flags(flow_e, baseline_e, residual_e))
        row.update(axis_features(row))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def slice_event_window(
    event: dict,
    flow: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    timestamps: pd.DatetimeIndex = None,
) -> tuple:
    """
    Extract time series slices for a single event.

    Returns
    -------
    flow_e, baseline_e, residual_e, time_e
        Sliced arrays. time_e is a DatetimeIndex slice if timestamps provided,
        otherwise an integer index array.
    """
    s = event['start_idx']
    e = event['end_idx'] + 1

    flow_e     = flow[s:e]
    baseline_e = baseline[s:e]
    residual_e = residual[s:e]
    time_e     = timestamps[s:e] if timestamps is not None else np.arange(s, e)

    return flow_e, baseline_e, residual_e, time_e


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timing_stats(time_index, residual: np.ndarray, timestep_hours: float) -> dict:
    """
    Compute timing-related statistics for a single event window.

    Returns
    -------
    dict with keys:
        duration_hrs        total event duration in hours
        time_to_peak_hrs    time from onset to peak residual (hours)
        rise_time_hrs       time from onset to peak
        recession_time_hrs  time from peak to end of event (hours)
        peak_timestep       local index of peak within the event window
    """
    n          = len(residual)
    peak_local = int(np.argmax(residual))

    return {
        'duration_hrs':        n * timestep_hours,
        'time_to_peak_hrs':    peak_local * timestep_hours,
        'rise_time_hrs':       peak_local * timestep_hours,
        'recession_time_hrs':  (n - peak_local - 1) * timestep_hours,
        'peak_timestep':       peak_local,
    }


# ---------------------------------------------------------------------------
# Magnitude
# ---------------------------------------------------------------------------

def magnitude_stats(
    flow: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    timestep_hours: float,
) -> dict:
    """
    Compute magnitude-related statistics.

    Negative residual values are clipped to 0 for volume/mean calculations
    but stored raw for peak_excess.

    Returns
    -------
    dict with keys:
        peak_flow           max observed flow during event (MGD)
        peak_excess         max residual during event (MGD)
        mean_excess         mean of clipped residual (MGD)
        total_volume_MG     excess volume in million gallons
        baseline_mean       mean baseline flow during event (MGD)
    """
    clipped = np.clip(residual, 0, None)

    return {
        'peak_flow':       float(np.max(flow)),
        'peak_excess':     float(np.max(residual)),
        'mean_excess':     float(np.mean(clipped)),
        'total_volume_MG': float(np.sum(clipped) * timestep_hours),
        'baseline_mean':   float(np.mean(baseline)),
    }


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def shape_stats(residual: np.ndarray, timestep_hours: float) -> dict:
    """
    Describe event shape.

    Peak count uses scipy.signal.find_peaks with a prominence threshold of
    10% of peak residual to avoid counting noise.

    Rise slope      = peak_excess / rise_time_hrs      (MGD/hr)
    Recession slope = peak_excess / recession_time_hrs (MGD/hr, stored positive)

    Returns
    -------
    dict with keys:
        skewness            skewness of residual (positive = long right tail)
        rise_slope          MGD/hr from onset to peak
        recession_slope     MGD/hr from peak to end (positive)
        peak_count          number of distinct peaks detected
    """
    n          = len(residual)
    peak_local = int(np.argmax(residual))
    peak_val   = float(residual[peak_local])

    residual_skew      = float(skew(residual)) if n > 2 else 0.0
    rise_time_hrs      = peak_local * timestep_hours
    recession_time_hrs = (n - peak_local - 1) * timestep_hours

    rise_slope      = (peak_val / rise_time_hrs)      if rise_time_hrs > 0      else np.nan
    recession_slope = (peak_val / recession_time_hrs) if recession_time_hrs > 0 else np.nan

    prominence_threshold = 0.1 * peak_val if peak_val > 0 else 0.0
    peaks, _   = find_peaks(residual, prominence=prominence_threshold)
    peak_count = max(1, len(peaks))

    return {
        'skewness':        residual_skew,
        'rise_slope':      float(rise_slope)      if not np.isnan(rise_slope)      else np.nan,
        'recession_slope': float(recession_slope) if not np.isnan(recession_slope) else np.nan,
        'peak_count':      int(peak_count),
    }


# ---------------------------------------------------------------------------
# Normalized stats
# ---------------------------------------------------------------------------

def normalized_stats(
    flow: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    timestep_hours: float,
) -> dict:
    """
    Compute scale-free metrics for cross-meter comparison.

    peak_to_base_ratio   : peak_excess / baseline_mean
                           event size relative to meter scale
    volume_to_peak_ratio : total_volume_MG / peak_excess (hours)
                           shape descriptor -- wide flat events score high,
                           sharp spikes score low
    rdii_ratio           : total_volume / (baseline_mean * duration_hrs)
                           excess volume as fraction of baseline-equivalent volume

    Returns
    -------
    dict with keys:
        peak_to_base_ratio
        volume_to_peak_ratio
        rdii_ratio
    """
    clipped       = np.clip(residual, 0, None)
    baseline_mean = float(np.mean(baseline))
    peak_excess   = float(np.max(residual))
    total_volume  = float(np.sum(clipped) * timestep_hours)
    duration_hrs  = len(residual) * timestep_hours

    peak_to_base   = (peak_excess   / baseline_mean)              if baseline_mean > 0 else np.nan
    vol_to_peak    = (total_volume  / peak_excess)                if peak_excess > 0   else np.nan
    rdii_ratio     = (total_volume  / (baseline_mean * duration_hrs)) if (baseline_mean > 0 and duration_hrs > 0) else np.nan

    return {
        'peak_to_base_ratio':   peak_to_base,
        'volume_to_peak_ratio': vol_to_peak,
        'rdii_ratio':           rdii_ratio,
    }


# ---------------------------------------------------------------------------
# Axis features (for classification)
# ---------------------------------------------------------------------------

def axis_features(stats_row: dict) -> dict:
    """
    Distill one representative feature per classification axis.

    Axes
    ----
    speed       : rise_slope (MGD/hr)
                  Fast-responding systems (direct inflow) have steep rises;
                  slow systems (infiltration) rise gradually.
    persistence : volume_to_peak_ratio (hrs)
                  Long, flat events score high; sharp spikes score low.
    intensity   : peak_to_base_ratio (dimensionless)
                  Normalizes event size for cross-meter comparison.

    Returns
    -------
    dict with keys: axis_speed, axis_persistence, axis_intensity
    """
    return {
        'axis_speed':       stats_row.get('rise_slope',           np.nan),
        'axis_persistence': stats_row.get('volume_to_peak_ratio', np.nan),
        'axis_intensity':   stats_row.get('peak_to_base_ratio',   np.nan),
    }


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

def quality_flags(
    flow: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
) -> dict:
    """
    Flag potential data quality issues.

    negative_residual_fraction : fraction of timesteps where residual < 0
                                 High values suggest baseline overestimation.
    baseline_drift_indicator   : std(baseline) / mean(baseline)
                                 High values mean baseline shifted during event,
                                 which distorts volume estimates.
    flat_event_indicator       : std(residual) / max(residual)
                                 Low values indicate a flat, possibly spurious event.

    Returns
    -------
    dict with keys:
        negative_residual_fraction  (float, 0-1)
        baseline_drift_indicator    (float >= 0)
        flat_event_indicator        (float >= 0)
    """
    n             = len(residual)
    baseline_mean = float(np.mean(baseline))
    peak_excess   = float(np.max(residual))

    neg_frac       = float(np.sum(residual < 0) / n) if n > 0        else np.nan
    baseline_drift = float(np.std(baseline) / baseline_mean)          if baseline_mean > 0 else np.nan
    flat_indicator = float(np.std(residual) / peak_excess)            if peak_excess > 0   else np.nan

    return {
        'negative_residual_fraction': neg_frac,
        'baseline_drift_indicator':   baseline_drift,
        'flat_event_indicator':       flat_indicator,
    }

def main(config_path: str = 'config.json'):

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)

    project_root  = Path(config['project_root']) if 'project_root' in config else Path(__file__).parent.parent.parent
    processed_dir = project_root / config['paths']['processed_data']
    bwf_file      = processed_dir / config['paths']['bwf_results_filename']
    events_file   = processed_dir / config['paths']['events_filename']

    try:
        bwf_data = pd.read_csv(bwf_file,    parse_dates=['DateTime'])
        events   = pd.read_csv(events_file, parse_dates=['start_time', 'end_time'])
        print(f"Loaded {len(bwf_data)} flow rows, {len(events)} events")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        sys.exit(1)

    timestep_hours = timestep_hours_from_frequency(config['cleaning']['frequency'])
    all_stats = []

    for meter_name, meter_bwf in bwf_data.groupby('Meter'):
        meter_events = events[events['Meter'] == meter_name]
        if meter_events.empty:
            print(f"  {meter_name}: no events, skipping")
            continue

        print(f"  {meter_name}: {len(meter_events)} events")

        meter_bwf  = meter_bwf.set_index('DateTime').sort_index()
        flow       = meter_bwf['Raw'].values.astype(float)
        baseline   = meter_bwf['BWF'].values.astype(float)
        residual   = flow - baseline
        timestamps = meter_bwf.index

        # Re-index event start/end indices against this meter's sorted index
        # since start_idx/end_idx in the CSV are relative to the full dataset
        dt_to_idx  = {dt: i for i, dt in enumerate(timestamps)}
        event_dicts = []
        for _, ev in meter_events.iterrows():
            start_idx = dt_to_idx.get(ev['start_time'])
            end_idx   = dt_to_idx.get(ev['end_time'])
            if start_idx is None or end_idx is None:
                print(f"    ✗ Could not align event {ev['start_time']} — skipping")
                continue
            event_dicts.append({**ev.to_dict(), 'start_idx': start_idx, 'end_idx': end_idx})

        stats = compute_event_stats(
            events=event_dicts,
            flow=flow,
            baseline=baseline,
            residual=residual,
            timestamps=timestamps,
            timestep_hours=timestep_hours,
        )
        all_stats.append(stats)

    if not all_stats:
        print("✗ No stats computed — check meter alignment")
        sys.exit(1)

    out_df   = pd.concat(all_stats, ignore_index=True)
    out_file = processed_dir / config['paths']['event_stats_filename']
    out_df.to_csv(out_file, index=False)
    print(f"\n✓ Saved stats for {len(out_df)} events to {out_file}")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    main(config_file)