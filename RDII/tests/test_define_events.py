"""
Unit tests for events.py
Run with: python test_events.py
"""

import numpy as np
import pandas as pd

from rdii.define_event import (
    EventConfig,
    build_event_record,
    check_termination,
    credible_event,
    define_events,
    detect_candidate_onsets,
    grow_event,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

cfg = EventConfig(
    deviation_threshold  = 0.10,
    sustain_duration     = 4,
    return_threshold     = 0.04,
    termination_duration = 8,
    min_inter_event_time = 16,
    merge_gap_duration   = 24,
    max_internal_gap     = 6,
    min_event_duration   = 8,
    min_peak_excess      = 0.10,
    allow_multi_peak = True
)

timestamps = pd.date_range("2023-01-01", periods=960, freq="15min")
n          = len(timestamps)

def flat_baseline(n, val=1.0):
    return np.full(n, val)

def make_event_signal(n, onset, duration, peak=0.5):
    """Triangular pulse centered in the event window."""
    signal = np.zeros(n)
    half   = duration // 2
    for i in range(duration):
        signal[onset + i] = peak * (1 - abs(i - half) / half)
    return signal


# ---------------------------------------------------------------------------
# detect_candidate_onsets
# ---------------------------------------------------------------------------

def test_detect_candidate_onsets_basic():
    residual          = np.zeros(n)
    residual[100:120] = 0.20
    residual[300:330] = 0.15

    onsets = detect_candidate_onsets(residual, cfg)

    assert len(onsets) == 2,  f"Expected 2 onsets, got {len(onsets)}"
    assert onsets[0] == 100,  f"Expected first onset at 100, got {onsets[0]}"
    assert onsets[1] == 300,  f"Expected second onset at 300, got {onsets[1]}"


def test_detect_candidate_onsets_no_signal():
    residual = np.zeros(n)
    assert len(detect_candidate_onsets(residual, cfg)) == 0


def test_detect_candidate_onsets_below_threshold():
    residual        = np.zeros(n)
    residual[50:80] = cfg.deviation_threshold * 0.5
    assert len(detect_candidate_onsets(residual, cfg)) == 0


def test_detect_candidate_onsets_too_short():
    """Elevation shorter than sustain_duration should not trigger."""
    residual        = np.zeros(n)
    residual[50:52] = 0.20   # 2 steps < sustain_duration=4
    assert len(detect_candidate_onsets(residual, cfg)) == 0


# ---------------------------------------------------------------------------
# check_termination
# ---------------------------------------------------------------------------

def test_check_termination_confirms():
    window = np.full(20, 0.01)
    assert check_termination(window, cfg) == True


def test_check_termination_spike_in_window():
    window    = np.full(20, 0.01)
    window[3] = 0.20
    assert check_termination(window, cfg) == False


def test_check_termination_too_short():
    window = np.array([0.01, 0.01])   # shorter than termination_duration
    assert check_termination(window, cfg) == False


# ---------------------------------------------------------------------------
# grow_event
# ---------------------------------------------------------------------------

def test_grow_event_basic():
    residual          = np.zeros(n)
    residual[100:160] = 0.30
    flow              = flat_baseline(n) + residual
    baseline          = flat_baseline(n)

    end_idx, meta = grow_event(100, residual, flow, baseline, cfg)

    assert end_idx >= 150,           f"Event ended too early: {end_idx}"
    assert meta['peak_excess'] == 0.30
    assert meta['peak_idx']    >= 100


def test_grow_event_tolerates_internal_dip():
    """A dip within max_internal_gap should not split the event."""
    residual          = np.zeros(n)
    residual[100:140] = 0.30
    residual[140:144] = 0.01   # 4-step dip, max_internal_gap=6
    residual[144:180] = 0.30
    flow              = flat_baseline(n) + residual
    baseline          = flat_baseline(n)

    end_idx, _ = grow_event(100, residual, flow, baseline, cfg)
    assert end_idx >= 170, f"Event split at internal dip, ended at {end_idx}"


def test_grow_event_terminates_after_long_gap():
    """Residual staying at 0 well past max_internal_gap should end the event."""
    residual          = np.zeros(n)
    residual[100:130] = 0.30
    flow              = flat_baseline(n) + residual
    baseline          = flat_baseline(n)

    end_idx, _ = grow_event(100, residual, flow, baseline, cfg)
    assert end_idx < 140, f"Event should have terminated, but end_idx={end_idx}"


# ---------------------------------------------------------------------------
# build_event_record
# ---------------------------------------------------------------------------

def test_build_event_record_fields():
    residual         = np.zeros(n)
    residual[50:100] = 0.20
    flow             = flat_baseline(n) + residual
    baseline         = flat_baseline(n)

    rec = build_event_record(
        start_time=timestamps[50], end_time=timestamps[99],
        start_idx=50, end_idx=99,
        flow=flow, baseline=baseline, residual=residual, timestamps=timestamps,
    )

    assert rec['duration']                == 49
    assert abs(rec['peak_excess'] - 0.20) <  1e-6
    assert rec['total_volume_MG']         >  0
    assert rec['mean_excess']             >  0
    assert rec['baseline_mean']           == 1.0


def test_build_event_record_volume():
    """40 steps * 0.10 MGD * 0.25 hr/step = 1.0 MG."""
    residual       = np.zeros(n)
    residual[0:40] = 0.10
    flow           = flat_baseline(n) + residual
    baseline       = flat_baseline(n)

    rec = build_event_record(
        start_time=timestamps[0], end_time=timestamps[39],
        start_idx=0, end_idx=39,
        flow=flow, baseline=baseline, residual=residual, timestamps=timestamps,
    )

    assert abs(rec['total_volume_MG'] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# credible_event
# ---------------------------------------------------------------------------

def test_credible_event_filters_short():
    assert len(credible_event([{'duration': 3, 'peak_excess': 0.30}], cfg)) == 0


def test_credible_event_filters_low_peak():
    assert len(credible_event([{'duration': 20, 'peak_excess': 0.02}], cfg)) == 0


def test_credible_event_passes():
    assert len(credible_event([{'duration': 20, 'peak_excess': 0.25}], cfg)) == 1


def test_credible_event_mixed():
    events = [
        {'duration': 20, 'peak_excess': 0.25},  # credible
        {'duration': 3,  'peak_excess': 0.25},  # too short
        {'duration': 20, 'peak_excess': 0.02},  # peak too small
    ]
    result = credible_event(events, cfg)
    assert len(result) == 1
    assert result[0]['peak_excess'] == 0.25


# ---------------------------------------------------------------------------
# define_events (end-to-end)
# ---------------------------------------------------------------------------

def test_define_events_two_events():
    baseline_series = pd.Series(flat_baseline(n), index=timestamps)
    signal          = make_event_signal(n, onset=100, duration=60, peak=0.40)
    signal         += make_event_signal(n, onset=500, duration=60, peak=0.35)
    flow_series     = baseline_series + signal

    events, diag = define_events(flow_series, baseline_series, cfg)

    assert diag['n_events'] >= 1
    assert len(diag['overlapping_events']) == 0


def test_define_events_no_signal():
    baseline_series = pd.Series(flat_baseline(n), index=timestamps)
    events, diag    = define_events(baseline_series.copy(), baseline_series, cfg)
    assert diag['n_events'] == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_detect_candidate_onsets_basic,
        test_detect_candidate_onsets_no_signal,
        test_detect_candidate_onsets_below_threshold,
        test_detect_candidate_onsets_too_short,
        test_check_termination_confirms,
        test_check_termination_spike_in_window,
        test_check_termination_too_short,
        test_grow_event_basic,
        test_grow_event_tolerates_internal_dip,
        test_grow_event_terminates_after_long_gap,
        test_build_event_record_fields,
        test_build_event_record_volume,
        test_credible_event_filters_short,
        test_credible_event_filters_low_peak,
        test_credible_event_passes,
        test_credible_event_mixed,
        test_define_events_two_events,
        test_define_events_no_signal,
    ]

    passed, failed = 0, []

    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed.append(test.__name__)

    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("Failed:", failed)