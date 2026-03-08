import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rdii.utils import load_config

# Typical event segmentation dimensions:

# Onset: sustained positive deviation from baseflow
# Termination: return to baseflow envelope
# Minimum inter-event dry time
# Shape continuity (avoid splitting long wet responses)

@dataclass
class EventConfig:
    """
    Configuration parameters controlling event detection behavior.
    All time-based values should be expressed in number of timesteps,
    not hours, to keep logic discrete.
    """

    # --- Onset detection ---
    # Deviation threshold options:
    # Absolute
    # Fraction of Baseline
    # Sigma-based

    # DEVIATION THRESHOLD
    # Absolute MGD value above which a timestep is considered "elevated".
    deviation_threshold: float 

    # SUSTAIN DURATION  
    # Number of consecutive timesteps that must exceed deviation_threshold
    # before an onset is confirmed. Prevents triggering on a single spike
    # or brief transient.   
    # Recommendation: 3-6 timesteps 
    sustain_duration: int         

    # --- Termination detection ---

    # RETURN THRESHOLD
    # Residual value below which a timestep is considered "recovered".
    # Must be <= deviation_threshold to create hysteresis -- this is intentional.
    # Recommendation: 0.3-0.5 * deviation_threshold, or a value near zero
    return_threshold: float         
    
    # TERMINATION DURATION
    # Number of consecutive timesteps below return_threshold required to
    # officially end an event. Prevents premature termination during
    # multi-day events with slow recessions.
    # Recommendation: 8-16 timesteps (2-4 hr).
    termination_duration: int    

    # --- Inter-event handling ---

    # MIN INTER-EVENT TIME
    # Minimum dry gap (in timesteps) required to treat two events as separate.
    # Recommendation: 16-32 timesteps (4-8 hr).
    min_inter_event_time: int   

    # MERGE GAP DURATION
    # Maximum gap length eligible for conditional merging (see merge_close_events).
    # Recommendation: 24-48 timesteps (6-12 hr).
    merge_gap_duration: int      

    # --- Shape continuity ---

    # MAX INTERNAL GAP
    # Number of consecutive below-threshold timesteps tolerated *inside* a
    # growing event before triggering the termination check
    # Recommendation: 4-8 timesteps (1-2 hr).
    max_internal_gap: int  

    # ALLOW MULTI-PEAK
    # If True, events are not split at secondary peaks -- the event continues
    # growing as long as flow stays elevated. Almost always True for sewers
    # where a single storm can produce multiple flow peaks.
    allow_multi_peak: bool 
   
    # --- Credibility screening ---
    
    # MIN EVENT DURATION
    # Events shorter than this (in timesteps) are rejected as spurious.
    # Catches brief sensor spikes that survived the sustain_duration check.
    # Recommendation: 8-12 timesteps (2-3 hr).
    min_event_duration: int 

    # MIN PEAK EXCESS
    # Minimum peak residual (MGD) an event must reach to be considered real.
    # Acts as a noise floor -- even if duration is long, a very flat, low
    # residual event is probably not a wet-weather response.
    # Recommendation: equal to or slightly above deviation_threshold.
    min_peak_excess: float

def define_events(
    flow_series,
    baseline_series,
    config: EventConfig
):
    """
    Segment a flow time series into wet-weather response events.

    Parameters
    ----------
    flow_series : pd.Series
        Observed sewer flow (MGD), datetime-indexed.
    baseline_series : pd.Series
        Estimated baseline flow, same index as flow_series.
    config : EventConfig

    Returns
    -------
    events : list[dict]
        Credible, merged event records from build_event_record().
    diagnostics : dict
        QA/QC metadata from event_diagnostics().
    """
        
    residual   = (flow_series - baseline_series).values.astype(float)
    flow       = flow_series.values.astype(float)
    baseline   = baseline_series.values.astype(float)
    timestamps = flow_series.index


    # 1. Detect candidate onsets
    onset_indices = detect_candidate_onsets(residual, config)
    print(f"  Candidate onsets: {len(onset_indices)}")

    # 2. Grow each onset into an event; skip indices already claimed
    raw_events: list[dict] = []
    claimed_up_to = -1

    for onset_idx in onset_indices:
        if onset_idx <= claimed_up_to:
            continue

        end_idx, meta = grow_event(onset_idx, residual, flow, baseline, config)

        raw_events.append(build_event_record(
            start_time=timestamps[onset_idx],
            end_time=timestamps[end_idx],
            start_idx=onset_idx,
            end_idx=end_idx,
            flow=flow,
            baseline=baseline,
            residual=residual,
        ))
        claimed_up_to = end_idx

    print(f"  Raw events grown: {len(raw_events)}")

    # 3. Credibility screen before merging
    credible_events = credible_event(raw_events, config)
    print(f"  Credible events: {len(credible_events)}")

    # 4. Merge close events
    merged_events = merge_close_events(credible_events, residual, timestamps, flow, baseline, config)
    print(f"  Events after merging: {len(merged_events)}")

    # 5. Diagnostics
    diagnostics = event_diagnostics(merged_events, flow, baseline, residual)

    return merged_events, diagnostics



def detect_candidate_onsets(residual, config):
    """
    Return indices where a sustained positive deviation begins.

    An onset is confirmed when residual[t] > deviation_threshold holds
    for sustain_duration consecutive timesteps starting at t.
    """
    n = len(residual)
    onsets = []
    t = 0

    while t < n - config.sustain_duration:
        if residual[t] > config.deviation_threshold:
            window = residual[t : t + config.sustain_duration]
            if np.all(window > config.deviation_threshold):
                onsets.append(t)
                # Advance past the entire elevated block, not just sustain_duration
                while t < n and residual[t] > config.deviation_threshold:
                    t += 1
                continue
        t += 1

    return onsets

def grow_event(
    start_idx,
    residual,
    flow,
    baseline,
    config
):
    """
    Extend an event forward from start_idx until termination is confirmed.

    Internal dips up to max_internal_gap are tolerated. Once the gap exceeds
    that, check_termination() is called. If confirmed, the event ends at the
    last elevated timestep (gap_start - 1), not at current_idx.
    """
    n = len(residual)
    current_idx       = start_idx
    internal_gap_counter = 0
    peak_idx          = start_idx
    peak_excess       = residual[start_idx]
    end_idx           = start_idx  # fallback

    while current_idx < n:
        r = residual[current_idx]

        if r > config.return_threshold:
            internal_gap_counter = 0
            if r > peak_excess:
                peak_excess = r
                peak_idx    = current_idx
            end_idx = current_idx  # keep updating last confirmed elevated point

        else:
            internal_gap_counter += 1

            if internal_gap_counter <= config.max_internal_gap:
                # small dip -- stay in event
                pass

            else:
                gap_start  = current_idx - internal_gap_counter + 1
                window     = residual[current_idx : current_idx + config.termination_duration]

                if check_termination(window, config):
                    end_idx = gap_start - 1  # back-place end before the dry tail
                    break

        current_idx += 1

    meta = {
        'peak_idx':    peak_idx,
        'peak_excess': peak_excess,
    }

    return end_idx, meta

def check_termination(residual_window: np.ndarray, config: EventConfig) -> bool:
    """
    Return True if residual_window contains termination_duration consecutive
    timesteps all below return_threshold.
    """
    if len(residual_window) < config.termination_duration:
        return False
    return bool(np.all(residual_window[:config.termination_duration] < config.return_threshold))




def merge_close_events(
    events: list[dict],
    residual: np.ndarray,
    timestamps: pd.DatetimeIndex,
    flow: np.ndarray,
    baseline: np.ndarray,
    config: EventConfig,
) -> list[dict]:
    """
    Merge adjacent credible events separated by short dry gaps.

    Rules (applied left to right):
      - gap > merge_gap_duration          -> always keep separate
      - gap < min_inter_event_time        -> always merge
      - otherwise                         -> merge if max residual in
                                             gap > return_threshold
                                             (flow never fully recovered)
    """
    if len(events) <= 1:
        return events

    events = sorted(events, key=lambda e: e['start_idx'])
    merged = [events[0]]

    for ev in events[1:]:
        prev = merged[-1]
        gap  = ev['start_idx'] - prev['end_idx']

        should_merge = False

        if gap < config.min_inter_event_time:
            should_merge = True
        elif gap <= config.merge_gap_duration:
            gap_residual = residual[prev['end_idx'] : ev['start_idx']]
            if len(gap_residual) > 0 and np.max(gap_residual) > config.return_threshold:
                should_merge = True  # flow never fully recovered

        if should_merge:
            merged[-1] = build_event_record(
                start_time=prev['start_time'],
                end_time=ev['end_time'],
                start_idx=prev['start_idx'],
                end_idx=ev['end_idx'],
                flow=flow,
                baseline=baseline,
                residual=residual,
                timestamps=timestamps,
            )
        else:
            merged.append(ev)

    return merged

def credible_event(events: list[dict], config: EventConfig) -> list[dict]:
    """
    Reject events that are too short or whose peak is too small.
    Both criteria must pass.
    """
    credible, rejected = [], []

    for ev in events:
        if ev['duration'] < config.min_event_duration:
            rejected.append(ev)
        elif ev['peak_excess'] < config.min_peak_excess:
            rejected.append(ev)
        else:
            credible.append(ev)

    if rejected:
        print(f"  Rejected {len(rejected)} non-credible events")

    return credible


def build_event_record(
    start_time,
    end_time,
    start_idx: int,
    end_idx: int,
    flow: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
    timestamps: pd.DatetimeIndex = None,
) -> dict:
    """
    Build a dict representing a single event.

    Volume is computed as sum(residual) * timestep_hours.
    Negative residual values inside the event window are clipped to 0
    for volume and mean_excess, but stored raw in peak_excess.

    Fields
    ------
    start_time, end_time    : datetime boundaries
    start_idx, end_idx      : integer positions into flow/residual arrays
    duration                : timesteps (end_idx - start_idx)
    peak_excess             : max residual inside event (MGD)
    peak_time               : datetime at peak (if timestamps provided)
    mean_excess             : mean of clipped residual inside event (MGD)
    total_volume_MG         : excess volume in MG (assumes 15-min timesteps)
    baseline_mean           : mean baseline flow during event (MGD)
    """
    window_residual = residual[start_idx : end_idx + 1]
    window_flow     = flow[start_idx : end_idx + 1]
    window_baseline = baseline[start_idx : end_idx + 1]
    clipped         = np.clip(window_residual, 0, None)

    peak_local_idx = int(np.argmax(window_residual))
    peak_time      = timestamps[start_idx + peak_local_idx] if timestamps is not None else None

    TIMESTEP_HOURS = 0.25  # 15-min data; adjust if needed

    return {
        'start_time':     start_time,
        'end_time':       end_time,
        'start_idx':      start_idx,
        'end_idx':        end_idx,
        'duration':       end_idx - start_idx,
        'peak_excess':    float(np.max(window_residual)),
        'peak_time':      peak_time,
        'mean_excess':    float(np.mean(clipped)),
        'total_volume_MG': float(np.sum(clipped) * TIMESTEP_HOURS),
        'baseline_mean':  float(np.mean(window_baseline)),
    }


def event_diagnostics(
    events: list[dict],
    flow: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
) -> dict:
    """
    QA/QC checks on the final event list.

    Checks
    ------
    - Negative mean excess (possible baseline overestimation)
    - Unusually long events (> 7 days)
    - Overlapping events (indicates a bug in grow or merge logic)
    """
    LONG_EVENT_THRESHOLD = 7 * 96  # 7 days at 15-min resolution

    negative_rdii, long_events, overlapping = [], [], []

    for i, ev in enumerate(events):
        if ev['mean_excess'] < 0:
            negative_rdii.append(i)
        if ev['duration'] > LONG_EVENT_THRESHOLD:
            long_events.append(i)

    for i in range(len(events) - 1):
        if events[i + 1]['start_idx'] <= events[i]['end_idx']:
            overlapping.append((i, i + 1))

    diagnostics = {
        'n_events':             len(events),
        'negative_rdii_events': negative_rdii,
        'long_events':          long_events,
        'overlapping_events':   overlapping,
    }

    print(f"\n  Diagnostics:")
    print(f"    Total events        : {diagnostics['n_events']}")
    print(f"    Negative RDII       : {len(negative_rdii)}")
    print(f"    Unusually long      : {len(long_events)}")
    print(f"    Overlapping (bugs)  : {len(overlapping)}")

    return diagnostics


def main(config_path: str = 'config.json'):
        
    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)

    # Setup paths
    project_root = Path(config['project_root']) if 'project_root' in config else Path(__file__).parent.parent.parent
    processed_dir = project_root / config['paths']['processed_data']
    cleaned_gwi = processed_dir / config['paths']['gwi_removed_filename']

    # Load data
    data = pd.read_csv(cleaned_gwi)
    print("Loading cleaned data ...")




if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    main(config_file)

