
# Typical event segmentation dimensions:

# Onset: sustained positive deviation from baseflow
# Termination: return to baseflow envelope
# Minimum inter-event dry time
# Shape continuity (avoid splitting long wet responses)

class EventConfig:
    """
    Configuration parameters controlling event detection behavior.
    All time-based values should be expressed in number of timesteps,
    not hours, to keep logic discrete.
    """

    # --- Onset detection ---
    deviation_threshold      # e.g., absolute residual or % of baseline
    sustain_duration         # N consecutive timesteps above threshold

    # --- Termination detection ---
    return_threshold         # lower than deviation_threshold
    termination_duration    # N consecutive timesteps below return_threshold

    # --- Inter-event handling ---
    min_inter_event_time     # minimum dry time to consider events separate
    merge_gap_duration      # maximum gap eligible for merging

    # --- Shape continuity ---
    max_internal_gap        # small dips allowed *inside* an event
    allow_multi_peak        # almost always True for sewers

    # --- Credibility screening ---
    min_event_duration      # reject very short events
    min_peak_excess         # noise floor protection

def define_events(
    flow_series,
    baseline_series,
    config: EventConfig
):
    """
    Segment flow time series into response events.

    Parameters
    ----------
    flow_series :
        Observed sewer flow time series
    baseline_series :
        Estimated baseline (sanitary + slow infiltration)
    config :
        EventConfig object


    1. Compute residual
    2. Detect candidate onsets
    3. For each onset:
        grow event conservatively
    4. Apply credibility screen
    5. Merge close events (with guardrails)
    6. Build event records
    7. Run diagnostics

    Returns
    -------
    events :
        List of event dictionaries or a DataFrame
    diagnostics :
        Optional metadata for QA/QC
    """



def detect_candidate_onsets(residual, config):
    """
    Identify time indices where sustained positive deviation begins.

    Logic:
        - residual exceeds deviation_threshold
        - condition persists for sustain_duration

    For each timestep t:
    if residual[t] > deviation_threshold:
        check next sustain_duration timesteps
        if condition holds:
            mark t as candidate onset

    Returns:
        list of onset timestamps
    """

def grow_event(
    start_time,
    residual,
    flow,
    baseline,
    config
):
    """
    Extend an event forward in time until termination criteria are met.

    Honors:
        - small internal dips
        - multi-peak continuity
        - conservative termination
    
    current_time = start_time
    internal_gap_counter = 0

    while current_time < end_of_series:
        if residual[current_time] > return_threshold:
            internal_gap_counter = 0
            include timestep in event
        else:
            internal_gap_counter += 1
            if internal_gap_counter <= max_internal_gap:
                include timestep (allow dip)
            else:
                check termination window
                if termination confirmed:
                    break
        current_time += 1
        

    Returns:
        event_end_time
        event_metadata (peak time, etc.)
    """

def check_termination(residual_window, config):
    """
    Determine whether event has ended.

    Conditions:
        - residual below return_threshold
        - persists for termination_duration
    
    If residual has stayed below return_threshold
    for termination_duration timesteps:
        return True
    Else:
        return False

    Returns:
        True / False
    """

def merge_close_events(events, config):
    """
    Merge events separated by short dry gaps.

    For each adjacent event pair:
        compute gap duration
        if gap > merge_gap_duration:
            keep separate
        else:
            if both events credible:
                merge
            elif one is credible and gap residual never fully recovers:
                merge
            else:
                keep separate

    Purpose:
        - preserve long wet-weather responses
        - avoid artificial splitting

    Returns:
        merged_events
    """

def credible_event(events, config):
    """
    Apply credibility screen to filter out spurious events.

    Criteria:
        - duration >= min_event_duration
        - peak excess >= min_peak_excess

    For each event:
        if duration < min_event_duration:
            mark as non-credible
        elif peak_excess < min_peak_excess:
            mark as non-credible"""



def build_event_record(
    start_time,
    end_time,
    flow,
    baseline,
    residual
):
    """
    Create a dictionary or dataclass representing an event.

    Fields (minimal):
        - start_time
        - end_time
        - duration
        - peak_time
        - peak_excess
        - mean_excess
    """

def event_diagnostics(events, flow, baseline):
    """
    Optional checks:
        - negative RDII inside events
        - unusually long events
        - events during known dry periods
    """