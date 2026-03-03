import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.ndimage import label

pio.renderers.default = 'browser'  # opens in browser — most reliable in VSCode


# =============================================================================
# CONFIGURATION
# =============================================================================

# Event detection parameters
MERGE_GAP_HOURS     = 3        # merge events within this many hours
MIN_DURATION_HRS    = 2        # drop events shorter than this
PEAK_THRESHOLD      = 0.2      # minimum RDII (MGD) above BWF to count as event

# Rule-based rainfall classifier thresholds (derived from CBO + rain gauge analysis)
PEAKING_FACTOR_THRESH   = 2.0  # peak raw flow / mean BWF during event
CUMULATIVE_RDII_THRESH  = 0.2  # total RDII volume in MG


# =============================================================================
# EVENT DETECTION
# =============================================================================

def extract_storm_events(df, meter_name,
                         merge_gap_hours=MERGE_GAP_HOURS,
                         min_duration_hrs=MIN_DURATION_HRS,
                         peak_threshold=PEAK_THRESHOLD):
    """
    Detect and box discrete anomaly events from BWF detection output.

    Uses BWF_Anomaly flags from calculate_BWF.py as the starting point,
    then applies gap filling and duration/peak filtering to produce clean
    event boxes.

    Parameters
    ----------
    df : DataFrame
        Must contain 'DateTime', 'Raw', 'BWF', 'BWF_Anomaly', 'Meter'.
    meter_name : str
    merge_gap_hours : float
        Merge events separated by less than this many hours (default 3).
    min_duration_hrs : float
        Drop events shorter than this (default 2hr).
    peak_threshold : float
        Minimum peak RDII in MGD to retain event (default 0.2 MGD).

    Returns
    -------
    events_df : DataFrame
        One row per event with detection stats and shape features.
    df_labeled : DataFrame
        Original df with Event_ID column added.
    """

    df = df.sort_values('DateTime').reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 1. GAP FILL BWF_Anomaly flags
    # -------------------------------------------------------------------------
    flags             = df['BWF_Anomaly'].values.astype(bool).copy()
    max_gap_intervals = int(merge_gap_hours * 4)   # 15-min intervals

    filled    = flags.copy()
    in_event  = False
    gap_count = 0

    for i in range(len(filled)):
        if filled[i]:
            in_event  = True
            gap_count = 0
        elif in_event:
            gap_count += 1
            if gap_count <= max_gap_intervals:
                filled[i] = True
            else:
                in_event  = False
                gap_count = 0

    # -------------------------------------------------------------------------
    # 2. LABEL CONTIGUOUS BLOCKS
    # -------------------------------------------------------------------------
    labeled, n = label(filled)

    # -------------------------------------------------------------------------
    # 3. REMOVE SHORT / WEAK EVENTS
    # -------------------------------------------------------------------------
    min_intervals = int(min_duration_hrs * 4)

    for eid in range(1, n + 1):
        mask = labeled == eid
        if mask.sum() < min_intervals:
            filled[mask] = False

    labeled, n = label(filled)

    df = df.copy()
    df['Event_ID'] = labeled

    # -------------------------------------------------------------------------
    # 4. BUILD EVENT SUMMARY WITH SHAPE FEATURES
    # -------------------------------------------------------------------------
    events = []

    for eid in range(1, n + 1):
        event = df[df['Event_ID'] == eid]
        if event.empty:
            continue

        rdii       = (event['Raw'] - event['BWF']).values
        raw        = event['Raw'].values
        bwf        = event['BWF'].values
        times      = event['DateTime'].values

        peak_idx       = rdii.argmax()
        peak_time      = pd.Timestamp(times[peak_idx])
        start_time     = pd.Timestamp(times[0])
        end_time       = pd.Timestamp(times[-1])

        rising_limb_hrs  = (peak_time  - start_time).total_seconds() / 3600
        falling_limb_hrs = (end_time   - peak_time).total_seconds() / 3600
        total_hrs        = (end_time   - start_time).total_seconds() / 3600
        duration_hrs     = len(event) * 0.25

        peaking_factor   = raw.max() / bwf.mean() if bwf.mean() > 0 else np.nan
        cumulative_rdii  = np.clip(rdii, 0, None).sum() * (15 / 1440)  # MG
        rising_slope     = rdii[peak_idx] / rising_limb_hrs if rising_limb_hrs > 0 else np.nan
        flashiness       = rdii[peak_idx] / rising_limb_hrs if rising_limb_hrs > 0 else np.nan
        skewness         = rising_limb_hrs / total_hrs       if total_hrs > 0       else np.nan

        events.append({
            # Identification
            'Event_ID'           : eid,
            'Meter'              : meter_name,
            'Start'              : start_time,
            'End'                : end_time,
            # Basic stats
            'Duration_hrs'       : round(duration_hrs,      2),
            'Peak_Raw_MGD'       : round(raw.max(),         4),
            'Peak_RDII_MGD'      : round(rdii.max(),        4),
            # Shape features
            'peaking_factor'     : round(peaking_factor,    3),
            'cumulative_rdii_MG' : round(cumulative_rdii,   6),
            'rising_limb_hrs'    : round(rising_limb_hrs,   2),
            'rising_limb_slope'  : round(rising_slope,      4) if rising_slope is not np.nan else np.nan,
            'falling_limb_hrs'   : round(falling_limb_hrs,  2),
            'rdii_flashiness'    : round(flashiness,        4) if flashiness is not np.nan else np.nan,
            'skewness'           : round(skewness,          3) if skewness is not np.nan else np.nan,
            'month'              : start_time.month,
        })

    events_df = pd.DataFrame(events)

    if events_df.empty:
        print(f"  No events found for {meter_name}")
        return events_df, df

    # -------------------------------------------------------------------------
    # 5. FILTER BY PEAK THRESHOLD
    # -------------------------------------------------------------------------
    events_df = events_df[events_df['Peak_RDII_MGD'] > peak_threshold].reset_index(drop=True)
    valid_ids = set(events_df['Event_ID'])
    df['Event_ID'] = df['Event_ID'].where(df['Event_ID'].isin(valid_ids), 0)

    # -------------------------------------------------------------------------
    # 6. RULE-BASED RAINFALL CLASSIFICATION
    #    Thresholds derived from CBO meter + rain gauge analysis
    # -------------------------------------------------------------------------
    events_df['is_rainfall_event'] = (
        (events_df['peaking_factor']     >= PEAKING_FACTOR_THRESH) &
        (events_df['cumulative_rdii_MG'] >= CUMULATIVE_RDII_THRESH)
    )

    n_rain    = events_df['is_rainfall_event'].sum()
    n_other   = (~events_df['is_rainfall_event']).sum()

    print(f"  {meter_name}: {len(events_df)} events | "
          f"Rainfall: {n_rain} | Non-rainfall: {n_other}")

    return events_df, df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_events_overview(df, events_df, meter_name):
    """
    Full record overview with event boxes colored by classification.
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(df['DateTime'], df['Raw'], linewidth=0.5, color='gray',     alpha=0.7, label='Raw')
    ax.plot(df['DateTime'], df['BWF'], linewidth=1.0, color='darkblue', alpha=0.9, label='BWF')

    for _, ev in events_df.iterrows():
        color = 'steelblue' if ev['is_rainfall_event'] else 'coral'
        ax.axvspan(ev['Start'], ev['End'], alpha=0.25, color=color)

    # Legend patches
    from matplotlib.patches import Patch
    handles = [
        ax.lines[0], ax.lines[1],
        Patch(color='steelblue', alpha=0.4, label='Rainfall Event'),
        Patch(color='coral',     alpha=0.4, label='Non-Rainfall Event'),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_ylabel('Flow (MGD)', fontsize=11)
    ax.set_title(f'{meter_name} — Event Classification Overview', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_event_detail(df, events_df, event_id, hours_context=12):
    """
    Detailed view of a single event with context window.
    """
    ev = events_df[events_df['Event_ID'] == event_id]
    if ev.empty:
        print(f"Event {event_id} not found")
        return
    ev = ev.iloc[0]

    t0     = ev['Start'] - pd.Timedelta(hours=hours_context)
    t1     = ev['End']   + pd.Timedelta(hours=hours_context)
    window = df[(df['DateTime'] >= t0) & (df['DateTime'] <= t1)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Flow panel
    axes[0].plot(window['DateTime'], window['Raw'], color='gray',     linewidth=1,   label='Raw')
    axes[0].plot(window['DateTime'], window['BWF'], color='darkblue', linewidth=1.5, label='BWF')
    axes[0].axvspan(ev['Start'], ev['End'],
                    color='steelblue' if ev['is_rainfall_event'] else 'coral',
                    alpha=0.2, label='Event')
    axes[0].set_ylabel('Flow (MGD)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"Event {event_id} | {ev['Start'].strftime('%Y-%m-%d')} | "
        f"{'Rainfall' if ev['is_rainfall_event'] else 'Non-Rainfall'} | "
        f"Peak RDII: {ev['Peak_RDII_MGD']:.3f} MGD | "
        f"Peaking Factor: {ev['peaking_factor']:.2f} | "
        f"Cum. RDII: {ev['cumulative_rdii_MG']:.4f} MG",
        fontsize=10, fontweight='bold'
    )

    # RDII panel
    rdii = (window['Raw'] - window['BWF']).clip(lower=0)
    axes[1].fill_between(window['DateTime'], rdii,
                         color='steelblue' if ev['is_rainfall_event'] else 'coral',
                         alpha=0.5, label='RDII')
    axes[1].axvspan(ev['Start'], ev['End'], alpha=0.1,
                    color='steelblue' if ev['is_rainfall_event'] else 'coral')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_ylabel('RDII (MGD)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_classification_scatter(events_df):
    """
    Scatter plot of peaking_factor vs cumulative_rdii_MG
    colored by classification, with threshold lines overlaid.
    """
    rain    = events_df[events_df['is_rainfall_event']]
    no_rain = events_df[~events_df['is_rainfall_event']]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(no_rain['cumulative_rdii_MG'], no_rain['peaking_factor'],
               color='coral',     alpha=0.6, s=40, label='Non-Rainfall',
               edgecolors='white', linewidth=0.5)
    ax.scatter(rain['cumulative_rdii_MG'], rain['peaking_factor'],
               color='steelblue', alpha=0.6, s=40, label='Rainfall',
               edgecolors='white', linewidth=0.5)

    ax.axhline(PEAKING_FACTOR_THRESH,  color='black', linestyle='--',
               linewidth=1.5, label=f'Peaking factor ≥ {PEAKING_FACTOR_THRESH}')
    ax.axvline(CUMULATIVE_RDII_THRESH, color='gray',  linestyle='--',
               linewidth=1.5, label=f'Cumulative RDII ≥ {CUMULATIVE_RDII_THRESH} MG')

    ax.set_xlabel('Cumulative RDII (MG)', fontsize=12)
    ax.set_ylabel('Peaking Factor',       fontsize=12)
    ax.set_title('Rainfall Classification — Peaking Factor vs Cumulative RDII',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_events_overview_interactive(df, events_df, meter_name):
    """
    Interactive overview with event boxes colored by classification.
    Hover shows event stats.
    """
    fig = go.Figure()

    # Raw flow
    fig.add_trace(go.Scatter(
        x=df['DateTime'], y=df['Raw'],
        mode='lines',
        line=dict(color='lightgray', width=0.8),
        name='Raw Flow',
        hovertemplate='%{x}<br>Raw: %{y:.3f} MGD'
    ))

    # BWF
    fig.add_trace(go.Scatter(
        x=df['DateTime'], y=df['BWF'],
        mode='lines',
        line=dict(color='darkblue', width=1.5),
        name='BWF',
        hovertemplate='%{x}<br>BWF: %{y:.3f} MGD'
    ))

    # Event boxes
    for _, ev in events_df.iterrows():
        color = 'steelblue' if ev['is_rainfall_event'] else 'coral'
        label = 'Rainfall' if ev['is_rainfall_event'] else 'Non-Rainfall'

        fig.add_vrect(
            x0=ev['Start'], x1=ev['End'],
            fillcolor=color, opacity=0.2,
            line_width=0,
            annotation_text=f"E{ev['Event_ID']}",
            annotation_position='top left',
            annotation_font_size=8,
        )

    fig.update_layout(
        title=f'{meter_name} — Event Classification',
        xaxis_title='Date',
        yaxis_title='Flow (MGD)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(orientation='h', y=1.05)
    )

    fig.show()

def plot_event_detail_interactive(df, events_df, event_id, hours_context=12):
    """
    Interactive detail view of a single event.
    """
    ev = events_df[events_df['Event_ID'] == event_id]
    if ev.empty:
        print(f"Event {event_id} not found")
        return
    ev = ev.iloc[0]

    t0     = ev['Start'] - pd.Timedelta(hours=hours_context)
    t1     = ev['End']   + pd.Timedelta(hours=hours_context)
    window = df[(df['DateTime'] >= t0) & (df['DateTime'] <= t1)].copy()
    window['RDII'] = (window['Raw'] - window['BWF']).clip(lower=0)

    color = 'steelblue' if ev['is_rainfall_event'] else 'coral'

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Flow', 'RDII'],
                        vertical_spacing=0.08)

    # Flow panel
    fig.add_trace(go.Scatter(
        x=window['DateTime'], y=window['Raw'],
        mode='lines', line=dict(color='gray', width=1),
        name='Raw'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=window['DateTime'], y=window['BWF'],
        mode='lines', line=dict(color='darkblue', width=1.5),
        name='BWF'
    ), row=1, col=1)

    # RDII panel
    fig.add_trace(go.Scatter(
        x=window['DateTime'], y=window['RDII'],
        mode='lines', fill='tozeroy',
        line=dict(color=color, width=1),
        fillcolor=f'rgba(70,130,180,0.3)' if ev['is_rainfall_event'] else 'rgba(255,127,80,0.3)',
        name='RDII'
    ), row=2, col=1)

    # Event span on both panels
    for row in [1, 2]:
        fig.add_vrect(
            x0=ev['Start'], x1=ev['End'],
            fillcolor=color, opacity=0.15,
            line_width=0, row=row, col=1
        )

    fig.update_layout(
        title=(f"Event {event_id} | {ev['Start'].strftime('%Y-%m-%d')} | "
               f"{'Rainfall' if ev['is_rainfall_event'] else 'Non-Rainfall'} | "
               f"Peak RDII: {ev['Peak_RDII_MGD']:.3f} MGD | "
               f"Peaking Factor: {ev['peaking_factor']:.2f}"),
        height=600,
        template='plotly_white',
        hovermode='x unified'
    )

    fig.show()


def plot_classification_scatter_interactive(events_df):

    rain    = events_df[events_df['is_rainfall_event']]
    no_rain = events_df[~events_df['is_rainfall_event']]

    fig = go.Figure()

    for subset, name, color in [
        (rain,    'Rainfall',     'steelblue'),
        (no_rain, 'Non-Rainfall', 'coral')
    ]:
        fig.add_trace(go.Scatter(
            x=subset['cumulative_rdii_MG'],
            y=subset['peaking_factor'],
            mode='markers',
            name=name,
            marker=dict(color=color, size=8, opacity=0.7,
                        line=dict(color='white', width=0.5)),
            customdata=subset[['Event_ID', 'Meter', 'Start',
                                'Duration_hrs', 'Peak_RDII_MGD']].values,
            hovertemplate=(
                '<b>Event %{customdata[0]}</b> — %{customdata[1]}<br>'
                'Date: %{customdata[2]}<br>'
                'Duration: %{customdata[3]:.1f} hrs<br>'
                'Peak RDII: %{customdata[4]:.3f} MGD<br>'
                'Cum. RDII: %{x:.4f} MG<br>'
                'Peaking Factor: %{y:.2f}'
            )
        ))

    # Threshold lines
    fig.add_hline(y=PEAKING_FACTOR_THRESH,  line_dash='dash',
                  line_color='black', annotation_text=f'Peaking factor = {PEAKING_FACTOR_THRESH}')
    fig.add_vline(x=CUMULATIVE_RDII_THRESH, line_dash='dash',
                  line_color='gray',  annotation_text=f'Cum. RDII = {CUMULATIVE_RDII_THRESH} MG')

    fig.update_layout(
        title='Rainfall Classification — Peaking Factor vs Cumulative RDII',
        xaxis_title='Cumulative RDII (MG)',
        yaxis_title='Peaking Factor',
        height=550,
        template='plotly_white',
        hovermode='closest'
    )

    fig.show()

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == '__main__':

    base = "C:\\Users\\inman\\Documents\\RDII\\CEE690_SoftwareEng_Spring26\\RDII"
    df_all = pd.read_csv(base + "\\data\\processed\\bwf_results.csv",
                         parse_dates=['DateTime'])

    all_events  = []
    all_labeled = []

    for meter in df_all['Meter'].unique():
        meter_df = df_all[df_all['Meter'] == meter].copy().reset_index(drop=True)

        events_df, df_labeled = extract_storm_events(
            meter_df,
            meter_name       = meter,
            merge_gap_hours  = MERGE_GAP_HOURS,
            min_duration_hrs = MIN_DURATION_HRS,
            peak_threshold   = PEAK_THRESHOLD
        )

        if not events_df.empty:
            all_events.append(events_df)
            all_labeled.append(df_labeled)

            plot_events_overview_interactive(df_labeled, events_df, meter)

    # Combined results
    all_events_df  = pd.concat(all_events,  ignore_index=True)
    all_labeled_df = pd.concat(all_labeled, ignore_index=True)

    print(f"\nTotal events across all meters: {len(all_events_df)}")
    print(all_events_df.groupby('Meter')['is_rainfall_event']
          .value_counts().unstack(fill_value=0))

    plot_classification_scatter_interactive(all_events_df)

    # Save
    all_events_df.to_csv(base + "\\data\\processed\\storm_events.csv", index=False)
    print("\n✓ Saved storm_events.csv")