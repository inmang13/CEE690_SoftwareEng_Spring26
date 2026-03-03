import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label

from rdii.utils import detect_n_workers, load_config


def extract_storm_events_single_meter(
    df,
    meter_name,
    res_ratio_thresh=0.05,      # ratio above baseflow to begin detection
    slope_window=4,             # 1 hr persistence (4 x 15-min)
    end_ratio_thresh=0.03,      # return-to-baseline threshold
    min_event_hours=1.0,
    merge_gap_hours=6
):
    """
    Shape-aware storm event detection using residual ratio + slope persistence.
    """

    df = df.sort_values("DateTime").reset_index(drop=True)

    # Residuals and ratios
    df["Residual"] = df["Raw"] - df["BWF"]
    df["Res_Ratio"] = df["Residual"] / df["BWF"]
    df["Res_Smooth"] = df["Res_Ratio"].rolling(window=slope_window, center=True, min_periods=1).median()
    df["Slope"] = df["Res_Smooth"].diff()
    df["Slope_Roll"] = df["Slope"].rolling(window=slope_window, min_periods=1).mean()

    # Rising limb detection
    rise_condition = (df["Res_Smooth"] > res_ratio_thresh) & (df["Slope_Roll"] > 0)
    rise_persistent = rise_condition.rolling(window=slope_window, min_periods=1).sum() >= slope_window
    df["Candidate"] = rise_persistent

    # Initial candidate events
    df["Event_ID"], num_events = label(df["Candidate"])
    
    # Expand forward/backward to return-to-baseline
    event_mask = np.zeros(len(df), dtype=bool)
    for eid in range(1, num_events + 1):
        indices = np.where(df["Event_ID"] == eid)[0]
        if len(indices) == 0:
            continue

        # Backward
        j = indices[0]
        while j > 0 and df["Res_Smooth"].iloc[j] > end_ratio_thresh:
            event_mask[j] = True
            j -= 1

        # Forward
        j = indices[-1]
        while j < len(df) and df["Res_Smooth"].iloc[j] > end_ratio_thresh:
            event_mask[j] = True
            j += 1

    df["is_storm"] = event_mask
    df["Event_ID"], num_events = label(df["is_storm"])

    # Merge nearby events based on time gap
    event_ids = sorted([e for e in df["Event_ID"].unique() if e != 0])
    for i in range(len(event_ids) - 1):
        e1, e2 = event_ids[i], event_ids[i + 1]
        end_time_e1 = df[df["Event_ID"] == e1]["DateTime"].max()
        start_time_e2 = df[df["Event_ID"] == e2]["DateTime"].min()
        time_gap_hours = (start_time_e2 - end_time_e1).total_seconds() / 3600

        if time_gap_hours <= merge_gap_hours:
            df.loc[df["Event_ID"] == e2, "Event_ID"] = e1

    # Relabel sequentially
    df["Event_ID"], num_events = label(df["Event_ID"] > 0)

    # Build summary
    storm_summary = []
    for eid in range(1, num_events + 1):
        storm = df[df["Event_ID"] == eid]
        if storm.empty:
            continue

        duration_hrs = (storm["DateTime"].iloc[-1] - storm["DateTime"].iloc[0]).total_seconds() / 3600
        if duration_hrs < min_event_hours:
            continue

        rdii = storm["Residual"]
        dt_days = storm["DateTime"].diff().dt.total_seconds().median() / 86400

        storm_summary.append({
            "Event_ID": eid,
            "Meter": meter_name,
            "Start_Time": storm["DateTime"].iloc[0],
            "End_Time": storm["DateTime"].iloc[-1],
            "Duration_Hrs": duration_hrs,
            "Peak_Flow_MGD": storm["Raw"].max(),
            "Peak_RDII_MGD": rdii.max(),
            "Peak_Ratio": (storm["Raw"].max() / storm["BWF"].mean()),
            "Total_Volume_MG": (rdii * dt_days).sum()
        })

    return pd.DataFrame(storm_summary), df


def plot_event_with_context(df, event_id, hours=12):
    storm = df[df["Event_ID"] == event_id]
    if storm.empty:
        return

    t0 = storm["DateTime"].min() - pd.Timedelta(hours=hours)
    t1 = storm["DateTime"].max() + pd.Timedelta(hours=hours)

    window = df[(df["DateTime"] >= t0) & (df["DateTime"] <= t1)]

    plt.figure(figsize=(13, 5))
    plt.plot(window["DateTime"], window["Raw"], label="Raw", color="black")
    plt.plot(window["DateTime"], window["BWF"], label="BWF", linestyle="--")

    plt.axvspan(
        storm["DateTime"].min(),
        storm["DateTime"].max(),
        color="red",
        alpha=0.2,
        label="Storm Event"
    )

    plt.legend()
    plt.grid(True)
    plt.title(f"Storm Event {event_id} (Context View)")
    plt.tight_layout()
    plt.show()







def plot_event_diagnostics(df, event_id, hours=12, res_ratio_thresh=0.05):

    storm = df[df["Event_ID"] == event_id]
    if storm.empty:
        print("Event not found.")
        return

    t0 = storm["DateTime"].min() - pd.Timedelta(hours=hours)
    t1 = storm["DateTime"].max() + pd.Timedelta(hours=hours)

    window = df[(df["DateTime"] >= t0) & (df["DateTime"] <= t1)]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 10),
        sharex=True
    )

    # ---------------------------------------------------------
    # 1. Raw Flow + BWF
    # ---------------------------------------------------------
    axes[0].plot(window["DateTime"], window["Raw"], label="Raw", color="black")
    axes[0].plot(window["DateTime"], window["BWF"], linestyle="--", label="BWF")

    axes[0].axvspan(
        storm["DateTime"].min(),
        storm["DateTime"].max(),
        color="red",
        alpha=0.2,
        label="Storm Event"
    )

    axes[0].set_ylabel("Flow (MGD)")
    axes[0].set_title(f"Storm Event {event_id} — Flow Context")
    axes[0].legend()
    axes[0].grid(True)

    # ---------------------------------------------------------
    # 2. Residual + Smoothed + Residual Ratio
    # ---------------------------------------------------------
    ax_res = axes[1]

    ax_res.plot(window["DateTime"], window["Residual"], label="Residual")
    ax_res.plot(window["DateTime"], window["Res_Smooth"], linestyle="--", label="Res_Smooth")
    ax_res.axhline(0, linestyle=":", linewidth=1)

    ax_res.axvspan(
        storm["DateTime"].min(),
        storm["DateTime"].max(),
        color="red",
        alpha=0.2
    )

    ax_res.set_ylabel("Residual (MGD)")
    ax_res.set_title("Residual + Residual Ratio")
    ax_res.grid(True)

    # Secondary axis for ratio
    ax_ratio = ax_res.twinx()
    ax_ratio.plot(
        window["DateTime"],
        window["Res_Ratio"],
        linestyle=":",
        label="Res_Ratio"
    )

        # Threshold line
    ax_ratio.axhline(
        res_ratio_thresh,
        linestyle="--",
        label=f"res_ratio_thresh ({res_ratio_thresh})"
    )

    ax_ratio.set_ylabel("Residual Ratio")

    # Combine legends
    lines1, labels1 = ax_res.get_legend_handles_labels()
    lines2, labels2 = ax_ratio.get_legend_handles_labels()
    ax_res.legend(lines1 + lines2, labels1 + labels2)

    # ---------------------------------------------------------
    # 3. Slope + Rolling Slope
    # ---------------------------------------------------------
    axes[2].plot(window["DateTime"], window["Slope"], label="Slope")
    axes[2].plot(window["DateTime"], window["Slope_Roll"], linestyle="--", label="Slope_Roll")
    axes[2].axhline(0, linestyle=":", linewidth=1)

    axes[2].axvspan(
        storm["DateTime"].min(),
        storm["DateTime"].max(),
        color="red",
        alpha=0.2
    )

    axes[2].set_ylabel("Slope")
    axes[2].set_title("Slope Diagnostics")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
#  # Load configuration
# try:
#     config = load_config(config_path)
# except FileNotFoundError:
#     print(f"✗ Config file not found: {config_path}")
#     sys.exit(1)


# # Setup paths
# project_root = (
#     Path(config["project_root"])
#     if "project_root" in config
#     else Path(__file__).parent.parent.parent
# )

# processed_dir = project_root / config["paths"]["processed_data"]
# plots_dir = project_root / config["paths"]["plots_dir"] / "bwf_results"
# processed_dir.mkdir(parents=True, exist_ok=True)

# #gwi_removed_file = processed_dir / config["paths"]["gwi_removed_filename"]
# cleaned_file = processed_dir / config["paths"]["cleaned_filename"]


# --- EXECUTION ---
base="C:\\Users\\inman\\Documents\\RDII\\CEE690_SoftwareEng_Spring26\\RDII"
all = pd.read_csv(base+"\\data\\processed\\bwf_results.csv", parse_dates=['DateTime'])
df=all[all["Meter"] == "CBO"]
events, df_labeled = extract_storm_events_single_meter(
    df,
    meter_name="CBO",
    res_ratio_thresh=1.5,        # 5% above baseflow to begin detection
    slope_window=4,               # 1 hr persistence (4 x 15-min)
    end_ratio_thresh=0.02,        # return-to-baseline threshold
    min_event_hours=6.0,
    merge_gap_hours=100)

print(events.head)


print(len(events.head()), "events detected:")

gap_slice = df_labeled[
    (df_labeled["DateTime"] >= pd.Timestamp("2023-04-30 16:00:00")) &
    (df_labeled["DateTime"] <= pd.Timestamp("2023-04-30 17:15:00"))
]

print(gap_slice[["DateTime","Res_Ratio","Res_Smooth"]])

# for event_id in events["Event_ID"]:
#     plot_event_diagnostics(df_labeled, event_id=event_id, hours=24,res_ratio_thresh=1.5)
#events.to_csv("storm_event_characterization.csv", index=False)

# # if __name__ == "__main__":
# #     config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
# #     main(config_file)