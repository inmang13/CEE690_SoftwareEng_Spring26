# src/rdii/process_rain.py
"""Module for processing rainfall data into multiple time resolutions.""" 
import os
import sys
from pathlib import Path

import pandas as pd

from rdii.utils import load_config


def main(config_path: str = "config.json"):

    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)


    # Setup paths
    project_root = (
        Path(config["project_root"])
        if "project_root" in config
        else Path(__file__).parent.parent.parent
    )

    plots_dir = project_root / config["paths"]["plots_dir"]

    processed_dir = project_root / config["paths"]["processed_data"]
    combined_file = processed_dir / config["paths"]["combined_filename"]
    rain_file = processed_dir / config["paths"]["rain_filename"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(combined_file)
    except Exception as e:
        print(f"✗ Failed to load combined data: {e}")
        sys.exit(1)

    df = df[["DateTime", "Rain_in", "Meter"]]
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime")

    # 5-minute
    rain_5min = df.groupby("Meter").resample("5min")["Rain_in"].sum().reset_index()

    # Hourly
    rain_hourly = df.groupby("Meter").resample("h")["Rain_in"].sum().reset_index()

    # Daily
    rain_daily = df.groupby("Meter").resample("D")["Rain_in"].sum().reset_index()

    rain_5min.to_csv(processed_dir / "rain_5min.csv", index=False)
    rain_hourly.to_csv(processed_dir / "rain_hourly.csv", index=False)
    rain_daily.to_csv(processed_dir / "rain_daily.csv", index=False)

    print("Saved rain_5min, rain_hourly, rain_daily")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    main(config_file)
