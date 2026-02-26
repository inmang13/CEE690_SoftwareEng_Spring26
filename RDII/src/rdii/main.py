# MAIN ALGORITHM OUTLINE
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
from rdii.data_loader import load_config
from rdii.plots import plot_average_diurnal_pattern_all


def main(config_path: str = "config.json"):

    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in config file: {e}")
        sys.exit(1)

    # Setup paths
    project_root = (
        Path(config["project_root"])
        if "project_root" in config
        else Path(__file__).parent.parent.parent
    )
    raw_data_dir = project_root / config["paths"]["raw_data"]
    processed_dir = project_root / config["paths"]["processed_data"]
    plots_dir = project_root / config["paths"]["plots_dir"]
    combined_file = processed_dir / config["paths"]["combined_filename"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    combined_file = processed_dir / config["paths"]["combined_filename"]
    rain_file = processed_dir / config["paths"]["rain_filename"]
    cleaned_file = processed_dir / config["paths"]["cleaned_filename"]
    gwi_removed_file = processed_dir / config["paths"]["gwi_removed_filename"]

    # Load cleaned data
    try:
        data = pd.read_csv(processed_dir / config["paths"]["bwf_results_filename"])
        rain = pd.read_csv(rain_file)
        print("Loading cleaned data with GWI removed...")
    except Exception as e:
        print(f"✗ Failed to load cleaned data: {e}")
        sys.exit(1)

    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data = data[["DateTime", "Meter", "Raw", "BWI", "BWF"]]
    data["RDII"] = data["Raw"] - data["BWI"] - data["BWF"]

    plot_average_diurnal_pattern_all(data, output_dir=plots_dir)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    main(config_file)
