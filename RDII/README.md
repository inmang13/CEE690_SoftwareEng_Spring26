# RDII Analysis - Durham Flow Meter Data

Software Engineering for Data Science class project analyzing rainfall-derived infiltration and inflow (RDII) in Durham's sewer system. The pipeline processes 15-minute resolution flow meter data from 15 meters across Durham, separating base wastewater flow (BWF) from infiltration and inflow contributions using iterative anomaly detection.

## Project Structure

```
RDII/
├── config.json                  # Pipeline configuration
├── environment.yml              # Conda environment
├── Makefile                     # Pipeline shortcuts
├── pyproject.toml               # Package configuration
├── README.md
├── data/
│   ├── raw/
│   │   └── Durham/              # Raw 15-min flow meter CSVs (15 meters)
│   └── processed/               # Intermediate and final outputs
├── src/
│   └── rdii/
│       ├── __init__.py
│       ├── main.py              # Pipeline entry point
│       ├── data_loader.py       # Load raw flow meter data
│       ├── data_cleaner.py      # QC and cleaning
│       ├── process_rain.py      # Rainfall processing
│       ├── remove_GWI.py        # Groundwater infiltration removal
│       ├── calculate_BWF.py     # Base wastewater flow estimation (Prophet)
│       └── plots.py             # Visualization
├── results/
│   └── plots/                   # Output figures per meter
├── scripts/
│   └── scrap.ipynb
└── tests/
    ├── test_data_loader.py
    ├── test_data_cleaner.py
    ├── test_anomoly_detection.py
    └── test_remove_BWI.py
```

## Background

Rainfall-derived infiltration and inflow (RDII) refers to stormwater and groundwater that enters sanitary sewer systems during and after rain events. Identifying and quantifying RDII is critical for sewer capacity planning and infrastructure management. This pipeline separates observed sewer flow into three components:

- **GWI** — Groundwater infiltration (slow, seasonal baseline)
- **BWF** — Base wastewater flow (normal dry-weather usage patterns)
- **RDII** — Residual flow attributed to rainfall events

## Installation

### Option 1: Conda (recommended)

```bash
git clone https://github.com/inmang13/CEE690_SoftwareEng_Spring26.git
cd RDII
conda env create -f environment.yml
conda activate RDII
pip install -e .
```

### Option 2: pip

```bash
git clone https://github.com/inmang13/CEE690_SoftwareEng_Spring26.git
cd RDII
pip install -e ".[dev]"
```

## Configuration

All pipeline parameters are controlled through `config.json`:


Key parameters:

## Configuration

All pipeline parameters are controlled through `config.json`:
```json
{
  "paths": {
    "raw_data": "data/raw/Durham",
    "processed_data": "data/processed",
    "combined_filename": "combined_flow_data.csv",
    "cleaned_filename": "cleaned_flow_data.csv",
    "rain_filename": "rain_daily.csv",
    "bwf_results_filename": "bwf_results.csv",
    "gwi_removed_filename": "cleaned_gwi.csv",
    "plots_dir": "results/plots"
  },
  "cleaning": {
    "flow_column": "Flow_MGD",
    "frequency": "15min",
    "interpolation_limit": 4
  },
  "gwi": {
    "fraction_min": 0.85,
    "rolling_window": 30,
    "night_start": 1,
    "night_end": 7
  },
  "plotting": {
    "plot": "False",
    "figsize": [14, 6],
    "dpi": 300
  },
  "parallel": {
    "n_workers": 15
  },
  "bwf": {
    "k": 2.2,
    "sigma_method": "robust",
    "max_iterations": 10,
    "threshold": 0.05
  }
}
```

Key parameters:

| Parameter | Description | Default |
|---|---|---|
| `cleaning.interpolation_limit` | Max consecutive missing values to interpolate | 4 |
| `gwi.fraction_min` | Percentile threshold for GWI estimation | 0.85 |
| `gwi.rolling_window` | Rolling window size in days for GWI | 30 |
| `gwi.night_start` / `night_end` | Night hours used for GWI estimation | 1–7 |
| `bwf.k` | Sigma multiplier for anomaly detection | 2.2 |
| `bwf.sigma_method` | `robust` (MAD) or `standard` (mean/std) | `robust` |
| `bwf.max_iterations` | Maximum iterations for BWF convergence | 10 |
| `bwf.threshold` | Residual convergence threshold | 0.05 |

## Usage

Each script in the pipeline can be run independently in the following order:

### 1. Load raw data
```bash
python src/rdii/data_loader.py config.json
```
Reads raw CSVs from `data/raw/Durham/` and combines into `combined_flow_data.csv`.

### 2. Clean data
```bash
python src/rdii/data_cleaner.py config.json
```
Removes outliers, interpolates short gaps, flags quality issues. Outputs `cleaned_flow_data.csv`.

### 3. Process rainfall
```bash
python src/rdii/process_rain.py config.json
```
Aggregates 5-minute rainfall data to hourly and daily resolution.

### 4. Remove GWI
```bash
python src/rdii/remove_GWI.py config.json
```
Estimates and removes the groundwater infiltration component. Outputs `cleaned_gwi.csv`.

### 5. Calculate BWF
```bash
python src/rdii/calculate_BWF.py config.json
```
Runs iterative Prophet-based anomaly detection to isolate base wastewater flow. Parallelizes across meters using joblib. Outputs `bwf_results.csv`.
### Run with a custom config

```bash
python src/rdii/main.py my_config.json
```



### On the Duke Compute Cluster (DCC)

The pipeline automatically detects available cores via `SLURM_NTASKS` and parallelizes across meters using joblib.


## Running Tests

```bash
pytest tests/
```

## Dependencies

Core: `pandas`, `numpy`, `scipy`, `prophet`, `holidays`, `joblib`, `matplotlib`

See `pyproject.toml` for full dependency list with version constraints.

## Author

Grace Inman — CEE 690 Software Engineering for Data Science, Duke University, Spring 2026