"""
Module for loading Durham flow meter data from CSV files"
"""

import glob
import os
import pandas as pd

def read_flow_meter_data(file_path):
    """
    Read a single flow meter CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, skiprows=2)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # Add metadata columns
    filename=os.path.basename(file_path)
    df['Meter'] = extract_meter_name(os.path.basename(file_path))
    df['Source_File'] = filename

    # Rename columns to standard names
    df = df.rename(columns={
        'MM/dd/yyyy h:mm:ss tt': 'DateTime',
        'in': 'Rain_in',
        'MGD': 'Flow_MGD',
        'MGD.1': 'Flow_MGD_1',
        'in.1': 'Depth_in',
        'ft/s': 'Velocity_ft_s'
    })

    # Validate required column exists
    if 'DateTime' not in df.columns:
        raise ValueError(
            f"Missing DateTime column in {filename}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Parse datetime
    df['DateTime'] = pd.to_datetime(
        df['DateTime'],
        format='%m/%d/%Y %I:%M:%S %p',
        errors='coerce'
    )

    return df


def read_all_flow_meters(directory_path, verbose=True):
    """
    Read all flow meters files in a directory.
    """
    
    file_pattern = os.path.join(directory_path, 'DURHAM_*.csv')
    csv_files = sorted(glob.glob(file_pattern))

    all_data = []
    failed_files = []
    combined_df = pd.DataFrame()

    for file_path in csv_files:
        try:
            df = read_flow_meter_data(file_path)
            if len(df)>0:
                all_data.append()
                if verbose:
                    print(
                    f"✓ Loaded {os.path.basename(file_path)} "
                        f"({len(df)} rows)"
                    )

            else:
                if verbose:
                    print(
                        f"⚠ Skipped {os.path.basename(file_path)} "
                        f"(no valid data)"
                    )
        except Exception  as e:
            failed_files.append((file_path, str(e)))
            if verbose:
                print(f"✗ Failed {os.path.basename(file_path)}: {e}")

        if not all_data:
            raise ValueError(
                f"All {len(csv_files)} files failed to load. "
                f"First error: {failed_files[0][1] if failed_files else 'Unknown'}"
                )

        if verbose and failed_files:
            print(
            f"\nSuccessfully loaded {len(all_data)}/{len(csv_files)} files"
            )

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['Meter', 'DateTime']).reset_index(drop=True)

    return combined_df

def extract_meter_name(filename):
    """
    Extract meter name from filename
    Ex. Filename = DURHAM_DBO_20230101-20260101.csv
    """
    parts= filename.split('_')
    if len(parts)<2:
        raise ValueError(
            f"Filename '{filename}' doesn't match expected pattern "
            f"'DURHAM_METER_...'")
    return parts[1]

def validate_columns(df,required_cols):
    """
    Validate DataFrame has required columns
    """
    missing =set(required_cols)-set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )




        
