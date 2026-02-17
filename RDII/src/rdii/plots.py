# src/rdii/plots.py
"""Module for creating visualizations of flow data and QC flags."""

import sys
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
from pathlib import Path
from rdii.remove_BWI import calculate_BWI_minflow



def plot_meter_qc(df,meter_name,output_dir='results/plots',figsize=(14, 6),dpi=300):
    """
    Create a plot of flow data with QC issues highlighted.
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter for specific meter
    meter_df = df[df['Meter'] == meter_name].copy()
    meter_df['DateTime'] = pd.to_datetime(meter_df['DateTime'])
    meter_df = meter_df.sort_values('DateTime')
    
    if len(meter_df) == 0:
        print(f"Warning: No data found for meter {meter_name}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main data line
    ax.plot(meter_df['DateTime'], meter_df['Flow_MGD'], 
            color='steelblue', linewidth=1.5, label='Flow Data', zorder=3)
    
    # Add shaded regions for non-OK periods
    non_ok = meter_df[meter_df['QC_flag'] != 'OK'].copy()
    
    if len(non_ok) > 0:
        # Group consecutive non-OK periods
        non_ok['group'] = (non_ok['DateTime'].diff() > pd.Timedelta('15min')).cumsum()
        
        qc_colors = {
            'INTERPOLATED': 'orange',
            'MISSING': 'red',
            'FLATLINE_REMOVED': 'purple',
            'NEGATIVE': 'brown'

        }
        
        for qc_flag, color in qc_colors.items():
            flag_groups = non_ok[non_ok['QC_flag'] == qc_flag].groupby('group')
            
            first_of_type = True
            for _, group in flag_groups:
                if len(group) > 0:
                    # Add label only for first occurrence of each type
                    label = qc_flag if first_of_type else None
                    first_of_type = False
                    
                    ax.axvspan(
                        group['DateTime'].min(),
                        group['DateTime'].max(),
                        color=color,
                        alpha=0.2,
                        label=label,
                        zorder=1
                    )
    
    # Formatting
    ax.set_xlabel('DateTime', fontsize=12)
    ax.set_ylabel('Flow (MGD)', fontsize=12)
    ax.set_title(f'Flow Data for {meter_name} Meter - QC Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{meter_name}_qc_plot.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    
    return output_file


def plot_BWI_estimate(df, meter_name, output_dir='results/plots', figsize=(14, 6), dpi=300):
    """
    Create a plot of flow data with BWI estimate overlaid.
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter for specific meter
    meter_df = df[df['Meter'] == meter_name].copy()
    meter_df['DateTime'] = pd.to_datetime(meter_df['DateTime'])
    meter_df = meter_df.sort_values('DateTime')
    
    if len(meter_df) == 0:
        print(f"Warning: No data found for meter {meter_name}")
        return None
    
    # Calculate BWI estimate
    bwi_estimate = calculate_BWI_minflow(meter_df, fraction_min=0.85)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main data line
    ax.plot(meter_df['DateTime'], meter_df['Flow_MGD'], 
            color='steelblue', linewidth=1.5, label='Flow Data', zorder=3)
    
    # Plot BWI estimate
    ax.plot(bwi_estimate.index, bwi_estimate.values, 
            color='orange', linewidth=1.5, label='Estimated BWI (MNF)', zorder=4)
    
    # Formatting
    ax.set_xlabel('DateTime', fontsize=12)
    ax.set_ylabel('Flow (MGD)', fontsize=12)
    ax.set_title(f'Flow Data and BWI Estimate for {meter_name} Meter', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{meter_name}_BWI_plot.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    
    return output_file

def plot_all_meters(df,output_dir = 'results/plots',figsize =(14, 6),dpi= 300,plot_type = 'qc',verbose = False):
    """
    Create  plots for all meters in the dataset.
    """
    # Get unique meters (excluding NaN)
    meters = df['Meter'].dropna().unique()

    if verbose:
        print(f"Creating {plot_type.upper()} plots for {len(meters)} meters...")
        print(f"Output directory: {output_dir}")
        print("="*60)
    
    saved_files = []
    
    for i, meter in enumerate(meters, 1):
        if verbose:
            print(f"[{i}/{len(meters)}] Plotting {meter}...", end=' ')
        
        try:
            if plot_type.lower() == 'qc':
                output_file = plot_meter_qc(
                    df, meter, 
                    output_dir=output_dir,
                    figsize=figsize,
                    dpi=dpi
                )
            elif plot_type.lower() == 'bwi':
                output_file = plot_BWI_estimate(
                    df, meter,
                    output_dir=output_dir,
                    figsize=figsize,
                    dpi=dpi
                )
            else:
                if verbose:
                    print(f"✗ Invalid plot_type: {plot_type}")
                continue
            
            if output_file:
                saved_files.append(output_file)
                if Plotting:
                    print(f"✓ Saved to {output_file.name}")
            else:
                if Plotting:
                    print("✗ No data")
                    
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
    
    if verbose:
        print("="*60)
        print(f"✓ Successfully created {len(saved_files)} plots")
    
    return saved_files

def load_config(config_path ):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
        

def main(config_path: str = 'config.json'):
        
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
        project_root = Path(config['project_root']) if 'project_root' in config else Path(__file__).parent.parent.parent
        raw_data_dir = project_root / config['paths']['raw_data']
        processed_dir = project_root / config['paths']['processed_data']
        combined_file = processed_dir / config['paths']['combined_filename']        
        processed_dir.mkdir(parents=True, exist_ok=True)

        combined_file = processed_dir / config['paths']['combined_filename']
        cleaned_file = processed_dir / config['paths']['cleaned_filename']

        # Load cleaned data
        try:
            cleaned_data = pd.read_csv(cleaned_file)
            print(f"✓ Loaded cleaned data with {len(cleaned_data)} rows and {len(cleaned_data['Meter'].unique())} meters")
        except Exception as e:
            print(f"✗ Failed to load cleaned data: {e}")
            sys.exit(1)

        try:
                # Create QC plots for all meters
                print("Creating QC plots for all meters...")
                plot_all_meters(cleaned_data, plot_type='qc')
                # Create BWI estimate plots for all meters
                print("Creating BWI estimate plots for all meters...")
                plot_all_meters(cleaned_data, plot_type='bwi')

        except Exception as e:
            print(f"✗ Failed to create plots: {e}")
            sys.exit(1)



if __name__ == "__main__":
        config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
        main(config_file)
        
