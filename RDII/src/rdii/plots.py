# src/rdii/plots.py
"""Module for creating visualizations of flow data and QC flags."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


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
    
    print(f"✓ Saved plot to {output_file}")
    
    return output_file

def plot_all_meters(df,output_dir: str = 'results/plots',figsize: tuple = (14, 6),dpi: int = 300,verbose: bool = True):
    """
    Create QC plots for all meters in the dataset.
    """
    # Get unique meters (excluding NaN)
    meters = df['Meter'].dropna().unique()

    saved_files = []
    
    for i, meter in enumerate(meters, 1):
        if verbose:
            print(f"[{i}/{len(meters)}] Plotting {meter}...", end=' ')
        
        try:
            output_file = plot_meter_qc(
                df, 
                meter, 
                output_dir=output_dir,
                figsize=figsize,
                dpi=dpi
            )
            
            if output_file:
                saved_files.append(output_file)
                if verbose:
                    print(f"✓ Saved to {output_file.name}")
            else:
                if verbose:
                    print("✗ No data")
                    
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
    
    if verbose:
        print("="*60)
        print(f"✓ Successfully created {len(saved_files)} plots")
    
    return saved_files