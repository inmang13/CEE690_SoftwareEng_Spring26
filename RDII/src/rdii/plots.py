# src/rdii/plots.py
"""Module for creating visualizations of flow data and QC flags."""

import sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from pathlib import Path
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
    
    
    return output_file


def plot_GWI_estimate(df, meter_name, output_dir='results/plots', figsize=(14, 6), dpi=300):
    """
    Create a plot of flow data with GWI estimate overlaid.
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
    
    # Plot GWI estimate
    ax.plot(meter_df['DateTime'], meter_df['GWI_estimate'], 
            color='orange', linewidth=1.5, label='Estimated GWI (MNF)', zorder=4)
    
    # Formatting
    ax.set_xlabel('DateTime', fontsize=12)
    ax.set_ylabel('Flow (MGD)', fontsize=12)
    ax.set_title(f'Flow Data and GWI Estimate for {meter_name} Meter', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{meter_name}_GWI_plot.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    
    return output_file


def plot_final_classification(results, meter_name, output_dir='results/plots', figsize=(14, 6), dpi=300):
    """
    Plot the final anomaly classification with forecast and anomaly labels.
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    
    k=results['k']

    original = results['original_data']
    anomaly_labels = results['anomaly_labels']
    final_forecast = results['forecast']
    bounds = results['final_bounds']
    
    # Plot normal points
    normal_mask = ~anomaly_labels
    ax.plot(original['ds'][normal_mask], original['y'][normal_mask], 'o',
            color='blue', alpha=0.5, markersize=3, label='Normal')
    
    # Plot anomalous points
    anomaly_mask = anomaly_labels
    ax.plot(original['ds'][anomaly_mask], original['y'][anomaly_mask], 'o',
            color='red', alpha=0.5, markersize=3, label='Anomaly')
    
    # Plot final forecast
    ax.plot(final_forecast['ds'], final_forecast['yhat'],
            linewidth=2, color='darkblue', label='Forecast', zorder=5)
    
    # Plot confidence bounds
    ax.fill_between(final_forecast['ds'],
                    final_forecast['yhat'] + bounds[0],
                    final_forecast['yhat'] + bounds[1],
                    alpha=0.2, color='blue', label=f'{k}σ Bounds')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Flow (MGD)', fontsize=12)
    ax.set_title('Final Anomaly Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Save figure
    output_file = output_path / f'{meter_name}_anomaly_detection_final_classification.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()



def plot_iteration_statistics(results,meter_name,output_dir='results/plots', figsize=(14, 6), dpi=300):        
    """
    Plot iteration statistics: training data reduction, anomalies removed,
    residual std deviation, and residual mean convergence.
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    history = results['iteration_history']
    iterations = [h['iteration'] + 1 for h in history]
    n_points = [h['n_points'] for h in history]
    n_anomalies = [h['n_anomalies'] for h in history]
    sigmas = [h['sigma'] for h in history]
    mus = [h['mu'] for h in history]
    
    # Training points remaining
    axes[0, 0].plot(iterations, n_points, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Training Points Remaining')
    axes[0, 0].set_title('Training Data Reduction')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Whole number x-axis
    
    # Anomalies removed per iteration (with secondary y-axis for %)
    bars = axes[0, 1].bar(iterations, n_anomalies, color='coral', alpha=0.7)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Anomalies Detected')
    axes[0, 1].set_title('Anomalies Removed Per Iteration')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Whole number x-axis
    
    # Calculate percentage anomalies detected relative to points at each iteration
    percent_anomalies = [(n_anomalies[i] / n_points[i]) * 100 if n_points[i] > 0 else 0 for i in range(len(n_points))]
    
    # Annotate percentage above bars
    for rect, pct in zip(bars, percent_anomalies):
        height = rect.get_height()
        axes[0, 1].annotate(f'{pct:.1f}%', 
                            xy=(rect.get_x() + rect.get_width() / 2, height), 
                            xytext=(0, 5),  # 5 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9,
                            color='darkred',
                            fontweight='bold')
    
    ax2 = axes[0, 1].twinx()  # Create secondary y-axis
    ax2.set_ylabel('% Anomalies Detected', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.set_ylim(0, max(percent_anomalies)*1.2 if percent_anomalies else 10)  # Add some padding on top
    
    ax2.legend(loc='upper right', fontsize=9)
    
    # Residual standard deviation
    axes[1, 0].plot(iterations, sigmas, 's-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Standard Deviation (σ)')
    axes[1, 0].set_title('Residual Standard Deviation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Whole number x-axis
    
    # Residual mean convergence
    axes[1, 1].plot(iterations, mus, 'd-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Mean Residual (μ)')
    axes[1, 1].set_title('Residual Mean Convergence')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Whole number x-axis
    
    # Save figure
    output_file = output_path / f'{meter_name}_anomaly_detection_statistics.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_average_diurnal_pattern_all(results,meter_name,output_dir='results/plots', figsize=(14, 6), dpi=300):
    """
    Plot the average diurnal pattern for:
    - Forecasted baseline
    - Non-anomalous points
    - All observed points
    
    - results: output from detect_wet_dry_periods
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    original = results['original_data'].copy()
    forecast = results['forecast'].copy()
    anomaly_labels = results['anomaly_labels']
    
    # Add 'hour' column
    original['hour'] = original['ds'].dt.hour
    forecast['hour'] = forecast['ds'].dt.hour
    
    # Non-anomalous points
    normal_data = original[~anomaly_labels]
    
    # Compute average flow per hour
    avg_all = original.groupby('hour')['y'].mean()
    avg_normal = normal_data.groupby('hour')['y'].mean()
    avg_forecast = forecast.groupby('hour')['yhat'].mean()
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(avg_forecast.index, avg_forecast.values, 'o-', color='darkblue', linewidth=2, label='Forecast (Baseline)')
    plt.plot(avg_normal.index, avg_normal.values, 's-', color='orange', linewidth=2, label='Non-Anomalous Observed')
    plt.plot(avg_all.index, avg_all.values, 'd--', color='green', linewidth=2, label='All Observed Points')
    
    plt.xticks(range(0,24))
    plt.xlabel('Hour of Day')
    plt.ylabel('Flow (MGD)')
    plt.title('Average Diurnal Flow Pattern')
    plt.grid(alpha=0.3)
    plt.legend()

    # Save figure
    output_file = output_path / f'{meter_name}_average_diurnal_pattern.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_average_diurnal_pattern_all(df, output_dir='results/plots', figsize=(14, 6), dpi=300):
    """
    Plot the average diurnal forecast pattern for multiple meters.
    
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df['hour'] = pd.to_datetime(df['DateTime']).dt.hour

    plt.figure(figsize=figsize)

    for meter_name, group in df.groupby('Meter'):
        avg_forecast = group.groupby('hour')['BWF_Residual_Flow'].mean()
        plt.plot(avg_forecast.index, avg_forecast.values, 'o-', linewidth=2, label=meter_name)

    plt.xticks(range(0, 24))
    plt.xlabel('Hour of Day')
    plt.ylabel('Flow (MGD)')
    plt.title('Average Diurnal Forecast Pattern — All Meters')
    plt.grid(alpha=0.3)
    plt.legend()

    output_file = output_path / 'all_meters_average_diurnal_forecast.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_file}")