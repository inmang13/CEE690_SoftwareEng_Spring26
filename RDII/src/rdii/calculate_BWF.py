# MAIN ALGORITHM OUTLINE
import holidays
import pandas as pd
import numpy as np  
from prophet import Prophet
from scipy import stats
from scipy.stats import pearsonr
import warnings
import sys
import json
from pathlib import Path
warnings.filterwarnings('ignore')
from rdii.plots import plot_final_classification, plot_iteration_statistics,plot_average_diurnal_pattern_all



def detect_wet_dry_periods(df, k=2.5,sigma_method='robust', max_iterations=50, threshold=0.05):
    """
    Iterative anomaly detection for I/I identification
    """
    
    # 1. INITIALIZE
    training_data = df[['DateTime', 'Flow_MGD_GWI_Corrected']].copy()

    training_data['DateTime'] = pd.to_datetime(training_data['DateTime'])
    
    training_data.rename(columns={
            'DateTime': 'ds',
            'Flow_MGD_GWI_Corrected': 'y'
        }, inplace=True)
    
    training_data = training_data.dropna()
    original_data = training_data.copy()

    ds_min = training_data['ds'].min()
    ds_max = training_data['ds'].max()

    # ensure min/max are valid datetimes
    if pd.isna(ds_min) or pd.isna(ds_max):
        raise ValueError("DateTime column contains NaT values.")

    us_holidays = holidays.US(years=range(ds_min.year, ds_max.year + 1))


    holiday_df = pd.DataFrame({
        'ds': pd.to_datetime(list(us_holidays.keys())),
        'holiday': list(us_holidays.values())
    })

    # Track iterations for visualization
    iteration_history = []
    previous_forecast = None

    iteration = 0
    
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}:")

        # 2. TRAIN BWF RECONSTRUCTION MODEL (Prophet

        if iteration <= 1:
            fit_data = training_data.set_index('ds').resample('D').sum().reset_index().dropna()
            daily_seasonality = False
        elif iteration <= 4:
            fit_data = training_data.set_index('ds').resample('1h').mean().reset_index().dropna()
            daily_seasonality = True
        else:
            fit_data = training_data  # full resolution for final fit only
        
        model = Prophet(
            # Trend component (piecewise linear)
            growth='linear',
            changepoint_prior_scale=0.05,
            yearly_seasonality=6,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=holiday_df
        )

        model.fit(fit_data, iter=200)
        
        #model.fit(training_data,iter=200)


        # 3. CALCULATE RESIDUALS (on training data only)
        # potentially revisit
        # Predict at fit_data resolution for residual calculation
        fit_forecast = model.predict(fit_data[['ds']])
        residuals = fit_data['y'].values - fit_forecast['yhat'].values

        # Predict on ORIGINAL data (not just training data)
        forecast = model.predict(original_data[['ds']])
        
        if sigma_method == 'standard':
            mu = residuals.mean()
            sigma = residuals.std()

        
        elif sigma_method == 'robust':
            mu = np.median(residuals)
            sigma = 1.4826 * np.median(np.abs(residuals - mu))
            
            if sigma == 0:
                sigma = np.std(residuals)
                print(f"  Residual stats: μ={mu:.4f}, σ={sigma:.4f}")

        
        # 4. ANOMALY DETECTION (k-sigma rule)
        lower_bound = mu - k * sigma
        upper_bound = mu + k * sigma
        anomalies = (residuals < lower_bound) | (residuals > upper_bound)
        n_anomalies = anomalies.sum()
        print(f"  Anomalies detected: {n_anomalies} ({100*n_anomalies/len(training_data):.1f}%)")

        # Store iteration info
        iteration_history.append({
            'iteration': iteration,
            'forecast': forecast.copy(),
            'n_points': len(training_data),
            'n_anomalies': n_anomalies,
            'mu': mu,
            'sigma': sigma,
            'bounds': (lower_bound, upper_bound)
        })

        # 5. CHECK TERMINATION CONDITIONS
        should_terminate, reason = check_termination(
            residuals, forecast, previous_forecast, threshold
        )
        
        if should_terminate:
            print(f"\nTerminating: {reason}")
            break
        
        if n_anomalies == 0:
            print("\nTerminating: No more anomalies detected")
            break

        # 6. REMOVE ANOMALIES FOR NEXT ITERATION
        if iteration <= 2:
            # anomalies are indexed against daily fit_data
            # must map back to 15-min training_data by date
            anomalous_dates = fit_data['ds'][anomalies]
            training_data = training_data[~training_data['ds'].dt.date.isin(anomalous_dates.dt.date)]
        elif iteration <= 4:
            # anomalies are indexed against hourly fit_data
            # map back to 15-min training_data by hour
            anomalous_hours = fit_data['ds'][anomalies]
            training_data = training_data[~training_data['ds'].dt.floor('1h').isin(anomalous_hours)]
        else:
            # anomalies are indexed directly against 15-min training_data
            training_data = training_data.loc[~anomalies]
        
        previous_forecast = forecast
        iteration += 1



    
    print(f"\nCompleted after {iteration + 1} iterations")
    
    # 7. FINAL ANOMALY CLASSIFICATION on all original data
    final_forecast = iteration_history[-1]['forecast']
    final_residuals = original_data['y'].values - final_forecast['yhat'].values
    final_mu = iteration_history[-1]['mu']
    final_sigma = iteration_history[-1]['sigma']
    final_lower = final_mu - k * final_sigma
    final_upper = final_mu + k * final_sigma
    
    anomaly_labels = (final_residuals < final_lower) | (final_residuals > final_upper)
    
    # 8. ISOLATED POINT REMOVAL
    anomaly_labels_cleaned = remove_isolated_points(anomaly_labels.astype(np.bool_), window_size=12)

    
    print(f"\nFinal classification:")
    print(f"  Anomalous points: {anomaly_labels_cleaned.sum()} ({100*anomaly_labels_cleaned.sum()/len(anomaly_labels_cleaned):.1f}%)")
    print(f"  Normal points: {(~anomaly_labels_cleaned).sum()} ({100*(~anomaly_labels_cleaned).sum()/len(anomaly_labels_cleaned):.1f}%)")
    
    # 9. CALCULATE RESIDUAL FLOW (anomalous deviation from forecast)
    residual_flow = original_data['y'].values - final_forecast['yhat'].values
    residual_flow[~anomaly_labels_cleaned] = 0  # Zero out non-anomalous points
    
    return {
        'anomaly_labels': anomaly_labels_cleaned, #Boolean Array where True = anomalous, False = normal
        'residual_flow': residual_flow, #Array of flow deviations from forecast, with non-anomalous points set to 0
        'forecast': final_forecast,    #DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper'] containing the final forecast values
        'original_data': original_data, #DataFrame with columns ['ds', 'y'] containing the original input data used for forecasting
        'iteration_history': iteration_history, #List of dictionaries containing info about each iteration (forecast, residual stats, anomaly counts, etc.) 
        'final_bounds': (final_lower, final_upper),  #Tuple containing the final lower and upper bounds used for anomaly classification in the last iteration
        'k': k  #The k value used for the final anomaly classification
    }




def check_termination(residuals, current_forecast, previous_forecast, threshold):
    """
    Check if any termination condition is met

    """
    
    # Condition 1: Normality test (Anderson-Darling)
    result = stats.anderson(residuals, dist='norm')
    # If statistic < critical value, data is normal
    if result.statistic < result.critical_values[2]:  # 5% significance
        return True, "Residuals pass normality test"
    
    # Condition 2: Maximum residual threshold
    max_abs_residual = np.abs(residuals).max()
    if max_abs_residual < threshold:
        return True, f"Max residual ({max_abs_residual:.4f}) below threshold ({threshold})"
    
    # Condition 3: High correlation between successive forecasts
    if previous_forecast is not None:
        correlation, _ = pearsonr(
            current_forecast['yhat'].values,
            previous_forecast['yhat'].values
        )
        if correlation > 0.999:
            return True, f"High correlation ({correlation:.6f}) between successive forecasts"
    
    return False, None


def remove_isolated_points(labels, window_size=12):
    n = len(labels)
    cleaned = np.empty(n, dtype=bool)
    half = window_size // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        count_true = 0
        for j in range(start, end):
            if labels[j]:
                count_true += 1
        if count_true > (end - start) / 2:
            cleaned[i] = True
        else:
            cleaned[i] = False
    return cleaned

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
        plots_dir= project_root / config['paths']['plots_dir']
        combined_file = processed_dir / config['paths']['combined_filename']  
        processed_dir.mkdir(parents=True, exist_ok=True)

        combined_file = processed_dir / config['paths']['combined_filename']
        cleaned_file = processed_dir / config['paths']['cleaned_filename']
        gwi_removed_file = processed_dir / config['paths']['gwi_removed_filename']


        # Load cleaned data
        try:
            data = pd.read_csv(gwi_removed_file)
            print("Loading cleaned data with GWI removed...")
        except Exception as e:
            print(f"✗ Failed to load cleaned data: {e}")
            sys.exit(1)

 
        results = []
        for meter_name, group in data.groupby('Meter'):
            print(f"\nProcessing {meter_name} meter...")
            result = detect_wet_dry_periods(group, 
                k=config['bwf']['k'], 
                sigma_method=config['bwf']['sigma_method'], 
                max_iterations=config['bwf']['max_iterations'], 
                threshold=config['bwf']['threshold']
            )
            
            plot_final_classification(result, meter_name, output_dir=plots_dir)
            plot_iteration_statistics(result,meter_name, output_dir=plots_dir)
            plot_average_diurnal_pattern_all(result,meter_name, output_dir=plots_dir)

            result_df = pd.DataFrame({
                'DateTime': result['original_data']['ds'],
                'Flow_MGD_GWI_Corrected': result['original_data']['y'],
                'BWF_Anomaly': result['anomaly_labels'],
                'BWF_Residual_Flow': result['residual_flow']
            })
            result_df['Meter'] = meter_name
            results.append(result_df)

        final_results = pd.concat(results, ignore_index=True)
        final_results.to_csv(processed_dir / config['paths']['bwf_results_filename'], index=False)
        print(f"✓ Saved BWF results to: {processed_dir / config['paths']['bwf_results_filename']}")

if __name__ == "__main__":
        config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
        main(config_file)



