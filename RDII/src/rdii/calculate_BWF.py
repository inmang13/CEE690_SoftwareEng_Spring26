import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


def detect_wet_dry_periods(df, k=3, max_iterations=10, threshold=0.5):
    """
    Iterative anomaly detection for I/I identification with visualization

    """
    
    # 1. INITIALIZE
    training_data = df[['DateTime', 'Flow_MGD_BWI_Corrected']].copy()
    training_data.rename(columns={
        'DateTime': 'ds',
        'Flow_MGD_BWI_Corrected': 'y'
    }, inplace=True)
    
    training_data = training_data.dropna().reset_index(drop=True)  # RESET INDEX
    original_data = training_data.copy()
    
    # Setup holidays
    us_holidays = holidays.US(years=range(
        training_data['ds'].min().year,
        training_data['ds'].max().year + 1
    ))
    
    holiday_df = pd.DataFrame({
        'ds': pd.to_datetime(list(us_holidays.keys())),
        'holiday': list(us_holidays.values())
    })
    
    # Track iterations for visualization
    iteration_history = []
    previous_forecast = None
    iteration = 0
    
    print(f"Starting with {len(training_data)} data points")
    
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}:")
        
        # 2. TRAIN BWF RECONSTRUCTION MODEL (Prophet)
        model = Prophet(
            growth='linear',
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=holiday_df,
            seasonality_mode='additive'
        )
        
        model.fit(training_data)
        
        # Predict on ORIGINAL data (not just training data)
        forecast = model.predict(original_data[['ds']])
        
        # 3. CALCULATE RESIDUALS (on training data only)
        # FIX: Use merge to align data properly
        train_with_forecast = training_data.merge(
            forecast[['ds', 'yhat']], 
            on='ds', 
            how='left'
        )
        residuals = train_with_forecast['y'].values - train_with_forecast['yhat'].values
        
        mu = residuals.mean()
        sigma = residuals.std()
        
        print(f"  Residual stats: μ={mu:.4f}, σ={sigma:.4f}")
        
        # 4. ANOMALY DETECTION (k-sigma rule)
        lower_bound = mu - k * sigma
        upper_bound = mu + k * sigma
        anomalies = (residuals < lower_bound) | (residuals > upper_bound)
        n_anomalies = anomalies.sum()
        
        print(f"  Anomalies detected: {n_anomalies} ({100*n_anomalies/len(training_data):.1f}%)")
        print(f"  Training data remaining: {len(training_data) - n_anomalies}")
        
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
        # FIX: Keep track using datetime instead of index
        anomaly_dates = train_with_forecast.loc[anomalies, 'ds']
        training_data = training_data[~training_data['ds'].isin(anomaly_dates)].reset_index(drop=True)
        
        previous_forecast = forecast
        iteration += 1
    
    print(f"\nCompleted after {iteration + 1} iterations")
    
    # 7. FINAL ANOMALY CLASSIFICATION on all original data
    final_forecast = iteration_history[-1]['forecast']
    
    # Merge to align properly
    final_data = original_data.merge(
        final_forecast[['ds', 'yhat']], 
        on='ds', 
        how='left'
    )
    final_residuals = final_data['y'].values - final_data['yhat'].values
    final_mu = iteration_history[-1]['mu']
    final_sigma = iteration_history[-1]['sigma']
    final_lower = final_mu - k * final_sigma
    final_upper = final_mu + k * final_sigma
    
    wet_labels = (final_residuals < final_lower) | (final_residuals > final_upper)
    
    # 8. ISOLATED POINT REMOVAL
    wet_labels_cleaned = remove_isolated_points(wet_labels, window_size=12)
    
    print(f"\nFinal classification:")
    print(f"  Wet periods: {wet_labels_cleaned.sum()} points ({100*wet_labels_cleaned.sum()/len(wet_labels_cleaned):.1f}%)")
    print(f"  Dry periods: {(~wet_labels_cleaned).sum()} points ({100*(~wet_labels_cleaned).sum()/len(wet_labels_cleaned):.1f}%)")
    
    # 9. CALCULATE I/I FLOW
    ii_flow = final_data['y'].values - final_data['yhat'].values
    ii_flow[~wet_labels_cleaned] = 0  # No I/I during dry periods
    
    return {
        'wet_dry_labels': wet_labels_cleaned,
        'ii_flow': ii_flow,
        'forecast': final_forecast,
        'original_data': original_data,
        'iteration_history': iteration_history,
        'final_bounds': (final_lower, final_upper)
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
    """
    Majority voting with sliding window to remove isolated points
    """
    labels = np.array(labels)
    cleaned_labels = labels.copy()
    
    half_window = window_size // 2
    
    for i in range(len(labels)):
        start = max(0, i - half_window)
        end = min(len(labels), i + half_window + 1)
        window = labels[start:end]
        
        # Majority vote
        if window.sum() > len(window) / 2:
            cleaned_labels[i] = True  # Wet
        else:
            cleaned_labels[i] = False  # Dry
    
    return cleaned_labels




