import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict
import os


def analyze_apple_results(logged_data_for_all_workouts: pd.DataFrame) -> Dict:
    """
    Analyze results from Apple's evaluate() function to get paper-ready metrics.

    Args:
        logged_data_for_all_workouts: DataFrame returned by Apple's evaluate() function

    Returns:
        Dictionary with all paper-ready metrics
    """

    # Filter to test data only (for paper results)
    test_data = logged_data_for_all_workouts[~logged_data_for_all_workouts["in_train"]]
    train_data = logged_data_for_all_workouts[logged_data_for_all_workouts["in_train"]]

    def calculate_stats(values):
        """Calculate mean, 95% CI, median, and IQR for a set of values."""
        values = values.dropna()
        if len(values) == 0:
            return {
                'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'median': np.nan, 'q25': np.nan, 'q75': np.nan, 'n': 0
            }

        mean_val = np.mean(values)
        sem_val = stats.sem(values)
        ci_lower, ci_upper = stats.t.interval(0.95, len(values) - 1, loc=mean_val, scale=sem_val)
        median_val = np.median(values)
        q25, q75 = np.percentile(values, [25, 75])

        return {
            'mean': mean_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'median': median_val,
            'q25': q25,
            'q75': q75,
            'n': len(values)
        }

    # Calculate all metrics for test data
    metrics = {
        'test': {
            'mae': calculate_stats(test_data['l1']),  # l1 = MAE in Apple's code
            'mae_after2min': calculate_stats(test_data['l1-after2min']),
            'rmse': calculate_stats(test_data['l2']),  # l2 = RMSE in Apple's code
            'rmse_after2min': calculate_stats(test_data['l2-after2min']),
            'mape': calculate_stats(test_data['relative'] * 100),  # Convert to percentage
            'mape_after2min': calculate_stats(test_data['relative-after2min'] * 100),
            'correlation': calculate_stats(test_data['correlation']),  # Add this line
            'correlation_after2min': calculate_stats(test_data['correlation-after2min']),
        },
        'train': {
            'mae': calculate_stats(train_data['l1']),
            'mape': calculate_stats(train_data['relative'] * 100),
        }
    }

    return metrics


def save_paper_results(logged_data_for_all_workouts: pd.DataFrame,
                       model_name: str = "model",
                       save_dir: str = "results"):
    """
    Simple function to analyze Apple's results and save everything for your paper.

    Usage after training:
        evaluation_logs = train_ode_model(...)
        save_paper_results(evaluation_logs[-1], model_name="My Model")

    Args:
        logged_data_for_all_workouts: DataFrame from Apple's evaluate() function
        model_name: Name for your model
        save_dir: Directory to save results
    """

    # Create results directory
    os.makedirs(save_dir, exist_ok=True)

    # Analyze results
    metrics = analyze_apple_results(logged_data_for_all_workouts)
    test_mae = metrics['test']['mae']
    test_mae_after2min = metrics['test']['mae_after2min']
    test_mape = metrics['test']['mape']
    test_mape_after2min = metrics['test']['mape_after2min']
    test_correlation = metrics['test']['correlation']
    test_correlation_after2min = metrics['test']['correlation_after2min']
    train_mae = metrics['train']['mae']
    train_mape = metrics['train']['mape']

    # Create results text
    results_text = f"""PAPER RESULTS FOR: {model_name}
{"=" * 60}

TRAINING PERFORMANCE:
• Train MAE: {train_mae['mean']:.3f} BPM (= {train_mape['mean']:.3f}%)

TEST PERFORMANCE:
• Test MAE: {test_mae['mean']:.3f} BPM (= {test_mape['mean']:.3f}%)
• Test MAE after 2min: {test_mae_after2min['mean']:.3f} BPM

PAPER-READY STATISTICS (N = {test_mae['n']} test workouts):
• Mean MAE: {test_mae['mean']:.2f} BPM [95% CI: {test_mae['ci_lower']:.2f}–{test_mae['ci_upper']:.2f}]
• Median MAE: {test_mae['median']:.2f} BPM [IQR: {test_mae['q25']:.2f}–{test_mae['q75']:.2f}]
• Median MAPE: {test_mape['median']:.2f} % [IQR: {test_mape['q25']:.2f}–{test_mape['q75']:.2f}]
• Correlation Median: {test_correlation['median']:.2f} % [IQR: {test_correlation['q25']:.2f}–{test_correlation['q75']:.2f}]
 Latex table:
 {test_mae['mean']:.2f} [{test_mae['ci_lower']:.2f}–{test_mae['ci_upper']:.2f}] & {test_mae['median']:.2f} [{test_mae['q25']:.2f}–{test_mae['q75']:.2f}] & {test_mape['median']:.2f} [{test_mape['q25']:.2f}–{test_mape['q75']:.2f}] & {test_correlation['median']:.2f} [{test_correlation['q25']:.2f}–{test_correlation['q75']:.2f}]
 

• Mean MAE after 2min: {test_mae_after2min['mean']:.2f} BPM [95% CI: {test_mae_after2min['ci_lower']:.2f}–{test_mae_after2min['ci_upper']:.2f}]
• Median MAE after 2min: {test_mae_after2min['median']:.2f} BPM [IQR: {test_mae_after2min['q25']:.2f}–{test_mae_after2min['q75']:.2f}]
• Median MAPE after 2min: {test_mape_after2min['median']:.2f} % [IQR: {test_mape_after2min['q25']:.2f}–{test_mape_after2min['q75']:.2f}]
• Correlation Median: {test_correlation_after2min['median']:.2f} % [IQR: {test_correlation_after2min['q25']:.2f}–{test_correlation_after2min['q75']:.2f}]
Latex table:
 {test_mae_after2min['mean']:.2f} [{test_mae_after2min['ci_lower']:.2f}–{test_mae_after2min['ci_upper']:.2f}] & {test_mae_after2min['median']:.2f} [{test_mae_after2min['q25']:.2f}–{test_mae_after2min['q75']:.2f}] & {test_mape_after2min['median']:.2f} [{test_mape_after2min['q25']:.2f}–{test_mape_after2min['q75']:.2f}] & {test_correlation_after2min['median']:.2f} [{test_correlation_after2min['q25']:.2f}–{test_correlation_after2min['q75']:.2f}]


"""

    # Print results
    print(results_text)

    # Save text results
    safe_name = model_name.replace(' ', '_').replace('/', '_')
    results_file = os.path.join(save_dir, f"{safe_name}_results.txt")
    with open(results_file, 'w') as f:
        f.write(results_text)



