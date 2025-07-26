import numpy as np
from collections import defaultdict

class BaselineAverageHRModel:
    def __init__(self):
        self.subject_avg_hr = {}
        self.global_avg_hr = 120.0

    def fit(self, train_dataset):
        subject_hrs = defaultdict(list)

        for i in range(len(train_dataset)):
            workout = train_dataset[i]
            subject_id = workout['subject_id']
            hr_values = workout['heart_rate']

            if hasattr(hr_values, 'numpy'):
                hr_values = hr_values.numpy()

            valid_hrs = hr_values[hr_values > 0]
            if len(valid_hrs) > 0:
                subject_hrs[subject_id].extend(valid_hrs)

        for subject_id, hrs in subject_hrs.items():
            if len(hrs) > 0:
                self.subject_avg_hr[subject_id] = np.mean(hrs)

        if subject_hrs:
            all_hrs = [hr for hrs in subject_hrs.values() for hr in hrs]
            self.global_avg_hr = np.mean(all_hrs)

    def forecast_single_workout(self, workout, **kwargs):
        subject_id = workout['subject_id']
        workout_length = len(workout['heart_rate'])

        predicted_hr = self.subject_avg_hr.get(subject_id, self.global_avg_hr)
        heart_rate_pred = np.full(workout_length, predicted_hr, dtype=np.float32)

        return {"heart_rate": heart_rate_pred}


def evaluate_baseline_model(baseline_model, test_dataset, train_workout_ids):
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr

    results = []
    for i in range(len(test_dataset)):
        workout = test_dataset[i]
        workout_id = workout['workout_id']
        is_train = workout_id in train_workout_ids

        prediction = baseline_model.forecast_single_workout(workout)

        # Get true and predicted HR
        true_hr = workout['heart_rate']
        pred_hr = prediction['heart_rate']

        if hasattr(true_hr, 'numpy'):
            true_hr = true_hr.numpy()
        if hasattr(pred_hr, 'numpy'):
            pred_hr = pred_hr.numpy()

        # Calculate metrics
        mae = np.mean(np.abs(true_hr - pred_hr))
        rmse = np.sqrt(np.mean((true_hr - pred_hr) ** 2))
        mape = np.mean(np.abs((true_hr - pred_hr) / true_hr))
        corr = pearsonr(true_hr, pred_hr)[0] if len(true_hr) > 1 else 0.0

        # After 2 minutes (assuming 10-second intervals, so index 12+)
        if len(true_hr) > 12:
            true_hr_after2 = true_hr[12:]
            pred_hr_after2 = pred_hr[12:]
            mae_after2 = np.mean(np.abs(true_hr_after2 - pred_hr_after2))
            rmse_after2 = np.sqrt(np.mean((true_hr_after2 - pred_hr_after2) ** 2))
            mape_after2 = np.mean(np.abs((true_hr_after2 - pred_hr_after2) / true_hr_after2))
            corr_after2 = pearsonr(true_hr_after2, pred_hr_after2)[0] if len(true_hr_after2) > 1 else 0.0
        else:
            mae_after2 = mae
            rmse_after2 = rmse
            mape_after2 = mape
            corr_after2 = corr

        # Match expected column names from your eval file
        metrics = {
            'subject_id': workout['subject_id'],
            'workout_id': workout_id,
            'in_train': is_train,
            'l1': mae,  # MAE
            'l1-after2min': mae_after2,  # MAE after 2min
            'l2': rmse,  # RMSE
            'l2-after2min': rmse_after2,  # RMSE after 2min
            'relative': mape,  # MAPE
            'relative-after2min': mape_after2,  # MAPE after 2min
            'correlation': corr,  # Correlation
            'correlation-after2min': corr_after2  # Correlation after 2min
        }
        results.append(metrics)

    return pd.DataFrame(results)


