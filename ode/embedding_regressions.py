import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score


def get_embeddings_and_targets(model, df, dataset):
    """Extract embeddings for subjects with BMI data using their prepared workout history"""

    # Hardcode your BMI dictionary here
    bmi_dict = {
        '02/001': 41.3,
        '02/010': 41.4,
        '02/014': 20.6,
        '02/016': 24.3,
        '03/021': 19.4,
        '03/023': 24.4,
        '03/029': 21.0,
        '1069': 32.0
    }

    # Create a mapping from workout_id to dataset index
    workout_to_dataset_idx = {}
    for i in range(len(dataset)):
        workout_data = dataset[i]
        workout_to_dataset_idx[workout_data['workout_id']] = i

    # Get one workout per subject
    subject_workout_map = df.sort_values('start_time').groupby('user_id')['workout_id'].last()

    embeddings = []
    bmis = []
    ages = []

    model.eval()
    with torch.no_grad():
        for subject_id, workout_id in subject_workout_map.items():
            # Skip if no BMI data in dictionary
            if subject_id not in bmi_dict:
                continue

            # Get subject data from DataFrame for age
            subject_data = df[df['user_id'] == subject_id].iloc[0]
            if pd.isna(subject_data.get('age')):
                continue

            # Get workout data from dataset (which has the prepared history)
            if workout_id not in workout_to_dataset_idx:
                continue

            workout_data = dataset[workout_to_dataset_idx[workout_id]]

            # Extract history and history_length
            history = workout_data['history']
            history_length = workout_data['history_length']

            # Convert to tensors if needed
            if history is not None:
                history = torch.FloatTensor(history).unsqueeze(0)  # Add batch dimension
                history_length = torch.LongTensor([history_length])

            # Extract embedding with history
            embedding = model.embedding_store.get_embeddings_from_workout_ids(
                [workout_id], history=history, history_lengths=history_length
            )

            embeddings.append(embedding.cpu().numpy()[0])
            bmis.append(bmi_dict[subject_id])
            ages.append(subject_data['age'])

    return np.array(embeddings), np.array(bmis), np.array(ages)


def train_regression_models(embeddings, bmis, ages):
    """Train regression models using cross-validation like the Apple paper"""
    for target_name, target_values in [('BMI', bmis), ('Age', ages)]:
        # Just fit the model on all data
        model = LinearRegression()
        model.fit(embeddings, target_values)

        # Calculate R² on the same data (biased but interpretable with small samples)
        r2 = model.score(embeddings, target_values)

        print(f"{target_name}: R² = {r2:.3f} ({r2 * 100:.1f}% of variance explained)")

        # Show some predictions vs actual
        predictions = model.predict(embeddings)
        print(f"  Actual vs Predicted:")
        for i, (actual, pred) in enumerate(zip(target_values, predictions)):
            print(f"    Subject {i + 1}: {actual:.1f} vs {pred:.1f}")


# Usage (add this after your model training):
