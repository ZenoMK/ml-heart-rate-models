import dataclasses
import pandas as pd
import sys
from plotting import plot_workout_predictions
import numpy as np
import pickle

# Method 1: Add parent directory manually
sys.path.insert(0, '..')  # Go one directory up


from ode.data import WorkoutDataset, WorkoutDatasetConfig, make_dataloaders
from ode.ode import ODEModel, OdeConfig
from ode.trainer import train_ode_model
from ode.eval import *

df = pd.read_feather("../data/apple_format_data.feather")
print("Loading metabolomics data...")
with open("../data/human_omics_mapped_dict.pkl", "rb") as f:
    metabolomics_dict = pickle.load(f)

print(f"Loaded metabolomics data for {len(metabolomics_dict)} subjects")


data_config_train = WorkoutDatasetConfig(
    subject_id_column = "user_id",
    workout_id_column = "workout_id",
    time_since_start_column ='time_grid',
    time_of_start_column = 'start_time',
    heart_rate_column = 'heart_rate',
    heart_rate_normalized_column = 'heart_rate_normalized',
    activity_columns = ["horizontal_speed_kph"],
    weather_columns = [],
    history_max_length=1000,
)
data_config_test = dataclasses.replace(data_config_train, chunk_size=None, stride=None)

train_dataset = WorkoutDataset(df[df["in_train"]], data_config_train)

# TODO whether the eval is actually just on test data
test_dataset = WorkoutDataset(df, data_config_test)

train_dataloader, test_dataloader = make_dataloaders(train_dataset, test_dataset, batch_size=16)

ode_config = OdeConfig(
    data_config_train,
    learning_rate=1e-3,
    seed=0,
    n_epochs=100,
    subject_embedding_dim=16,
    encoder_embedding_dim=32,
    metabolomics_encoder_output_dim=16,
    metabolomics_encoder_hidden_dim=64,
    metabolomics_encoder_input_dim=-1


)

model = ODEModel(
    workouts_info=df[["user_id", "workout_id"]],
    config=ode_config,

)
print(model)

train_workout_ids = set(df[df["in_train"]]["workout_id"].values)
# Use Apple's original training function (no modifications needed):
evaluation_logs = train_ode_model(model, train_dataloader, test_dataloader, train_workout_ids)

# After training is complete, analyze the final epoch results:
final_epoch_results = evaluation_logs[-1]  # Last epoch DataFrame
save_paper_results(evaluation_logs[-1], model_name="My Heart Rate Model (1e-3)")

# Generate paper-ready results:
for i in range(len(test_dataset)):
    workout = test_dataset[i]
    plot_workout_predictions(model, workout, savepath=f"results/predplots/predictions{i}.png")
