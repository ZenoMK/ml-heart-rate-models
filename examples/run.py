import dataclasses
import pandas as pd
import sys
from plotting import plot_workout_predictions
import numpy as np

# Method 1: Add parent directory manually
sys.path.insert(0, '..')  # Go one directory up


from ode.data import WorkoutDataset, WorkoutDatasetConfig, make_dataloaders
from ode.ode import ODEModel, OdeConfig
from ode.trainer import train_ode_model

df = pd.read_feather("../data/apple_format_data.feather")
print(df)


data_config_train = WorkoutDatasetConfig(
    subject_id_column = "user_id",
    workout_id_column = "workout_id",
    time_since_start_column ='time_grid',
    time_of_start_column = 'start_time',
    heart_rate_column = 'heart_rate',
    heart_rate_normalized_column = 'heart_rate_normalized',
    activity_columns = ["horizontal_speed_kph"],
    weather_columns = [],
    history_max_length=512,
)
data_config_test = dataclasses.replace(data_config_train, chunk_size=None, stride=None)

train_dataset = WorkoutDataset(df[df["in_train"]], data_config_train)
test_dataset = WorkoutDataset(df, data_config_test)

train_dataloader, test_dataloader = make_dataloaders(train_dataset, test_dataset, batch_size=16)

ode_config = OdeConfig(
    data_config_train,
    learning_rate=1e-3,
    seed=0,
    n_epochs=10,
    encoder_embedding_dim=8,
    subject_embedding_dim=4,

)

model = ODEModel(
    workouts_info=df[["user_id", "workout_id"]],
    config=ode_config,

)
print(model)

train_workout_ids = set(df[df["in_train"]]["workout_id"].values)
res = train_ode_model(model, train_dataloader, test_dataloader, train_workout_ids)
for i in np.random.choice(len(test_dataset), 10):
    workout = test_dataset[i]
    plot_workout_predictions(model, workout, savepath=f"predictions{i}.png")
