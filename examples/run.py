import dataclasses
import sys
from plotting import plot_workout_predictions
import pickle


# Method 1: Add parent directory manually
sys.path.insert(0, '..')  # Go one directory up


from ode.data import WorkoutDataset, WorkoutDatasetConfig, make_dataloaders
from ode.ode import ODEModel, OdeConfig
from ode.trainer import train_ode_model
from ode.eval import *
from ode.baseline_average_hr import BaselineAverageHRModel, evaluate_baseline_model
from ode.embedding_regressions import train_regression_models, get_embeddings_and_targets

df = pd.read_feather("../data/apple_format_data.feather")
print("Loading metabolomics data...")
with open("../data/human_omic_mapped_dict.pkl", "rb") as f:
    metabolomics_dict = pickle.load(f)

gods_df = pd.read_csv("../data/gods_trajectories.csv")
workout_to_gods_mapping = {wid: str(uid) for wid, uid in zip(df['workout_id'], df['user_id'])}

sample_gods_id = list(metabolomics_dict.keys())[0]
sample_metabolomics = metabolomics_dict[sample_gods_id].get('m')
if sample_metabolomics is not None:
    metabolomics_dim = len(sample_metabolomics)
    print(f"Metabolomics tensor dimension: {metabolomics_dim}")
else:
    print("ERROR: No 'm' key found in metabolomics data")
    metabolomics_dim = 0


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

train_dataloader, test_dataloader = make_dataloaders(train_dataset, test_dataset, batch_size=8)

ode_config = OdeConfig(
    data_config_train,
    learning_rate=1e-3,
    seed=0,
    n_epochs=100,
    subject_embedding_dim=16,
    encoder_embedding_dim=16,
    metabolomics_encoder_output_dim=8,
    metabolomics_encoder_hidden_dim=128,
    metabolomics_encoder_input_dim=metabolomics_dim


)

model = ODEModel(
    workouts_info=df[["user_id", "workout_id"]],
    config=ode_config,
    metabolomics_dict=metabolomics_dict,
    workout_to_gods_mapping=workout_to_gods_mapping
)
print(model)

train_workout_ids = set(df[df["in_train"]]["workout_id"].values)
# Use Apple's original training function (no modifications needed):
evaluation_logs = train_ode_model(model, train_dataloader, test_dataloader, train_workout_ids)

# After training is complete, analyze the final epoch results:
final_epoch_results = evaluation_logs[-1]  # Last epoch DataFrame
save_paper_results(evaluation_logs[-1], model_name="Heart Rate Model with Metabolomics (1e-3)(256_64_8) batch 8")

baseline_model = BaselineAverageHRModel()
baseline_model.fit(train_dataset)
baseline_results = evaluate_baseline_model(baseline_model, test_dataset, train_workout_ids)
save_paper_results(baseline_results, model_name="Baseline: Subject Average HR")


print("Extracting embeddings and targets...")
embeddings, bmis, ages = get_embeddings_and_targets(model, df, test_dataset)  # or train_dataset
print(f"Found {len(embeddings)} subjects with complete data")

print("\nTraining regression models...")
train_regression_models(embeddings, bmis, ages)

# Generate paper-ready results:
for i in range(len(test_dataset)):
    workout = test_dataset[i]
    plot_workout_predictions(model, workout, savepath=f"results/predplots/predictions{i}_(M_1e-3).png")
