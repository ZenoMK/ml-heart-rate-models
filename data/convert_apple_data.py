#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import interpolate
import ast
import json
from typing import List, Tuple


def parse_list_column(list_str: str) -> List:
    """Parse string representation of list back to actual list."""
    try:
        if isinstance(list_str, str):
            return ast.literal_eval(list_str)
        return list_str if isinstance(list_str, list) else []
    except:
        return []


def interpolate_to_uniform_grid(times: List[float], values: List[float],
                                target_interval: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate irregular time series to uniform time grid.

    Args:
        times: List of time points in seconds
        values: List of corresponding values
        target_interval: Target sampling interval in seconds (default: 10s)

    Returns:
        Tuple of (uniform_times, interpolated_values)
    """
    if len(times) < 2 or len(values) < 2:
        return np.array([]), np.array([])

    # Create uniform time grid
    start_time = min(times)
    end_time = max(times)
    uniform_times = np.arange(start_time, end_time + target_interval, target_interval)

    # Interpolate values to uniform grid
    try:
        f = interpolate.interp1d(times, values, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
        interpolated_values = f(uniform_times)
        return uniform_times, interpolated_values
    except:
        return np.array([]), np.array([])


def normalize_heart_rate(hr_values: np.ndarray, min_hr: float = 40, max_hr: float = 200) -> np.ndarray:
    """
    Normalize heart rate values for neural networks.
    Apple's models typically use min-max normalization.
    """
    hr_clipped = np.clip(hr_values, min_hr, max_hr)
    return (hr_clipped - min_hr) / (max_hr - min_hr)


def preview_filtering_effects(input_csv: str):
    """
    Preview the effects of filtering without actually converting the data.
    Useful to understand what will be removed.

    Args:
        input_csv: Path to your gods_trajectories.csv file
    """
    print("PREVIEW: Loading data to analyze filtering effects...")
    df = pd.read_csv(input_csv)

    if not validate_input_data(df):
        return

    print(f"\nOriginal dataset: {len(df)} workouts from {df['gods21_id'].nunique()} users")

    # Analyze duration distribution
    print(f"\nDuration analysis:")
    print(f"- Workouts < 15 min: {(df['duration_minutes'] < 15).sum()}")
    print(f"- Workouts 15-120 min: {((df['duration_minutes'] >= 15) & (df['duration_minutes'] <= 120)).sum()}")
    print(f"- Workouts > 120 min: {(df['duration_minutes'] > 120).sum()}")

    # Analyze user workout counts
    user_counts = df['gods21_id'].value_counts()
    print(f"\nUser workout count analysis:")
    print(f"- Users with ≥10 workouts: {(user_counts >= 10).sum()}")
    print(f"- Users with <10 workouts: {(user_counts < 10).sum()}")
    print(f"- Distribution of workout counts per user:")
    for threshold in [1, 5, 10, 20, 50]:
        count = (user_counts >= threshold).sum()
        print(f"  - ≥{threshold} workouts: {count} users")

    # Show what filtering would do
    duration_filtered = df[(df['duration_minutes'] >= 15) & (df['duration_minutes'] <= 120)]
    user_workout_counts_filtered = duration_filtered['gods21_id'].value_counts()
    users_with_min_workouts = user_workout_counts_filtered[user_workout_counts_filtered >= 10].index
    final_preview = duration_filtered[duration_filtered['gods21_id'].isin(users_with_min_workouts)]

    print(f"\nFiltering preview:")
    print(f"- After duration filter: {len(duration_filtered)} workouts")
    print(
        f"- After user count filter: {len(final_preview)} workouts from {final_preview['gods21_id'].nunique()} users")
    print(
        f"- Total reduction: {len(df) - len(final_preview)} workouts ({(len(df) - len(final_preview)) / len(df) * 100:.1f}%)")


def validate_input_data(df: pd.DataFrame) -> bool:
    """
    Validate that the input dataframe has all required columns.

    Args:
        df: Input dataframe to validate

    Returns:
        True if valid, False otherwise
    """
    required_columns = [
        'log_id', 'gods21_id', 'duration_minutes',
        'measurements', 'relative_times_seconds', 'speed',
        'activity_name', 'start_time', 'age', 'gender',
        'max_hr', 'min_hr'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return False

    print(f"✓ All required columns present")

    # Check for empty essential columns
    essential_cols = ['log_id', 'gods21_id', 'duration_minutes', 'measurements', 'relative_times_seconds']
    for col in essential_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"WARNING: Column '{col}' has {null_count} null values")

    return True


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply filtering criteria to the workout data.

    Filters:
    1. Remove workouts < 15 minutes or > 120 minutes duration
    2. Remove users with < 10 total workouts

    Args:
        df: Input dataframe with workout data

    Returns:
        Filtered dataframe
    """
    print("Applying filters...")
    initial_count = len(df)
    initial_users = df['gods21_id'].nunique()

    print(f"Initial dataset: {initial_count} workouts from {initial_users} users")

    # Show duration distribution before filtering
    duration_stats = df['duration_minutes'].describe()
    print(
        f"Duration statistics (minutes): min={duration_stats['min']:.1f}, max={duration_stats['max']:.1f}, mean={duration_stats['mean']:.1f}")

    # Filter 1: Duration between 15-120 minutes
    too_short = (df['duration_minutes'] < 15).sum()
    too_long = (df['duration_minutes'] > 120).sum()

    duration_filtered = df[(df['duration_minutes'] >= 15) & (df['duration_minutes'] <= 120)]
    print(
        f"Duration filter (15-120 min): removed {too_short} too short + {too_long} too long = {too_short + too_long} workouts")
    print(f"Remaining after duration filter: {len(duration_filtered)}/{initial_count} workouts")

    # Filter 2: Users with at least 10 workouts
    # Count workouts per user in duration-filtered data
    user_workout_counts = duration_filtered['gods21_id'].value_counts()
    users_with_min_workouts = user_workout_counts[user_workout_counts >= 10].index
    users_excluded = user_workout_counts[user_workout_counts < 10]

    print(f"User workout distribution after duration filter:")
    print(f"- Users with ≥10 workouts: {len(users_with_min_workouts)}")
    print(f"- Users with <10 workouts: {len(users_excluded)} (excluded)")
    if len(users_excluded) > 0:
        print(f"- Excluded users' workout counts: {users_excluded.tolist()}")

    final_filtered = duration_filtered[duration_filtered['gods21_id'].isin(users_with_min_workouts)]

    workouts_removed_by_user_filter = len(duration_filtered) - len(final_filtered)

    print(
        f"User filter (≥10 workouts): removed {workouts_removed_by_user_filter} workouts from users with <10 workouts")
    print(f"FINAL RESULT: {len(final_filtered)} workouts from {final_filtered['gods21_id'].nunique()} users")
    print(
        f"Total removed: {initial_count - len(final_filtered)} workouts ({(initial_count - len(final_filtered)) / initial_count * 100:.1f}%)")

    if len(final_filtered) > 0:
        # Show final statistics
        final_user_counts = final_filtered['gods21_id'].value_counts()
        print(
            f"Final user workout counts: min={final_user_counts.min()}, max={final_user_counts.max()}, mean={final_user_counts.mean():.1f}")

    return final_filtered


def create_per_user_chronological_split(df: pd.DataFrame, train_ratio: float = 0.8) -> pd.DataFrame:
    """
    Create per-user chronological train/test split.
    For each user, their first 80% of workouts go to train, last 20% to test.

    Args:
        df: DataFrame with workout data (must have 'start_time' column)
        train_ratio: Proportion of each user's data to use for training

    Returns:
        DataFrame with added 'in_train' column
    """
    df = df.copy()

    # Convert start_time to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
        df['start_time'] = pd.to_datetime(df['start_time'])

    # Split chronologically within each user
    df['in_train'] = False

    user_train_counts = []
    user_test_counts = []

    for user_id in df['user_id'].unique():
        user_mask = df['user_id'] == user_id
        user_data = df[user_mask].sort_values('start_time')

        n_train = int(len(user_data) * train_ratio)
        train_indices = user_data.index[:n_train]  # First n_train workouts for this user
        df.loc[train_indices, 'in_train'] = True

        user_train_counts.append(n_train)
        user_test_counts.append(len(user_data) - n_train)

    print(f"Per-user chronological split:")
    print(f"  - Total training workouts: {df['in_train'].sum()} ({df['in_train'].mean() * 100:.1f}%)")
    print(f"  - Total test workouts: {(~df['in_train']).sum()} ({(~df['in_train']).mean() * 100:.1f}%)")
    print(f"  - Users: {len(df['user_id'].unique())}")
    print(f"  - Avg train workouts per user: {sum(user_train_counts) / len(user_train_counts):.1f}")
    print(f"  - Avg test workouts per user: {sum(user_test_counts) / len(user_test_counts):.1f}")
    print(f"  - Every user contributes to both train and test sets!")

    # Show example for first user
    first_user = df['user_id'].unique()[0]
    first_user_data = df[df['user_id'] == first_user].sort_values('start_time')
    train_workouts = first_user_data[first_user_data['in_train']]
    test_workouts = first_user_data[~first_user_data['in_train']]

    if len(train_workouts) > 0 and len(test_workouts) > 0:
        train_date_range = f"{train_workouts['start_time'].min().strftime('%Y-%m-%d')} to {train_workouts['start_time'].max().strftime('%Y-%m-%d')}"
        test_date_range = f"{test_workouts['start_time'].min().strftime('%Y-%m-%d')} to {test_workouts['start_time'].max().strftime('%Y-%m-%d')}"
        print(f"  - Example - User {first_user}:")
        print(f"    • Train: {len(train_workouts)} workouts ({train_date_range})")
        print(f"    • Test: {len(test_workouts)} workouts ({test_date_range})")

    return df

def convert_fitbit_to_apple_format(input_csv: str, output_feather: str,
                                  apply_filtering: bool = True, train_ratio: float = 0.8):
    """
    Convert Fitbit CSV format to Apple heart rate model format.

    Args:
        input_csv: Path to your gods_trajectories.csv file
        output_feather: Output path for the converted data in feather format
        apply_filtering: Whether to apply duration and user count filters
    """

    print("Loading Fitbit data...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} workouts from {df['gods21_id'].nunique()} users")

    # Validate input data
    if not validate_input_data(df):
        return None

    # Apply filtering if requested
    if apply_filtering:
        df = apply_filters(df)
        if len(df) == 0:
            print("ERROR: No workouts remain after filtering!")
            return None

    converted_workouts = []

    print(f"Processing {len(df)} workouts...")

    for idx, row in df.iterrows():
        try:
            # Parse the stored lists
            hr_measurements = parse_list_column(row['measurements'])
            time_measurements = parse_list_column(row['relative_times_seconds'])
            speed_measurements = parse_list_column(row['speed'])

            if not hr_measurements or not time_measurements:
                continue

            # Ensure same length
            min_length = min(len(hr_measurements), len(time_measurements))
            if len(hr_measurements) != len(time_measurements):
                raise ValueError("HR measurements and time_measurements length mismatch")
            hr_measurements = hr_measurements[:min_length]
            time_measurements = time_measurements[:min_length]

            # Interpolate to uniform 10-second grid (as per Apple's approach)
            uniform_times, uniform_hr = interpolate_to_uniform_grid(
                time_measurements, hr_measurements, target_interval=10.0
            )

            if len(uniform_hr) == 0:
                continue

            # Interpolate speed data to same grid
            uniform_speed = np.zeros_like(uniform_hr)
            if speed_measurements and len(speed_measurements) > 1:
                # Speed is per-minute, so we need to interpolate differently
                speed_times = np.arange(0, len(speed_measurements)) * 60  # Every minute
                if len(speed_times) > 1:
                    try:
                        # TODO consider again if this is 100% right
                        f_speed = interpolate.interp1d(speed_times, speed_measurements,
                                                       kind='linear', bounds_error=False,
                                                       fill_value='extrapolate')
                        uniform_speed = f_speed(uniform_times)
                        uniform_speed = np.clip(uniform_speed, 0, None)  # No negative speeds
                    except:
                        pass

            # Normalize heart rate
            normalized_hr = normalize_heart_rate(uniform_hr, min_hr=row['min_hr'], max_hr= row['max_hr'])

            # Create workout record for Apple format
            workout_data = {
                # Required columns for Apple model
                'user_id': row['gods21_id'],  # User identifier
                'workout_id': f"{row['gods21_id']}_{row['log_id']}",  # Unique workout ID
                'time_seconds': uniform_times.tolist(),  # Uniform time grid
                'heart_rate': uniform_hr.tolist(),  # Heart rate measurements
                'heart_rate_normalized': normalized_hr.tolist(),  # Normalized HR
                'horizontal_speed_kph': uniform_speed.tolist(),  # Convert km/h to m/s

                # Additional metadata that might be useful
                'activity_type': row['activity_name'],
                'duration_minutes': row['duration_minutes'],
                'start_time': row['start_time'],
                'age': row['age'],
                'gender': row['gender'],
                'max_hr_theoretical': row['max_hr'],
                'min_hr_observed': row['min_hr'],

                # Workout statistics
                'workout_length': len(uniform_hr),
                'avg_hr': float(np.mean(uniform_hr)),
                'max_hr_workout': float(np.max(uniform_hr)),
                'avg_speed_kmh': float(np.mean(uniform_speed)) if np.any(uniform_speed > 0) else 0.0
            }

            converted_workouts.append(workout_data)

        except Exception as e:
            print(f"Error processing workout {idx}: {e}")
            continue

    print(f"Successfully converted {len(converted_workouts)} workouts")

    # Create Apple format DataFrame - one row per workout with lists in columns
    apple_format_rows = []

    for workout in converted_workouts:
        apple_row = {
            # Required Apple format columns
            'user_id': workout['user_id'],
            'workout_id': workout['workout_id'],
            'time_grid': workout['time_seconds'],  # List of time points
            'heart_rate': workout['heart_rate'],  # List of HR measurements
            'heart_rate_normalized': workout['heart_rate_normalized'],  # Normalized HR list
            'horizontal_speed_kph': workout['horizontal_speed_kph'],  # List of speeds in m/s

            # Additional metadata
            'activity_type': workout['activity_type'],
            'duration_minutes': workout['duration_minutes'],
            'start_time': workout['start_time'],
            'age': workout['age'],
            'gender': workout['gender'],
            'max_hr_theoretical': workout['max_hr_theoretical'],
            'min_hr_observed': workout['min_hr_observed'],
            'workout_length': workout['workout_length'],
            'avg_hr': workout['avg_hr'],
            'max_hr_workout': workout['max_hr_workout'],
            'avg_speed_kmh': workout['avg_speed_kmh']
        }
        apple_format_rows.append(apple_row)

    # Create DataFrame - one row per workout
    final_df = pd.DataFrame(apple_format_rows)

    print(f"Final dataset shape: {final_df.shape}")
    print(f"Number of unique users: {final_df['user_id'].nunique()}")
    print(f"Number of unique workouts: {final_df['workout_id'].nunique()}")
    print(f"Average time points per workout: {final_df['workout_length'].mean():.1f}")

    print(f"\nCreating per-user chronological train/test split (train_ratio={train_ratio})")
    final_df = create_per_user_chronological_split(final_df, train_ratio=train_ratio)

    # Save in feather format (efficient for large datasets with lists)
    final_df.reset_index(drop=True).to_feather(output_feather)
    print(f"Saved to {output_feather}")

    analyze_all_patients(final_df, "data_summary.txt")

    return final_df


def analyze_all_patients(df: pd.DataFrame, output_file: str = "data_summary.txt"):
    """
    Create comprehensive analysis for all patients and save to file.

    Args:
        df: DataFrame with converted workout data
        output_file: Path to save the analysis
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PATIENT ANALYSIS")
    print("=" * 80)

    analysis_lines = []
    analysis_lines.append("=" * 80)
    analysis_lines.append("COMPREHENSIVE PATIENT ANALYSIS")
    analysis_lines.append("=" * 80)
    analysis_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    analysis_lines.append("")

    # Overall dataset statistics
    total_workouts = len(df)
    total_users = df['user_id'].nunique()

    overall_stats = [
        f"OVERALL DATASET STATISTICS:",
        f"- Total running workouts: {total_workouts:,}",
        f"- Total users: {total_users}",
        f"- Average workouts per user: {total_workouts / total_users:.1f}",
        f"- Training workouts: {df['in_train'].sum():,} ({df['in_train'].mean() * 100:.1f}%)",
        f"- Test workouts: {(~df['in_train']).sum():,} ({(~df['in_train']).mean() * 100:.1f}%)",
        ""
    ]

    for line in overall_stats:
        print(line)
        analysis_lines.append(line)

    # Per-user detailed analysis
    print("PER-USER DETAILED ANALYSIS:")
    analysis_lines.append("PER-USER DETAILED ANALYSIS:")
    analysis_lines.append("-" * 50)

    for user_id in sorted(df['user_id'].unique()):
        user_data = df[df['user_id'] == user_id].copy()
        user_data['start_time'] = pd.to_datetime(user_data['start_time'])
        user_data = user_data.sort_values('start_time')

        # Basic info
        total_user_workouts = len(user_data)
        train_workouts = user_data['in_train'].sum()
        test_workouts = total_user_workouts - train_workouts

        # Duration statistics
        duration_stats = user_data['duration_minutes'].describe()
        duration_sum = user_data['duration_minutes'].sum()

        # Heart rate statistics (from lists)
        all_hr_values = []
        all_speeds = []
        total_time_points = 0

        for _, workout in user_data.iterrows():
            hr_list = workout['heart_rate']
            speed_list = workout['horizontal_speed_kph']

            if isinstance(hr_list, list) and len(hr_list) > 0:
                all_hr_values.extend(hr_list)
                total_time_points += len(hr_list)

            if isinstance(speed_list, list) and len(speed_list) > 0:
                # Convert m/s to km/h for display
                speed_kmh = [s * 3.6 for s in speed_list if s > 0]
                all_speeds.extend(speed_kmh)

        hr_stats = {
            'min': min(all_hr_values) if all_hr_values else 0,
            'max': max(all_hr_values) if all_hr_values else 0,
            'mean': sum(all_hr_values) / len(all_hr_values) if all_hr_values else 0
        }

        speed_stats = {
            'min': min(all_speeds) if all_speeds else 0,
            'max': max(all_speeds) if all_speeds else 0,
            'mean': sum(all_speeds) / len(all_speeds) if all_speeds else 0
        } if all_speeds else {'min': 0, 'max': 0, 'mean': 0}

        # Date range
        date_range = f"{user_data['start_time'].min().strftime('%Y-%m-%d')} to {user_data['start_time'].max().strftime('%Y-%m-%d')}"

        # User demographics
        age = user_data['age'].iloc[0]
        gender = user_data['gender'].iloc[0]
        max_hr_theoretical = user_data['max_hr_theoretical'].iloc[0]
        min_hr_observed = user_data['min_hr_observed'].iloc[0]

        # Create user summary
        user_summary = [
            f"\nUSER: {user_id}",
            f"  Demographics:",
            f"    - Age: {age} years",
            f"    - Gender: {gender}",
            f"    - Theoretical Max HR: {max_hr_theoretical} bpm",
            f"    - Observed Min HR: {min_hr_observed} bpm",
            f"  Workout Summary:",
            f"    - Total workouts: {total_user_workouts}",
            f"    - Training workouts: {train_workouts}",
            f"    - Test workouts: {test_workouts}",
            f"    - Date range: {date_range}",
            f"    - Total data points: {total_time_points:,}",
            f"  Duration Statistics (minutes):",
            f"    - Min: {duration_stats['min']:.1f}",
            f"    - Max: {duration_stats['max']:.1f}",
            f"    - Mean: {duration_stats['mean']:.1f}",
            f"    - Total running time: {duration_sum:.1f} min ({duration_sum/60:.1f} hours)",  # Fixed line
            f"  Heart Rate Statistics (bpm):",
            f"    - Min: {hr_stats['min']:.1f}",
            f"    - Max: {hr_stats['max']:.1f}",
            f"    - Mean: {hr_stats['mean']:.1f}",
            f"  Speed Statistics (km/h):",
            f"    - Min: {speed_stats['min']:.1f}",
            f"    - Max: {speed_stats['max']:.1f}",
            f"    - Mean: {speed_stats['mean']:.1f}",
        ]

        for line in user_summary:
            print(line)
            analysis_lines.append(line)

    # Add summary statistics across all users
    user_workout_counts = df.groupby('user_id').size()
    user_duration_totals = df.groupby('user_id')['duration_minutes'].sum()

    summary_stats = [
        "",
        "CROSS-USER SUMMARY STATISTICS:",
        f"  Workouts per user:",
        f"    - Min: {user_workout_counts.min()}",
        f"    - Max: {user_workout_counts.max()}",
        f"    - Mean: {user_workout_counts.mean():.1f}",
        f"    - Median: {user_workout_counts.median():.1f}",
        f"  Total running time per user (hours):",
        f"    - Min: {user_duration_totals.min() / 60:.1f}",
        f"    - Max: {user_duration_totals.max() / 60:.1f}",
        f"    - Mean: {user_duration_totals.mean() / 60:.1f}",
        f"    - Total across all users: {user_duration_totals.sum() / 60:.1f} hours",
        "",
        "DATA QUALITY METRICS:",
        f"  - Average time points per workout: {df['workout_length'].mean():.1f}",
        f"  - Total data points across all workouts: {df['workout_length'].sum():,}",
        f"  - Users with >20 workouts: {(user_workout_counts > 20).sum()}",
        f"  - Users with >50 workouts: {(user_workout_counts > 50).sum()}",
    ]

    for line in summary_stats:
        print(line)
        analysis_lines.append(line)

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in analysis_lines:
            f.write(line + '\n')

    print(f"\n✓ Analysis saved to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    # Usage example
    input_file = "gods_trajectories.csv"  # Your current CSV
    output_file = "apple_format_data.feather"  # Output for Apple model

    print("=" * 60)
    print("FITBIT TO APPLE HEART RATE MODEL CONVERSION")
    print("=" * 60)
    print("Filtering criteria:")
    print("- Workout duration: 15-120 minutes")
    print("- Minimum workouts per user: 10")
    print("=" * 60)

    # Convert to Apple format with filtering
    df_apple = convert_fitbit_to_apple_format(input_file, output_file, apply_filtering=True)

    if df_apple is not None:
        print(f"\n✓ Comprehensive analysis completed")
        print(f"✓ Individual patient statistics saved to data_summary.txt")
        print("Done")