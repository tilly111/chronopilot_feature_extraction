import os
import pandas as pd
import neurokit2 as nk

# Base directories
input_dir = "dataset8_POPANE"
output_file = "agg_data/dataset8/eda_features.csv"
baseline_dir = os.path.join(input_dir, "Baselines")
SAMPLING_RATE = 1000

def extract_scalar_features(features):
    """Convert nested objects to scalar values."""
    return {key: (val.iloc[0] if isinstance(val, pd.Series) else val) for key, val in features.items()}

data_list = []

for stimuli_folder in os.listdir(input_dir):
    # Skip folders containing "Neutral" or named "Baselines"
    if "Neutral" in stimuli_folder or stimuli_folder == "Baselines":
        continue

    stimuli_path = os.path.join(input_dir, stimuli_folder)
    if not os.path.isdir(stimuli_path):
        continue

    for file_name in os.listdir(stimuli_path):
        if "Neutral" in file_name or not file_name.endswith(".csv"):
            continue

        parts = file_name.split("_")
        if len(parts) < 3:
            continue

        task = parts[0][1:]
        participant = parts[1][1:]

        # Process baseline file to calculate baseline features only
        baseline_file_name = f"S{task}_P{participant}_Baseline.csv"
        baseline_file_path = os.path.join(baseline_dir, baseline_file_name)
        baseline_features = {}

        if os.path.exists(baseline_file_path):
            try:
                # Open the file and find the start of the actual data
                with open(baseline_file_path, "r") as f:
                    lines = f.readlines()

                # Find the line where the actual data starts (e.g., 'timestamp,affect,...')
                data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))

                # Read the CSV starting from the correct line
                baseline_data = pd.read_csv(baseline_file_path, skiprows=data_start_idx)

                if "EDA" not in baseline_data.columns:
                    print(f"No 'EDA' column in baseline file {baseline_file_path}. Skipping baseline.")
                else:
                    baseline_signal = baseline_data["EDA"].dropna().values
                    if len(baseline_signal) >= 15:
                        processed_baseline, _ = nk.eda_process(baseline_signal, sampling_rate=SAMPLING_RATE)
                        analyzed_baseline = nk.eda_analyze(
                            processed_baseline, sampling_rate=SAMPLING_RATE, method="interval-related"
                        )
                        baseline_features = {
                            f"Baseline_{key}": val for key, val in extract_scalar_features(analyzed_baseline).items()
                        }
                    else:
                        print(f"Baseline signal too short in {baseline_file_path}. Skipping baseline.")
            except Exception as e:
                print(f"Error processing baseline file {baseline_file_path}: {e}")
        else:
            print(f"Baseline file not found: {baseline_file_name}")

        # Skip processing this file if baseline features are missing
        if not baseline_features:
            print(f"Baseline features missing for participant {participant} in task {task}. Skipping...")
            continue


        emotion_file_path = os.path.join(stimuli_path, file_name)
        try:
            with open(emotion_file_path, "r") as f:
                lines = f.readlines()

            data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))

            emotion_data = pd.read_csv(emotion_file_path, skiprows=data_start_idx)

            if "EDA" not in emotion_data.columns:
                print(f"No 'EDA' column in {emotion_file_path}. Skipping.")
                continue

            emotion_signal = emotion_data["EDA"].dropna().values
            if len(emotion_signal) < 15:
                print(f"EDA signal too short in {emotion_file_path}. Skipping.")
                continue

            # Extract the first marker value if present
            emotion_data.columns = emotion_data.columns.str.strip()  # Remove any extra spaces from column names
            first_marker = (
                emotion_data["marker"].dropna().iloc[0]
                if "marker" in emotion_data.columns and not emotion_data["marker"].dropna().empty
                else None
            )

            processed_emotion, _ = nk.eda_process(emotion_signal, sampling_rate=SAMPLING_RATE)
            analyzed_emotion = nk.eda_analyze(processed_emotion, sampling_rate=SAMPLING_RATE, method="interval-related")
            emotion_features = extract_scalar_features(analyzed_emotion)

        except Exception as e:
            print(f"Error processing emotion file {emotion_file_path}: {e}")
            continue

        combined_features = {
            "Participant": int(participant),
            "Task": int(task),  
            "Marker": first_marker,  
        }
        combined_features.update(emotion_features)
        combined_features.update(baseline_features)

        data_list.append(combined_features)

if data_list:
    results_df = pd.DataFrame(data_list)

    output_df = results_df.sort_values(by=["Participant", "Task"]).reset_index(drop=True)
    output_df = results_df[
        ["Participant", "Task", "Marker"]
        + [col for col in results_df.columns if col not in ["Participant", "Task", "Marker"]]
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    print(f"EDA results saved to '{output_file}'!")
else:
    print("No data processed.")
