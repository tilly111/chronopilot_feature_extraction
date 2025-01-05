import pandas as pd
import neurokit2 as nk
import os

# Base directories
BASE_DIR = "dataset3_MAUS/Data/Raw_data"
OUTPUT_FOLDER = "agg_data/dataset3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Sampling rates
SAMPLING_RATE = {
    "ppg": 256,     #
    "pixart": 100,  
}

PARTICIPANTS = range(1, 26)

# Helper function to extract scalar features
def extract_scalar_features(features):
    """Convert nested objects to scalar values."""
    return {key: (val.iloc[0] if isinstance(val, pd.Series) else val) for key, val in features.items()}

# Function to process a signal (PPG or Pixart)
def process_signal(participant, participant_dir, signal_type, resting_file, resting_column, task_file_name):
    results = []

    resting_file_path = os.path.join(participant_dir, resting_file)
    task_file_path = os.path.join(participant_dir, task_file_name)

    if not os.path.exists(resting_file_path):
        print(f"Resting file not found for {signal_type}, Participant {participant}: {resting_file_path}")
        return results
    if not os.path.exists(task_file_path):
        print(f"Task file not found for {signal_type}, Participant {participant}: {task_file_path}")
        return results

    try:
        # Process baseline data
        resting_data = pd.read_csv(resting_file_path)
        if resting_column not in resting_data.columns:
            print(f"Column '{resting_column}' not found in {resting_file_path}")
            return results

        baseline_signal = resting_data[resting_column].dropna().values
        if len(baseline_signal) < 15:
            print(f"Baseline signal too short for Participant {participant}, {signal_type}. Skipping.")
            return results

        processed_baseline, _ = nk.ppg_process(baseline_signal, sampling_rate=SAMPLING_RATE[signal_type])
        analyzed_baseline = nk.ppg_analyze(processed_baseline, sampling_rate=SAMPLING_RATE[signal_type], method="interval-related")
        baseline_features = {f"Baseline_{key}": val for key, val in extract_scalar_features(analyzed_baseline).items()}

        # Process task data
        task_data = pd.read_csv(task_file_path)
        for idx, column in enumerate(task_data.columns, start=1):
            signal = task_data[column].dropna().values

            if len(signal) < 15:
                print(f"Task signal too short in column {column} for Participant {participant}, {signal_type}. Skipping.")
                continue

            processed_task, _ = nk.ppg_process(signal, sampling_rate=SAMPLING_RATE[signal_type])
            analyzed_task = nk.ppg_analyze(processed_task, sampling_rate=SAMPLING_RATE[signal_type], method="interval-related")
            task_features = extract_scalar_features(analyzed_task)

            combined_features = {
                "Participant": participant,
                "Task": idx  # Numeric task label
            }
            combined_features.update(task_features)
            combined_features.update(baseline_features)

            results.append(combined_features)

    except Exception as e:
        print(f"Error processing {signal_type} data for Participant {participant}: {e}")
    return results

# Process Pixart data
pixart_results = []
for participant in PARTICIPANTS:
    participant_dir = f"{BASE_DIR}/{participant:03}"
    pixart_results.extend(process_signal(
        participant=participant,
        participant_dir=participant_dir,
        signal_type="pixart",
        resting_file="pixart_resting.csv",
        resting_column="Resting",
        task_file_name="pixart.csv"
    ))

# Save Pixart results
pixart_file_path = os.path.join(OUTPUT_FOLDER, "ppg_features_pixart.csv")
if pixart_results:
    pd.DataFrame(pixart_results).to_csv(pixart_file_path, index=False)
    print(f"Pixart PPG features saved to {pixart_file_path}")

# Process PPG data
ppg_results = []
for participant in PARTICIPANTS:
    participant_dir = f"{BASE_DIR}/{participant:03}"
    ppg_results.extend(process_signal(
        participant=participant,
        participant_dir=participant_dir,
        signal_type="ppg",
        resting_file="inf_resting.csv",
        resting_column="Resting_PPG",
        task_file_name="inf_ppg.csv"
    ))

# Save PPG results
ppg_file_path = os.path.join(OUTPUT_FOLDER, "ppg_features.csv")
if ppg_results:
    pd.DataFrame(ppg_results).to_csv(ppg_file_path, index=False)
    print(f"PPG features saved to {ppg_file_path}")
