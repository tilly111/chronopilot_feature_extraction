import pandas as pd
import neurokit2 as nk
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

# Base directories
base_dir = os.path.join(const.BASE_DIR, "dataset3_MAUS/Data/Raw_data")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset3")
os.makedirs(output_folder, exist_ok=True)

# Constants
SAMPLING_RATE = 256  # Hz
participants = range(1, 26)  # Participant IDs from 1 to 25
results = []


# Process ECG data for each participant
for participant in participants:
    participant_dir = f"{base_dir}/{participant:03}"  # Zero-padded directory
    resting_file = f"{participant_dir}/inf_resting.csv"
    task_file = f"{participant_dir}/inf_ecg.csv"

    # Check if files exist
    if not os.path.exists(resting_file):
        print(f"Resting file not found: {resting_file}")
        continue
    if not os.path.exists(task_file):
        print(f"Task file not found: {task_file}")
        continue

    try:
        # Process baseline data
        resting_data = pd.read_csv(resting_file)
        if "Resting_ECG" not in resting_data.columns:
            print(f"Column 'Resting_ECG' not found in {resting_file}")
            continue

        baseline_signal = resting_data["Resting_ECG"].dropna().values
        if len(baseline_signal) < 15:
            print(f"Baseline signal too short for participant {participant}. Skipping.")
            continue

        processed_baseline, _ = nk.ecg_process(ecg_signal=baseline_signal, sampling_rate=SAMPLING_RATE)
        analyzed_baseline = nk.ecg_analyze(processed_baseline, sampling_rate=SAMPLING_RATE, method="interval-related")
        baseline_features = {f"Baseline_{key}": val for key, val in extract_scalar_features(analyzed_baseline).items()}

        # Process task data
        task_data = pd.read_csv(task_file)
        for idx, column in enumerate(task_data.columns, start=1):
            signal = task_data[column].dropna().values

            if len(signal) < 15:
                print(f"Task signal too short in column {column} for participant {participant}. Skipping.")
                continue

            processed_task, _ = nk.ecg_process(ecg_signal=signal, sampling_rate=SAMPLING_RATE)
            analyzed_task = nk.ecg_analyze(processed_task, sampling_rate=SAMPLING_RATE, method="interval-related")
            task_features = extract_scalar_features(analyzed_task)

            # Combine participant, task, and features
            combined_features = {
                "Participant": participant,
                "Task": idx  # Numeric task label
            }
            combined_features.update(task_features)
            combined_features.update(baseline_features)

            results.append(combined_features)

    except Exception as e:
        print(f"Error processing data for participant {participant}: {e}")

if results:
    results_df = pd.DataFrame(results)

    columns_order = ["Participant", "Task"] + [col for col in results_df.columns if col not in ["Participant", "Task"]]
    results_df = results_df[columns_order]

    output_file = os.path.join(output_folder, "ecg_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"ECG features saved to {output_file}")
else:
    print("No features extracted. Please check input data.")
