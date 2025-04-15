import pandas as pd
import neurokit2 as nk
import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

warnings.filterwarnings("ignore")

base_dir = os.path.join(const.BASE_DIR, "dataset3_MAUS/Data/Raw_data")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset3")
os.makedirs(output_folder, exist_ok=True)

participants = range(1, 26)  # P01 included now
results = []

for participant in participants:
    # Use 128 Hz for participant 1, 256 Hz for all others
    sampling_rate = 128 if participant == 1 else 256

    # Window settings in samples based on participant-specific sampling rate
    window_length_samples = const.INTERVAL * sampling_rate
    step_samples = const.STEP * sampling_rate

    participant_dir = os.path.join(base_dir, f"{participant:03}")
    resting_file = os.path.join(participant_dir, "inf_resting.csv")
    task_file = os.path.join(participant_dir, "inf_gsr.csv")

    if not os.path.exists(resting_file):
        print(f"Resting file not found: {resting_file}")
        continue
    if not os.path.exists(task_file):
        print(f"Task file not found: {task_file}")
        continue

    try:
        # Process baseline data
        resting_data = pd.read_csv(resting_file)
        if "Resting_GSR" not in resting_data.columns:
            print(f"'Resting_GSR' column not found for participant {participant}. Skipping.")
            continue

        baseline_signal = pd.to_numeric(resting_data["Resting_GSR"].dropna(), errors='coerce').values
        if len(baseline_signal) < 15:
            print(f"Baseline signal too short for participant {participant}. Skipping.")
            continue

        processed_baseline, _ = nk.eda_process(baseline_signal, sampling_rate=sampling_rate)
        analyzed_baseline = nk.eda_analyze(processed_baseline, sampling_rate=sampling_rate, method="interval-related")
        baseline_features = {f"Baseline_{key}": val for key, val in extract_scalar_features(analyzed_baseline).items()}

        # Process task data for each column in the CSV
        task_data = pd.read_csv(task_file)
        for idx, column in enumerate(task_data.columns, start=1):
            signal = pd.to_numeric(task_data[column].dropna(), errors='coerce').values

            if len(signal) < window_length_samples:
                print(f"Task signal too short in column {column} for participant {participant}. Skipping.")
                continue

            slice_count = 1

            # Sliding-window iteration
            for start_idx in range(0, len(signal) - window_length_samples + 1, step_samples):
                end_idx = start_idx + window_length_samples
                window_segment = signal[start_idx:end_idx]

                processed_task, _ = nk.eda_process(window_segment, sampling_rate=sampling_rate)
                analyzed_task = nk.eda_analyze(processed_task, sampling_rate=sampling_rate, method="interval-related")
                task_features = extract_scalar_features(analyzed_task)

                combined_features = {
                    "Participant": participant,
                    "Task": idx,
                    "Slice": slice_count
                }
                combined_features.update(task_features)
                combined_features.update(baseline_features)

                results.append(combined_features)
                slice_count += 1

    except Exception as e:
        print(f"Error processing data for participant {participant}: {e}")

if results:
    results_df = pd.DataFrame(results)
    columns_order = ["Participant", "Task", "Slice"] + [col for col in results_df.columns if col not in ["Participant", "Task", "Slice"]]
    results_df = results_df[columns_order]
    output_file = os.path.join(output_folder, "gsr_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"GSR features saved to {output_file}")
else:
    print("No features extracted. Please check input data.")