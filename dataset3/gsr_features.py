import pandas as pd
import neurokit2 as nk
import os
import warnings

warnings.filterwarnings("ignore")


base_dir = os.path.join("dataset3_MAUS", "Data", "Raw_data")
output_folder = os.path.join("agg_data", "dataset3")
os.makedirs(output_folder, exist_ok=True)


# Helper function to extract scalar features
def extract_scalar_features(features):
    return {key: (val.iloc[0] if isinstance(val, pd.Series) else val) for key, val in features.items()}


SAMPLING_RATE = 256  
participants = range(2, 26)  
results = []

# Window settings in samples (72 sec window, 20 sec step)
window_length_samples = 72 * SAMPLING_RATE  # 18432 samples
step_samples = 20 * SAMPLING_RATE           # 5120 samples

for participant in participants:
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

        processed_baseline, _ = nk.eda_process(baseline_signal, sampling_rate=SAMPLING_RATE)
        analyzed_baseline = nk.eda_analyze(processed_baseline, sampling_rate=SAMPLING_RATE, method="interval-related")
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

                processed_task, _ = nk.eda_process(window_segment, sampling_rate=SAMPLING_RATE)
                analyzed_task = nk.eda_analyze(processed_task, sampling_rate=SAMPLING_RATE, method="interval-related")
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
