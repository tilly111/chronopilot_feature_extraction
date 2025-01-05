import pandas as pd
import neurokit2 as nk
import os
import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Directories
base_dir = "dataset1_SenseCobot/ECG_Shimmer3_Signals"
output_folder = "agg_data/dataset1"
os.makedirs(output_folder, exist_ok=True)

# Sampling rate and participants/tasks
SAMPLING_RATE = 512
participants = range(1, 26)
tasks = range(1, 6)

# ECG columns to process
ecg_columns = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]

# Helper to extract scalar features
def extract_scalar_features(features):
    """Flatten and extract scalar values."""
    return {key: (val.iloc[0] if isinstance(val, pd.Series) else val) for key, val in features.items()}

# Process ECG data
def process_ecg():
    results = {col: [] for col in ecg_columns}

    for participant in participants:
        participant_id = f"{participant:02}"
        baseline_file = f"{base_dir}/ECG_Baseline_P_{participant_id}.csv"
        baseline_features = {col: {} for col in ecg_columns}

        # Process baseline data
        if os.path.exists(baseline_file):
            try:
                print(f"Processing baseline for Participant {participant_id}")
                baseline_data = pd.read_csv(baseline_file)
                for col in ecg_columns:
                    if col in baseline_data.columns:
                        ecg_signals, _ = nk.ecg_process(baseline_data[col].dropna().values, sampling_rate=SAMPLING_RATE)
                        analyzed_baseline = nk.ecg_analyze(ecg_signals, sampling_rate=SAMPLING_RATE, method="interval-related")
                        # Add Participant and Baseline features directly in correct order
                        baseline_features[col] = {f"Baseline_{key}": val for key, val in extract_scalar_features(analyzed_baseline).items()}
                    else:
                        print(f"Column '{col}' missing in {baseline_file}")
            except Exception as e:
                print(f"Error processing baseline for Participant {participant_id}: {e}")
        else:
            print(f"Baseline file missing for Participant {participant_id}")

        # Process task data
        for task in tasks:
            task_file = f"{base_dir}/ECG_Task {task}_P_{participant_id}.csv"

            if os.path.exists(task_file):
                try:
                    print(f"Processing Task {task} for Participant {participant_id}")
                    task_data = pd.read_csv(task_file)
                    for col in ecg_columns:
                        if col in task_data.columns:
                            ecg_signals, _ = nk.ecg_process(task_data[col].dropna().values, sampling_rate=SAMPLING_RATE)
                            analyzed_task = nk.ecg_analyze(ecg_signals, sampling_rate=SAMPLING_RATE, method="interval-related")

                            # Combine task and baseline features directly in the desired order
                            combined_features = {"Participant": participant, "Task": task}
                            combined_features.update(extract_scalar_features(analyzed_task))
                            combined_features.update(baseline_features[col])
                            results[col].append(combined_features)
                        else:
                            print(f"Column '{col}' missing in {task_file}")
                except Exception as e:
                    print(f"Error processing Task {task} for Participant {participant_id}: {e}")
            else:
                print(f"Task file missing for Participant {participant_id}, Task {task}")

    # Save results for each ECG column
    for col, data in results.items():
        if data:
            results_df = pd.DataFrame(data)
            output_file = os.path.join(output_folder, f"ecg_features_{col.replace(' ', '_')}.csv")
            results_df.to_csv(output_file, index=False)
            print(f"ECG features for {col} saved to: {output_file}")

# Main function
def main():
    process_ecg()

if __name__ == "__main__":
    main()
