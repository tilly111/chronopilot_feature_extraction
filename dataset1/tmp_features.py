import os
import pandas as pd
import numpy as np
from scipy.signal import welch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const

# Directories
base_dir = os.path.join(const.BASE_DIR, "dataset1_SenseCobot/EDA_Empatica_Signals")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset1")
os.makedirs(output_folder, exist_ok=True)

TEMP_SAMPLING_RATE = 1 
window_length = int(const.INTERVAL * TEMP_SAMPLING_RATE)
step = int(const.STEP * TEMP_SAMPLING_RATE)

participants = range(1, 22) 
tasks = range(1, 6) 

results = []

def calculate_temp_features(temp_segment):
    temp = temp_segment.dropna().values
    if len(temp) < 10:
        raise ValueError("TEMP signal too short for feature extraction.")

    mean_temp = np.mean(temp)
    gradient_temp = np.sum(np.gradient(temp))
    f, psd = welch(temp, fs=TEMP_SAMPLING_RATE, nperseg=50)
    psd_power_temp = sum(psd)

    temp_features = {
        "Mean_TEMP": mean_temp,
        "Gradient_TEMP": gradient_temp,
        "PSD_Power_TEMP": psd_power_temp
    }
    return temp_features

# Processing loop
for participant in participants:
    participant_id = f"{participant:02}"
    baseline_file = f"{base_dir}/EDA_Empatica_Baseline_P_{participant_id}.csv"

    baseline_features = {}
    if os.path.exists(baseline_file):
        try:
            baseline_data = pd.read_csv(baseline_file)
            if "TEMP" not in baseline_data.columns:
                print(f"Column 'TEMP' missing in {baseline_file}. Skipping baseline.")
                continue
            baseline_temp = baseline_data["TEMP"]
            baseline_features = calculate_temp_features(baseline_temp)
            baseline_features = {f"Baseline_{key}": val for key, val in baseline_features.items()}
        except Exception as e:
            print(f"Error processing baseline data for participant {participant_id}: {e}")
    else:
        print(f"Baseline file not found for participant {participant_id}: {baseline_file}")

    for task in tasks:
        task_file = f"{base_dir}/EDA_Empatica_Task {task}_P_{participant_id}.csv"
        if os.path.exists(task_file):
            try:
                task_data = pd.read_csv(task_file)
                if "TEMP" not in task_data.columns:
                    print(f"Column 'TEMP' missing in {task_file}. Skipping task.")
                    continue

                temp_signal = task_data["TEMP"].dropna().values
                total_samples = len(temp_signal)

                start_idx = 0
                slice_count = 1

                while start_idx + window_length <= total_samples:
                    temp_segment = pd.Series(temp_signal[start_idx:start_idx + window_length])
                    task_features = calculate_temp_features(temp_segment)

                    combined_features = {
                        "Participant": participant,
                        "Task": task,
                        "Slice": slice_count
                    }
                    combined_features.update(task_features)
                    combined_features.update(baseline_features)

                    results.append(combined_features)

                    start_idx += step
                    slice_count += 1

            except Exception as e:
                print(f"Error processing task data for participant {participant_id}, Task {task}: {e}")
        else:
            print(f"Task file not found for participant {participant_id}, Task {task}: {task_file}")

if results:
    results_df = pd.DataFrame(results)
    columns_order = ["Participant", "Task", "Slice"] + [col for col in results_df.columns if col not in ["Participant", "Task", "Slice"]]
    results_df = results_df[columns_order]

    output_file = os.path.join(output_folder, "temperature_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Temperature features successfully extracted and saved: {output_file}")
else:
    print("No TEMP data processed. Please check input data.")
