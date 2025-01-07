import os
import pandas as pd
import numpy as np
from scipy.signal import welch

# Directories
base_dir = "dataset1_SenseCobot/EDA_Empatica_Signals"
output_folder = "agg_data/dataset1"
os.makedirs(output_folder, exist_ok=True)

TEMP_SAMPLING_RATE = 4  

participants = range(1, 22) 
tasks = range(1, 6) 

results = []

os.makedirs(output_folder, exist_ok=True)

def calculate_temp_features(temp_data):
    """
    Calculates features for the TEMP column:
    - Mean_TEMP
    - Gradient_TEMP
    - PSD_Power_TEMP
    """
    temp = temp_data.dropna().values
    if len(temp) < 10:
        raise ValueError("TEMP signal too short for feature extraction.")
    
    # Calculate features
    mean_temp = np.mean(temp)
    gradient_temp = np.sum(np.gradient(temp))
    f, psd = welch(temp, fs=TEMP_SAMPLING_RATE, nperseg=50)
    psd_power_temp = sum(psd)

    # Create feature dictionary
    temp_features = {
        "Mean_TEMP": mean_temp,
        "Gradient_TEMP": gradient_temp,
        "PSD_Power_TEMP": psd_power_temp
    }
    return temp_features

# Processing loop
for participant in participants:
    participant_id = f"{participant:02}"  # Zero-padded Participant ID
    baseline_file = f"{base_dir}/EDA_Empatica_Baseline_P_{participant_id}.csv"

    # Process baseline TEMP data
    baseline_features = {}
    if os.path.exists(baseline_file):
        try:
            baseline_data = pd.read_csv(baseline_file)
            if "TEMP" not in baseline_data.columns:
                print(f"Column 'TEMP' missing in {baseline_file}. Skipping baseline.")
                continue

            # Calculate baseline TEMP features
            baseline_temp = baseline_data["TEMP"]
            baseline_features = calculate_temp_features(baseline_temp)
            baseline_features = {f"Baseline_{key}": val for key, val in baseline_features.items()}

        except Exception as e:
            print(f"Error processing baseline data for participant {participant_id}: {e}")
    else:
        print(f"Baseline file not found for participant {participant_id}: {baseline_file}")

    # Process task TEMP data
    for task in tasks:
        task_file = f"{base_dir}/EDA_Empatica_Task {task}_P_{participant_id}.csv"

        if os.path.exists(task_file):
            try:
                task_data = pd.read_csv(task_file)
                if "TEMP" not in task_data.columns:
                    print(f"Column 'TEMP' missing in {task_file}. Skipping task.")
                    continue

                task_temp = task_data["TEMP"]
                task_features = calculate_temp_features(task_temp)

                # Combine task and baseline features
                combined_features = {
                    "Participant": participant,
                    "Task": task
                }
                combined_features.update(task_features)  # Add task TEMP features
                combined_features.update(baseline_features)  # Add baseline TEMP features

                results.append(combined_features)

            except Exception as e:
                print(f"Error processing task data for participant {participant_id}, Task {task}: {e}")
        else:
            print(f"Task file not found for participant {participant_id}, Task {task}: {task_file}")

if results:
    results_df = pd.DataFrame(results)

    columns_order = ["Participant", "Task"] + [col for col in results_df.columns if col not in ["Participant", "Task"]]
    results_df = results_df[columns_order]

    output_file = os.path.join(output_folder, "temperature_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Temperature features successfully extracted and saved: {output_file}")
else:
    print("No TEMP data processed. Please check input data.")
