import warnings
import pandas as pd
import neurokit2 as nk
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

# Suppress specific NeuroKitWarning
warnings.filterwarnings("ignore", category=nk.NeuroKitWarning)

# Directories
base_dir = os.path.join(const.BASE_DIR, "dataset1_SenseCobot/EDA_Empatica_Signals")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset1")
os.makedirs(output_folder, exist_ok=True)

# Sampling rate
SAMPLING_RATE = 4  

# Participants and tasks
participants = range(1, 22) 
tasks = range(1, 6) 

results = []

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


# Processing loop
for participant in participants:
    participant_id = f"{participant:02}"  # Zero-padded Participant ID
    baseline_file = f"{base_dir}/EDA_Empatica_Baseline_P_{participant_id}.csv"

    # Process baseline data
    baseline_features = {}
    if os.path.exists(baseline_file):
        try:
            baseline_data = pd.read_csv(baseline_file)
            if "EDA" not in baseline_data.columns:
                print(f"Column 'EDA' missing in {baseline_file}")
                continue
            
            baseline_eda = baseline_data["EDA"].dropna().values
            if len(baseline_eda) < 10:
                print(f"Baseline signal too short for participant {participant_id}")
                continue

            # EDA processing and analysis
            processed_baseline, _ = nk.eda_process(baseline_eda, sampling_rate=SAMPLING_RATE)
            analyzed_baseline = nk.eda_analyze(processed_baseline, sampling_rate=SAMPLING_RATE, method="interval-related")
            baseline_features = extract_scalar_features(analyzed_baseline)
            baseline_features = {f"Baseline_{key}": val for key, val in baseline_features.items()}
        except Exception as e:
            print(f"Error processing baseline data for participant {participant_id}: {e}")
    else:
        print(f"Baseline file not found for participant {participant_id}: {baseline_file}")

    # Process task data
    for task in tasks:
        task_file = f"{base_dir}/EDA_Empatica_Task {task}_P_{participant_id}.csv"

        if os.path.exists(task_file):
            try:
                task_data = pd.read_csv(task_file)
                if "EDA" not in task_data.columns:
                    print(f"Column 'EDA' missing in {task_file}.")
                    continue

                task_eda = task_data["EDA"].dropna().values
                if len(task_eda) < 10:
                    continue

                # EDA processing and analysis
                processed_task, _ = nk.eda_process(task_eda, sampling_rate=SAMPLING_RATE)
                analyzed_task = nk.eda_analyze(processed_task, sampling_rate=SAMPLING_RATE, method="interval-related")
                task_features = extract_scalar_features(analyzed_task)

                # Combine task and baseline features
                combined_features = {
                    "Participant": participant,
                    "Task": task
                }
                combined_features.update(task_features)  # Add task features
                combined_features.update(baseline_features)  # Add baseline features

                results.append(combined_features)

            except Exception as e:
                print(f"Error processing task data for participant {participant_id}, Task {task}: {e}")
        else:
            print(f"Task file not found for participant {participant_id}, Task {task}: {task_file}")

# Save results
if results:
    results_df = pd.DataFrame(results)

    # Reorder columns: Participant and Task first
    columns_order = ["Participant", "Task"] + [col for col in results_df.columns if col not in ["Participant", "Task"]]
    results_df = results_df[columns_order]

    # Save to file
    output_file = os.path.join(output_folder, "eda_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Features successfully extracted and saved: {output_file}")
else:
    print("No features extracted. Please check input data.")
