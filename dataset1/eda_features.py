import pandas as pd
import neurokit2 as nk
import os
import warnings

warnings.filterwarnings("ignore")

base_dir = "dataset1_SenseCobot/EDA_Empatica_Signals"
output_folder = "agg_data/dataset1"
os.makedirs(output_folder, exist_ok=True)

SAMPLING_RATE = 4

participants = range(1, 22)
tasks = range(1, 6)

def extract_scalar_features(features):
    return {key: (val.iloc[0] if isinstance(val, pd.Series) else val) for key, val in features.items()}

# Sliding-Window-Parameter
window_length = pd.Timedelta(seconds=72)  
step = pd.Timedelta(seconds=20)           
results = []

for participant in participants:
    participant_id = f"{participant:02}"  
    baseline_file = os.path.join(base_dir, f"EDA_Empatica_Baseline_P_{participant_id}.csv")
    
    # Baseline-Features 
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

            processed_baseline, _ = nk.eda_process(baseline_eda, sampling_rate=SAMPLING_RATE)
            analyzed_baseline = nk.eda_analyze(processed_baseline, sampling_rate=SAMPLING_RATE, method="interval-related")
            baseline_features = extract_scalar_features(analyzed_baseline)

            baseline_features = {f"Baseline_{key}": val for key, val in baseline_features.items()}
        except Exception as e:
            print(f"Error processing baseline for participant {participant_id}: {e}")
    else:
        print(f"Baseline file not found for participant {participant_id}: {baseline_file}")
    
    for task in tasks:
        task_file = os.path.join(base_dir, f"EDA_Empatica_Task {task}_P_{participant_id}.csv")
        if os.path.exists(task_file):
            try:
                task_data = pd.read_csv(task_file)
                # Prüfung auf erforderliche Spalten "EDA" und "Timestamp"
                if "EDA" not in task_data.columns or "Timestamp" not in task_data.columns:
                    missing = []
                    if "EDA" not in task_data.columns:
                        missing.append("EDA")
                    if "Timestamp" not in task_data.columns:
                        missing.append("Timestamp")
                    print(f"Missing columns {missing} in {task_file}")
                    continue

                task_data["Timestamp"] = pd.to_datetime(task_data["Timestamp"])
                start_time = task_data["Timestamp"].min()
                end_time = task_data["Timestamp"].max()

                current_start = start_time
                slice_count = 1  

                while current_start + window_length <= end_time:
                    current_end = current_start + window_length
                    window_data = task_data[(task_data["Timestamp"] >= current_start) &
                                            (task_data["Timestamp"] < current_end)]
                    
                    if len(window_data) < 10:
                        print(f"Insufficient data for participant {participant_id}, Task {task} in window starting at {current_start}")
                        current_start += step
                        continue

                    window_eda = window_data["EDA"].dropna().values
                    processed_window, _ = nk.eda_process(window_eda, sampling_rate=SAMPLING_RATE)
                    analyzed_window = nk.eda_analyze(processed_window, sampling_rate=SAMPLING_RATE, method="interval-related")
                    window_features = extract_scalar_features(analyzed_window)

                    # Zusammenführen der Features inklusive Participant, Task, Slice und Baseline-Features
                    combined_features = {
                        "Participant": participant,
                        "Task": task,
                        "Slice": slice_count
                    }
                    combined_features.update(window_features)
                    combined_features.update(baseline_features)
                    results.append(combined_features)

                    slice_count += 1
                    current_start += step  
                    
            except Exception as e:
                print(f"Error processing task file for participant {participant_id}, Task {task}: {e}")
        else:
            print(f"Task file not found for participant {participant_id}, Task {task}: {task_file}")

if results:
    results_df = pd.DataFrame(results)
    columns_order = ["Participant", "Task", "Slice"] + [col for col in results_df.columns if col not in ["Participant", "Task", "Slice"]]
    results_df = results_df[columns_order]
    output_file = os.path.join(output_folder, "eda_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Features successfully extracted and saved: {output_file}")
else:
    print("No features extracted. Please check input data.")
