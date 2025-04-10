import pandas as pd
import neurokit2 as nk
import os
import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Directories
base_dir = "dataset1_SenseCobot/GSR_Shimmer3_Signals"
output_folder = "agg_data/dataset1"
os.makedirs(output_folder, exist_ok=True)

# Sampling rate and participants/tasks
SAMPLING_RATE = 128
participants = range(1, 26)
tasks = range(1, 6)

# Helper to extract scalar features
def extract_scalar_features(features):
    return {key: (val.iloc[0] if isinstance(val, pd.Series) else val) for key, val in features.items()}

window_length = pd.Timedelta(seconds=72)  
step = pd.Timedelta(seconds=20)          

results = []

for participant in participants:
    participant_id = f"{participant:02}"
    baseline_file = f"{base_dir}/GSR_Baseline_P_{participant_id}.csv"

    baseline_features = {}
    if os.path.exists(baseline_file):
        try:
            print(f"Processing baseline for Participant {participant_id}")
            baseline_data = pd.read_csv(baseline_file)
            if "GSR Conductance CAL" in baseline_data.columns:
                baseline_conductance = baseline_data["GSR Conductance CAL"].dropna().values
                if len(baseline_conductance) >= 15:
                    processed_baseline, _ = nk.eda_process(baseline_conductance, sampling_rate=SAMPLING_RATE)
                    analyzed_baseline = nk.eda_analyze(processed_baseline, sampling_rate=SAMPLING_RATE, method="interval-related")
                    baseline_features = {f"Baseline_{key}": val for key, val in extract_scalar_features(analyzed_baseline).items()}
            else:
                print(f"Column 'GSR Conductance CAL' missing in {baseline_file}")
        except Exception as e:
            print(f"Error processing baseline for Participant {participant_id}: {e}")
    else:
        print(f"Baseline file missing for Participant {participant_id}")

    for task in tasks:
        task_file = f"{base_dir}/GSR_Task {task}_P_{participant_id}.csv"

        if os.path.exists(task_file):
            try:
                print(f"Processing Task {task} for Participant {participant_id}")
                task_data = pd.read_csv(task_file)
                if "GSR Conductance CAL" in task_data.columns and "Timestamp" in task_data.columns:

                    task_data["Timestamp"] = pd.to_datetime(task_data["Timestamp"])
                    start_time = task_data["Timestamp"].min()
                    end_time = task_data["Timestamp"].max()
                    
                    current_start = start_time
                    interval_count = 1  

                    while current_start + window_length <= end_time:
                        current_end = current_start + window_length
                        window_data = task_data[(task_data["Timestamp"] >= current_start) & (task_data["Timestamp"] < current_end)]
                        
                        if len(window_data) >= 15:
                            window_conductance = window_data["GSR Conductance CAL"].dropna().values
                            processed_window, _ = nk.eda_process(window_conductance, sampling_rate=SAMPLING_RATE)
                            analyzed_window = nk.eda_analyze(processed_window, sampling_rate=SAMPLING_RATE, method="interval-related")
                            window_features = extract_scalar_features(analyzed_window)
                            
                            combined_features = {
                                "Participant": participant,
                                "Task": task,
                                "Slice": interval_count  
                            }
                            combined_features.update(window_features)
                            combined_features.update(baseline_features)
                            results.append(combined_features)
                            
                            interval_count += 1
                        else:
                            print(f"Insufficient data points for Participant {participant_id}, Task {task} in window starting at {current_start}")
                        
                        current_start += step  
                else:
                    missing = []
                    if "GSR Conductance CAL" not in task_data.columns:
                        missing.append("'GSR Conductance CAL'")
                    if "Timestamp" not in task_data.columns:
                        missing.append("'Timestamp'")
                    print(f"Column(s) {', '.join(missing)} missing in {task_file}")
            except Exception as e:
                print(f"Error processing Task {task} for Participant {participant_id}: {e}")
        else:
            print(f"Task file missing for Participant {participant_id}, Task {task}")

if results:
    results_df = pd.DataFrame(results)
    columns_order = ["Participant", "Task", "Slice"] + [col for col in results_df.columns if col not in ["Participant", "Task", "Slice"]]
    results_df = results_df[columns_order]
    output_file = os.path.join(output_folder, "gsr_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
else:
    print("No features extracted. Check input data.")
