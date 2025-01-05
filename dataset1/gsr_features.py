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

# Process data
results = []
for participant in participants:
    participant_id = f"{participant:02}"
    baseline_file = f"{base_dir}/GSR_Baseline_P_{participant_id}.csv"

    # Process baseline
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

    # Process tasks
    for task in tasks:
        task_file = f"{base_dir}/GSR_Task {task}_P_{participant_id}.csv"

        if os.path.exists(task_file):
            try:
                print(f"Processing Task {task} for Participant {participant_id}")
                task_data = pd.read_csv(task_file)
                if "GSR Conductance CAL" in task_data.columns:
                    task_conductance = task_data["GSR Conductance CAL"].dropna().values
                    if len(task_conductance) >= 15:
                        processed_task, _ = nk.eda_process(task_conductance, sampling_rate=SAMPLING_RATE)
                        analyzed_task = nk.eda_analyze(processed_task, sampling_rate=SAMPLING_RATE, method="interval-related")
                        task_features = extract_scalar_features(analyzed_task)

                        combined_features = {"Participant": participant, "Task": task}
                        combined_features.update(task_features)
                        combined_features.update(baseline_features)
                        results.append(combined_features)
                    else:
                        print(f"Task signal too short for Participant {participant_id}, Task {task}")
                else:
                    print(f"Column 'GSR Conductance CAL' missing in {task_file}")
            except Exception as e:
                print(f"Error processing Task {task} for Participant {participant_id}: {e}")
        else:
            print(f"Task file missing for Participant {participant_id}, Task {task}")

# Save results
if results:
    results_df = pd.DataFrame(results)
    columns_order = ["Participant", "Task"] + [col for col in results_df.columns if col not in ["Participant", "Task"]]
    results_df = results_df[columns_order]
    output_file = os.path.join(output_folder, "gsr_features.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
else:
    print("No features extracted. Check input data.")
