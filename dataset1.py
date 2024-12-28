import os
import pandas as pd
import neurokit2 as nk
import warnings

warnings.filterwarnings("ignore", category=nk.NeuroKitWarning)

# Base directories
base_dir = {
    "labels": "dataset1_SenseCobot/Additional_Information",
    "gsr": "dataset1_SenseCobot/GSR_Shimmer3_Signals",
    "eda": "dataset1_SenseCobot/EDA_Empatica_Signals",
    "ecg": "dataset1_SenseCobot/ECG_Shimmer3_Signals"
}
output_folder = "agg_data/dataset1"
os.makedirs(output_folder, exist_ok=True)

# Sampling rates
SAMPLING_RATE = {
    "gsr": 128,
    "eda": 4,
    "ecg": 512
}

# Participants and tasks
participants = range(1, 22)
tasks = range(1, 6)

# Label processing
def process_labels():
    file_path = os.path.join(base_dir['labels'], 'NASA_TLX.csv')
    nasa_tlx_data = pd.read_csv(file_path)
    nasa_tlx_data.columns = nasa_tlx_data.columns.str.strip()
    nasa_tlx_data = nasa_tlx_data.dropna(how='all')

    data = []

    for participant_id, row in enumerate(nasa_tlx_data.iterrows(), start=1):
        row = row[1]
        for task in tasks:
            a_score = row[f'A_TASK {task}']
            b_score = row[f'B_TASK {task}']
            stress_label = 1 if a_score + b_score >= 7 else 0
            well_being_label = 0 if stress_label == 1 else 1

            data.append({
                'Participant': participant_id,
                'Task': task,
                'Stress': stress_label,
                'Well-Being': well_being_label,
                'A_Score': a_score,
                'B_Score': b_score
            })

    processed_data = pd.DataFrame(data)
    output_file = os.path.join(output_folder, 'labels.csv')
    processed_data.to_csv(output_file, index=False)
    print(f'Labels saved to: {output_file}')

# GSR processing
def process_gsr():
    results = []

    for participant in participants:
        participant_id = f"{participant:02}"
        baseline_file = os.path.join(base_dir['gsr'], f"GSR_Baseline_P_{participant_id}.csv")
        baseline_features = {}

        if os.path.exists(baseline_file):
            baseline_data = pd.read_csv(baseline_file, low_memory=False)
            if "GSR Conductance CAL" in baseline_data.columns:
                baseline_conductance = baseline_data["GSR Conductance CAL"].dropna().values
                if len(baseline_conductance) >= 15:
                    processed_baseline, _ = nk.eda_process(baseline_conductance, sampling_rate=SAMPLING_RATE['gsr'])
                    baseline_features = processed_baseline.mean(axis=0).to_dict()
                    baseline_features = {f"Baseline_{key}": val for key, val in baseline_features.items()}

        for task in tasks:
            task_file = os.path.join(base_dir['gsr'], f"GSR_Task {task}_P_{participant_id}.csv")

            if os.path.exists(task_file):
                task_data = pd.read_csv(task_file, low_memory=False)
                if "GSR Conductance CAL" in task_data.columns:
                    task_conductance = task_data["GSR Conductance CAL"].dropna().values
                    if len(task_conductance) >= 15:
                        processed_task, _ = nk.eda_process(task_conductance, sampling_rate=SAMPLING_RATE['gsr'])
                        task_features = processed_task.mean(axis=0).to_dict()
                        combined_features = {"Participant": participant, "Task": task}
                        combined_features.update(task_features)
                        combined_features.update(baseline_features)
                        results.append(combined_features)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_folder, 'gsr_features.csv'), index=False)
        print(f'GSR features saved to: {os.path.join(output_folder, "gsr_features.csv")}')

# EDA processing
def process_eda():
    results = []

    for participant in participants:
        participant_id = f"{participant:02}"
        baseline_file = os.path.join(base_dir['eda'], f"EDA_Empatica_Baseline_P_{participant_id}.csv")
        baseline_features = {}

        if os.path.exists(baseline_file):
            baseline_data = pd.read_csv(baseline_file, low_memory=False)
            if "EDA" in baseline_data.columns:
                baseline_eda = baseline_data["EDA"].dropna().values
                if len(baseline_eda) >= 10:
                    processed_baseline, _ = nk.eda_process(baseline_eda, sampling_rate=SAMPLING_RATE['eda'])
                    baseline_features = processed_baseline.mean(axis=0).to_dict()
                    baseline_features = {f"Baseline_{key}": val for key, val in baseline_features.items()}

        for task in tasks:
            task_file = os.path.join(base_dir['eda'], f"EDA_Empatica_Task {task}_P_{participant_id}.csv")

            if os.path.exists(task_file):
                task_data = pd.read_csv(task_file, low_memory=False)
                if "EDA" in task_data.columns:
                    task_eda = task_data["EDA"].dropna().values
                    if len(task_eda) >= 10:
                        processed_task, _ = nk.eda_process(task_eda, sampling_rate=SAMPLING_RATE['eda'])
                        task_features = processed_task.mean(axis=0).to_dict()
                        combined_features = {"Participant": participant, "Task": task}
                        combined_features.update(task_features)
                        combined_features.update(baseline_features)
                        results.append(combined_features)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_folder, 'eda_features.csv'), index=False)
        print(f'EDA features saved to: {os.path.join(output_folder, "eda_features.csv")}')

# ECG processing
def process_ecg():
    ecg_columns = ["ECG LL-RA CAL", "ECG LA-RA CAL", "ECG Vx-RL CAL"]
    results_per_column = {col: [] for col in ecg_columns}

    for participant in participants:
        participant_id = f"{participant:02}"
        baseline_file = os.path.join(base_dir['ecg'], f"ECG_Baseline_P_{participant_id}.csv")
        baseline_features = {col: {} for col in ecg_columns}

        if os.path.exists(baseline_file):
            baseline_data = pd.read_csv(baseline_file, low_memory=False)
            for col in ecg_columns:
                if col in baseline_data.columns:
                    ecg_cleaned = nk.ecg_clean(baseline_data[col].dropna().values, sampling_rate=SAMPLING_RATE['ecg'])
                    if len(ecg_cleaned) >= 10:
                        try:
                            r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=SAMPLING_RATE['ecg'])
                            if r_peaks is None or len(r_peaks.get("ECG_R_Peaks", [])) == 0:
                                print(f"No R-peaks detected for Participant {participant_id}, Column {col}")
                                continue
                            hrv_metrics = nk.hrv(r_peaks, sampling_rate=SAMPLING_RATE['ecg'])
                            hrv_metrics = {k: v.values[0] if isinstance(v, pd.Series) else v for k, v in hrv_metrics.items()}
                            baseline_features[col] = {f"Baseline_{feature}": val for feature, val in hrv_metrics.items()}
                        except Exception as e:
                            print(f"Error processing baseline ECG for Participant {participant_id}, Column {col}: {e}")
                            continue

        for task in tasks:
            task_file = os.path.join(base_dir['ecg'], f"ECG_Task {task}_P_{participant_id}.csv")

            if os.path.exists(task_file):
                task_data = pd.read_csv(task_file, low_memory=False)
                for col in ecg_columns:
                    if col in task_data.columns:
                        task_features = {"Participant": participant, "Task": task, "Signal_Type": col}
                        ecg_cleaned = nk.ecg_clean(task_data[col].dropna().values, sampling_rate=SAMPLING_RATE['ecg'])
                        if len(ecg_cleaned) >= 10:
                            try:
                                r_peaks, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=SAMPLING_RATE['ecg'])
                                if r_peaks is None or len(r_peaks.get("ECG_R_Peaks", [])) == 0:
                                    print(f"No R-peaks detected for Participant {participant_id}, Task {task}, Column {col}")
                                    continue
                                hrv_metrics = nk.hrv(r_peaks, sampling_rate=SAMPLING_RATE['ecg'])
                                hrv_metrics = {k: v.values[0] if isinstance(v, pd.Series) else v for k, v in hrv_metrics.items()}
                                task_features.update(hrv_metrics)
                                task_features.update(baseline_features[col])
                                results_per_column[col].append(task_features)
                            except Exception as e:
                                print(f"Error processing ECG for Participant {participant_id}, Task {task}, Column {col}: {e}")
                                continue

    for col, results in results_per_column.items():
        if results:
            results_df = pd.DataFrame(results)
            output_file = os.path.join(output_folder, f"ecg_features_{col.replace(' ', '_')}.csv")
            results_df.to_csv(output_file, index=False)
            print(f'ECG features for {col} saved to: {output_file}')

# Main function
def main():
    process_labels()
    process_gsr()
    process_eda()
    process_ecg()

if __name__ == "__main__":
    main()
