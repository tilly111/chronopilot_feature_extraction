import os
import pandas as pd
import neurokit2 as nk
import wfdb

# Folder path
folder_path = 'dataset4_Vollmer2022/generated_data/'

# Sampling frequency
sampling_frequency = 256  # Hz
task_duration_minutes = 5  # Minutes
task_duration_samples = task_duration_minutes * 60 * sampling_frequency  # Samples per task

# Function to process tasks with start and end points
def process_tasks(aux_annotations, signal_length):
    tasks = []
    ignore_labels = ["Wrong Marker"] 
    processed_labels = set() 

    for i, (start, label) in enumerate(zip(aux_annotations.sample, aux_annotations.aux_note)):
        label = label.strip()
        if any(ignore in label for ignore in ignore_labels):
            continue

        task_name = label.split("/")[-1]

        # Skip if task already processed
        if task_name in processed_labels:
            continue

        end = min(start + task_duration_samples, signal_length)  # Prevent signal overflow

        # Add task and mark as processed
        tasks.append({'label': task_name, 'start': start, 'end': end})
        processed_labels.add(task_name)

    return tasks

# Function to extract features with a defined baseline
def extract_task_and_baseline_features(tasks, ecg_signal, sampling_rate, participant):
    results = []
    baseline_features = None  # Store baseline features

    for idx, task in enumerate(tasks, start=1):  # Add numeric task index
        start, end = task['start'], task['end']
        task_name = task['label']

        # Extract task-specific ECG segment
        task_segment = ecg_signal[start:end]

        # Check if the task is "Rest" to set as baseline
        if task_name.lower() == "rest":
            process_baseline, _ = nk.ecg_process(task_segment, sampling_rate=sampling_rate)
            baseline_features = nk.ecg_analyze(process_baseline, sampling_rate=sampling_rate, method="interval-related")

            # Add Participant and Task information
            baseline_features.insert(0, 'Participant', participant)
            baseline_features.insert(1, 'Task', idx)  # Use numeric index for task

            print(f"Baseline features extracted for task: {task_name}")
            continue

        # Process ECG data for the task
        process_task, _ = nk.ecg_process(task_segment, sampling_rate=sampling_rate)
        features_task = nk.ecg_analyze(process_task, sampling_rate=sampling_rate, method="interval-related")

        # Add Participant and Task information
        features_task.insert(0, 'Participant', participant)
        features_task.insert(1, 'Task', idx)  # Use numeric index for task

        # Add Baseline Features to the Task
        if baseline_features is not None:
            for col in baseline_features.columns:
                if col not in ["Participant", "Task"]:
                    features_task[f"Baseline_{col}"] = baseline_features[col].iloc[0]

        results.append(features_task)

    # Combine results into a DataFrame
    if results:
        combined_results = pd.concat(results, ignore_index=True)
        # Ensure Participant and Task are the first columns
        columns_order = ['Participant', 'Task'] + [col for col in combined_results.columns if col not in ['Participant', 'Task']]
        return combined_results[columns_order]
    else:
        print("No features extracted.")
        return pd.DataFrame()

# Create output folder
output_folder = "agg_data/dataset4"
os.makedirs(output_folder, exist_ok=True)

# Main function
def main():
    all_faros_results = []
    all_sot_results = []
    all_nexus_results = []
    all_hexoskin_results = []

    for i, file_prefix in enumerate([f"x{str(j).zfill(3)}" for j in range(1, 14)]):
        file_path = os.path.join(folder_path, file_prefix)

        # Check if necessary files exist
        if not os.path.exists(f"{file_path}.aux") or not os.path.exists(f"{file_path}.dat"):
            print(f"File {file_prefix} missing AUX or DAT file.")
            continue

        # Load AUX file and record
        aux_annotations = wfdb.rdann(file_path, 'aux')
        record = wfdb.rdrecord(file_path)

        # Process FAROS/ECG_filtered channel
        try:
            ecg_signal_faros = record.p_signal[:, record.sig_name.index("FAROS/ECG_filtered")]
            signal_length = len(ecg_signal_faros)
            tasks = process_tasks(aux_annotations, signal_length)

            # Extract features for FAROS
            df_faros_results = extract_task_and_baseline_features(tasks, ecg_signal_faros, sampling_frequency, participant=str(i + 1))

            if not df_faros_results.empty:
                all_faros_results.append(df_faros_results)
        except ValueError:
            print(f"No FAROS/ECG_filtered channel found in {file_prefix}.")

        # Process SOT/EKG_filtered channel
        try:
            ecg_signal_sot = record.p_signal[:, record.sig_name.index("SOT/EKG_filtered")]
            signal_length = len(ecg_signal_sot)
            tasks = process_tasks(aux_annotations, signal_length)

            # Extract features for SOT
            df_sot_results = extract_task_and_baseline_features(tasks, ecg_signal_sot, sampling_frequency, participant=str(i + 1))

            if not df_sot_results.empty:
                all_sot_results.append(df_sot_results)
        except ValueError:
            print(f"No SOT/EKG_filtered channel found in {file_prefix}.")

        # Process NEXUS/Sensor-B:EEG_filtered channel
        try:
            ecg_signal_nexus = record.p_signal[:, record.sig_name.index("NEXUS/Sensor-B:EEG_filtered")]
            signal_length = len(ecg_signal_nexus)
            tasks = process_tasks(aux_annotations, signal_length)

            # Extract features for NEXUS
            df_nexus_results = extract_task_and_baseline_features(tasks, ecg_signal_nexus, sampling_frequency, participant=str(i + 1))

            if not df_nexus_results.empty:
                all_nexus_results.append(df_nexus_results)
        except ValueError:
            print(f"No NEXUS/Sensor-B:EEG_filtered channel found in {file_prefix}.")

        # Process HEXOSKIN/ECG_I_filtered channel
        try:
            ecg_signal_hexoskin = record.p_signal[:, record.sig_name.index("HEXOSKIN/ECG_I_filtered")]
            signal_length = len(ecg_signal_hexoskin)
            tasks = process_tasks(aux_annotations, signal_length)

            # Extract features for HEXOSKIN
            df_hexoskin_results = extract_task_and_baseline_features(tasks, ecg_signal_hexoskin, sampling_frequency, participant=str(i + 1))

            if not df_hexoskin_results.empty:
                all_hexoskin_results.append(df_hexoskin_results)
        except ValueError:
            print(f"No HEXOSKIN/ECG_I_filtered channel found in {file_prefix}.")

    # Combine results for FAROS
    if all_faros_results:
        combined_faros_results = pd.concat(all_faros_results, ignore_index=True)
        output_faros_file = os.path.join(output_folder, "ecg_features_FAROS.csv")
        combined_faros_results.to_csv(output_faros_file, index=False)
        print(f"\nAll FAROS features saved to {output_faros_file}")
    else:
        print("No FAROS features extracted.")

    # Combine results for SOT
    if all_sot_results:
        combined_sot_results = pd.concat(all_sot_results, ignore_index=True)
        output_sot_file = os.path.join(output_folder, "ecg_features_SOT.csv")
        combined_sot_results.to_csv(output_sot_file, index=False)
        print(f"\nAll SOT features saved to {output_sot_file}")
    else:
        print("No SOT features extracted.")

    # Combine results for NEXUS
    if all_nexus_results:
        combined_nexus_results = pd.concat(all_nexus_results, ignore_index=True)
        output_nexus_file = os.path.join(output_folder, "ecg_features_NEXUS.csv")
        combined_nexus_results.to_csv(output_nexus_file, index=False)
        print(f"\nAll NEXUS features saved to {output_nexus_file}")
    else:
        print("No NEXUS features extracted.")

    # Combine results for HEXOSKIN
    if all_hexoskin_results:
        combined_hexoskin_results = pd.concat(all_hexoskin_results, ignore_index=True)
        output_hexoskin_file = os.path.join(output_folder, "ecg_features_HEXOSKIN.csv")
        combined_hexoskin_results.to_csv(output_hexoskin_file, index=False)
        print(f"\nAll HEXOSKIN features saved to {output_hexoskin_file}")
    else:
        print("No HEXOSKIN features extracted.")

if __name__ == "__main__":
    main()
