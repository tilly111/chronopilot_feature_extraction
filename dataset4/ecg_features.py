import os
import pandas as pd
import neurokit2 as nk
import wfdb
import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Constants
FOLDER_PATH = 'dataset4_Vollmer2022/generated_data/'
SAMPLING_RATE = 256  
TASK_DURATION_MINUTES = 5 
TASK_DURATION_SAMPLES = TASK_DURATION_MINUTES * 60 * SAMPLING_RATE 
OUTPUT_FOLDER = "agg_data/dataset4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_tasks(aux_annotations, signal_length):
    task_map = {}  # Map for task starts: {Task-Name: [Start, End, Status]}
    ignore_labels = ["Wrong Marker"]  # Labels to ignore

    for i, (start, label) in enumerate(zip(aux_annotations.sample, aux_annotations.aux_note)):
        label = label.strip()

        # Ignore irrelevant markers
        if any(ignore in label for ignore in ignore_labels):
            continue

        # Extract task name
        task_name = label.split("/")[-1]

        # Prioritize Manual/ as start, otherwise FAROS_Marker/
        if "Manual/" in label:
            task_map[task_name] = {'start': start, 'end': None, 'status': 'Manual'}
        elif "FAROS_Marker" in label:
            # If task does not exist, add it
            if task_name not in task_map:
                task_map[task_name] = {'start': start, 'end': None, 'status': 'FAROS'}
            else:
                # If it already exists, set the endpoint
                if task_map[task_name]['end'] is None:  # Ensure end is not already set
                    next_index = i + 1
                    task_map[task_name]['end'] = (
                        aux_annotations.sample[next_index]
                        if next_index < len(aux_annotations.sample)
                        else start + TASK_DURATION_SAMPLES  # Default end if no subsequent marker
                    )

    # Convert task_map to tasks list
    tasks = []
    for task_name, times in task_map.items():
        tasks.append({
            'label': task_name,
            'start': times['start'],
            'end': times['end'] or (times['start'] + TASK_DURATION_SAMPLES),  
            'status': times['status']
        })

    return tasks

# Function to extract features with a defined baseline
def extract_task_and_baseline_features(tasks, ecg_signal, participant):
    results = []
    baseline_features = None  # Store baseline features

    for idx, task in enumerate(tasks, start=1):  # Add numeric task index
        start, end = task['start'], task['end']
        task_name = task['label']

        # Extract task-specific ECG segment
        task_segment = ecg_signal[start:end]

        # Check if the task is "Rest" to set as baseline
        if task_name.lower() == "rest":
            process_baseline, _ = nk.ecg_process(task_segment, sampling_rate=SAMPLING_RATE)
            baseline_features = nk.ecg_analyze(process_baseline, sampling_rate=SAMPLING_RATE, method="interval-related")

            # Add Participant and Task information
            baseline_features.insert(0, 'Participant', participant)
            baseline_features.insert(1, 'Task', idx)  # Use numeric index for task

            print(f"Baseline features extracted for task: {task_name}")
            continue

        # Process ECG data for the task
        process_task, _ = nk.ecg_process(task_segment, sampling_rate=SAMPLING_RATE)
        features_task = nk.ecg_analyze(process_task, sampling_rate=SAMPLING_RATE, method="interval-related")

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

# Main function
def main():
    all_faros_results = []
    all_sot_results = []
    all_nexus_results = []
    all_hexoskin_results = []

    for i, file_prefix in enumerate([f"x{str(j).zfill(3)}" for j in range(1, 14)]):
        file_path = os.path.join(FOLDER_PATH, file_prefix)

        # Check if necessary files exist
        if not os.path.exists(f"{file_path}.aux") or not os.path.exists(f"{file_path}.dat"):
            print(f"File {file_prefix} missing AUX or DAT file.")
            continue

        # Load AUX file and record
        aux_annotations = wfdb.rdann(file_path, 'aux')
        record = wfdb.rdrecord(file_path)

        # Process each channel
        channels = {
            "FAROS/ECG_filtered": all_faros_results,
            "SOT/EKG_filtered": all_sot_results,
            "NEXUS/Sensor-B:EEG_filtered": all_nexus_results,
            "HEXOSKIN/ECG_I_filtered": all_hexoskin_results
        }

        for channel, results_list in channels.items():
            try:
                ecg_signal = record.p_signal[:, record.sig_name.index(channel)]
                signal_length = len(ecg_signal)
                tasks = process_tasks(aux_annotations, signal_length)

                # Extract features
                df_results = extract_task_and_baseline_features(tasks, ecg_signal, participant=str(i + 1))
                if not df_results.empty:
                    results_list.append(df_results)
            except ValueError:
                print(f"No {channel} channel found in {file_prefix}.")

    for channel_name, results_list in zip(["FAROS", "SOT", "NEXUS", "HEXOSKIN"],
                                          [all_faros_results, all_sot_results, all_nexus_results, all_hexoskin_results]):
        if results_list:
            combined_results = pd.concat(results_list, ignore_index=True)
            output_file = os.path.join(OUTPUT_FOLDER, f"ecg_features_{channel_name}.csv")
            combined_results.to_csv(output_file, index=False)
            print(f"\nAll {channel_name} features saved to {output_file}")
        else:
            print(f"No {channel_name} features extracted.")

if __name__ == "__main__":
    main()
