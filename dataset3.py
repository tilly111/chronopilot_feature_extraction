import pandas as pd
import numpy as np
import neurokit2 as nk
import os

# Base directories
BASE_DIR = "dataset3_MAUS/Data/Raw_data"
OUTPUT_FOLDER = "agg_data/dataset3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Sampling rates
SAMPLING_RATE = {
    "gsr": 256,
    "ppg": 256,
    "pixart": 100,
    "ecg": 256
}

# Participants range
PARTICIPANTS = range(1, 26)

def process_signal(participant, participant_dir, modality, resting_col, task_file, process_func):
    """
    General function to process physiological signals.
    """
    try:
        resting_file = os.path.join(participant_dir, "inf_resting.csv")
        task_file_path = os.path.join(participant_dir, task_file)

        if not os.path.exists(resting_file) or not os.path.exists(task_file_path):
            return []

        resting_data = pd.read_csv(resting_file)
        if resting_col not in resting_data.columns:
            return []

        baseline_signal = pd.to_numeric(resting_data[resting_col].dropna(), errors='coerce').values
        processed_baseline, _ = process_func(baseline_signal, sampling_rate=SAMPLING_RATE[modality])
        baseline_features = {f"{modality.upper()}_Baseline_{key}": val for key, val in processed_baseline.mean(axis=0).to_dict().items()}

        task_data = pd.read_csv(task_file_path)
        results = []
        for idx, column in enumerate(task_data.columns, start=1):
            signal = pd.to_numeric(task_data[column].dropna(), errors='coerce').values
            processed_task, _ = process_func(signal, sampling_rate=SAMPLING_RATE[modality])
            task_features = {f"{modality.upper()}_Task_{key}": val for key, val in processed_task.mean(axis=0).to_dict().items()}

            results.append({
                'Participant': participant,
                'Task': idx,
                **task_features,
                **baseline_features
            })

        return results

    except Exception as e:
        print(f"Error processing {modality.upper()} for participant {participant}: {e}")
        return []

def process_labels():
    """
    Process subjective ratings and derive well-being labels.
    """
    source_folder = "dataset3_MAUS/Subjective_rating"
    task_mapping = {
        "Trial 1: 0_back": 1,
        "Trial 2: 2_back": 2,
        "Trial 3: 3_back": 3,
        "Trial 4: 2_back": 4,
        "Trial 5: 3_back": 5,
        "Trial 6: 0_back": 6
    }
    results = []

    for participant_id in os.listdir(source_folder):
        source_subfolder = os.path.join(source_folder, participant_id)

        if os.path.isdir(source_subfolder):
            nasa_tlx_path = os.path.join(source_subfolder, "NASA_TLX.csv")

            if os.path.exists(nasa_tlx_path):
                df = pd.read_csv(nasa_tlx_path).set_index("Scale Title").T.apply(pd.to_numeric, errors='coerce')

                for trial, row in df.iterrows():
                    task = task_mapping.get(trial)
                    if task is None:
                        continue

                    mental = row.get("Mental Demand", 0)
                    physical = row.get("Physical Demand", 0)
                    temporal = row.get("Temporal Demand", 0)
                    frustration = row.get("Frustration", 0)
                    well_being_index = (100 - (mental + physical + temporal + frustration) / 4)
                    well_being = 1 if well_being_index >= 50 else 0

                    participant_number = int(participant_id.lstrip('0'))

                    results.append({
                        "Participant": participant_number,
                        "Task": task,
                        "Well-being": well_being,
                        "Well-being Index": round(well_being_index, 2),
                        "Mental Demand": mental,
                        "Physical Demand": physical,
                        "Temporal Demand": temporal,
                        "Frustration": frustration
                    })

    if results:
        labels_df = pd.DataFrame(results).sort_values(by=["Participant", "Task"]).reset_index(drop=True)
        return labels_df
    return None

def main():


    print("Processing labels...")
    labels_df = process_labels()
    if labels_df is not None:
        labels_file_path = os.path.join(OUTPUT_FOLDER, "labels.csv")
        labels_df.to_csv(labels_file_path, index=False)
        print(f"Labels saved to {labels_file_path}")
    else:
        print("No labels found to save.")

    print("Processing GSR...")
    gsr_results = []
    for participant in PARTICIPANTS:
        participant_dir = f"{BASE_DIR}/{participant:03}"
        gsr_results.extend(process_signal(participant, participant_dir, "gsr", "Resting_GSR", "inf_gsr.csv", nk.eda_process))

    gsr_file_path = os.path.join(OUTPUT_FOLDER, "gsr_features.csv")
    if gsr_results:
        pd.DataFrame(gsr_results).to_csv(gsr_file_path, index=False)
        print(f"GSR features saved to {gsr_file_path}")

    print("Processing PPG...")
    ppg_results = []
    for participant in PARTICIPANTS:
        participant_dir = f"{BASE_DIR}/{participant:03}"
        ppg_results.extend(process_signal(participant, participant_dir, "ppg", "Resting_PPG", "inf_ppg.csv", nk.ppg_process))

    ppg_file_path = os.path.join(OUTPUT_FOLDER, "ppg_features.csv")
    if ppg_results:
        pd.DataFrame(ppg_results).to_csv(ppg_file_path, index=False)
        print(f"PPG features saved to {ppg_file_path}")

    print("Processing Pixart PPG...")
    pixart_results = []
    for participant in PARTICIPANTS:
        participant_dir = f"{BASE_DIR}/{participant:03}"
        pixart_results.extend(process_signal(participant, participant_dir, "pixart", "Resting", "pixart.csv", nk.ppg_process))

    pixart_file_path = os.path.join(OUTPUT_FOLDER, "pixart_features.csv")
    if pixart_results:
        pd.DataFrame(pixart_results).to_csv(pixart_file_path, index=False)
        print(f"Pixart PPG features saved to {pixart_file_path}")

    print("Processing ECG...")
    ecg_results = []
    for participant in PARTICIPANTS:
        participant_dir = f"{BASE_DIR}/{participant:03}"
        ecg_results.extend(process_signal(participant, participant_dir, "ecg", "Resting_ECG", "inf_ecg.csv", nk.ecg_process))

    ecg_file_path = os.path.join(OUTPUT_FOLDER, "ecg_features.csv")
    if ecg_results:
        pd.DataFrame(ecg_results).to_csv(ecg_file_path, index=False)
        print(f"ECG features saved to {ecg_file_path}")

    print("Processing completed.")

if __name__ == "__main__":
    main()
