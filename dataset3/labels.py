import pandas as pd
import numpy as np
import os

BASE_DIR = "dataset3_MAUS/Data/Raw_data"
OUTPUT_FOLDER = "agg_data/dataset3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PARTICIPANTS = range(1, 26)

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


if __name__ == "__main__":
    main()
