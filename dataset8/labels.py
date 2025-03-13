import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const

# Paths
input_dir = os.path.join(const.BASE_DIR, "dataset8_POPANE")
output_file = os.path.join(const.OUTPUT_DIR, "dataset8/labels.csv")

data_list = []

for stimuli_folder in os.listdir(input_dir):
    # Skip folders containing "Neutral" or named "Baselines"
    if "Neutral" in stimuli_folder or stimuli_folder == "Baselines":
        continue

    stimuli_path = os.path.join(input_dir, stimuli_folder)
    if not os.path.isdir(stimuli_path):
        continue

    for file_name in os.listdir(stimuli_path):
        # Skip files containing "Neutral"
        if "Neutral" in file_name or not file_name.endswith(".csv"):
            continue

        parts = file_name.split("_")
        if len(parts) < 3:
            continue

        task = parts[0][1:]
        participant = parts[1][1:]

        file_path = os.path.join(stimuli_path, file_name)

        with open(file_path, "r") as f:
            lines = f.readlines()

        data_start_idx = next(i for i, line in enumerate(lines) if line.startswith("timestamp"))
        df = pd.read_csv(file_path, skiprows=data_start_idx)

        if "affect" not in df.columns:
            continue

        affect_mean = df["affect"].mean()
        well_being = 1 if affect_mean > 5 else 0

        df.columns = df.columns.str.strip()
        first_marker = df["marker"].dropna().iloc[0] if "marker" in df.columns and not df["marker"].dropna().empty else None

        # Add processed data to the list
        data_list.append({
            "Participant": int(participant),
            "Task": int(task),
            "Marker": first_marker,
            "Stimuli": stimuli_folder,
            "Well-being": well_being,
            "Affect": round(affect_mean, 2)
        })

output_df = pd.DataFrame(data_list)

output_df = output_df.sort_values(by=["Participant", "Task", "Marker"]).reset_index(drop=True)
output_df = output_df[["Participant", "Task", "Marker", "Stimuli", "Well-being", "Affect"]]

os.makedirs(os.path.dirname(output_file), exist_ok=True)
output_df.to_csv(output_file, index=False)

print(f"Aggregated data saved to '{output_file}'!")
