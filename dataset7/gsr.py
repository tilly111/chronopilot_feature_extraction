import os
import pandas as pd
import neurokit2 as nk

SAMPLING_RATE = 40

def extract_file_info(file_name):
    """Extracts Participant, Task, and Date information from the file name."""
    parts = file_name.split('_')
    participant_task = parts[1].split('-')
    participant = participant_task[0]
    task = participant_task[2] if len(participant_task) > 2 else "Unknown"
    date = participant_task[1] if len(participant_task) > 1 else "Unknown"
    return participant, task, date

def process_single_file(file_path):
    """Processes a single file and extracts GSR features."""
    file_name = os.path.basename(file_path)
    participant, task, date = extract_file_info(file_name)

    # Load the file
    data = pd.read_csv(file_path)

    # Check if the 'GSR' column exists
    if "GSR" not in data.columns:
        raise KeyError(f"The file '{file_name}' does not contain the required 'GSR' column.")

    # Clean the GSR data
    raw_signal = data["GSR"]
    cleaned_signal = nk.eda_clean(raw_signal, sampling_rate=SAMPLING_RATE)

    # Fully process the EDA signal and extract features
    eda_signals, _ = nk.eda_process(cleaned_signal, sampling_rate=SAMPLING_RATE)
    features_df = nk.eda_intervalrelated(eda_signals, sampling_rate=SAMPLING_RATE)

    # Convert features to dictionary and add metadata
    if isinstance(features_df, pd.DataFrame):
        features = features_df.to_dict(orient="records")[0]  # Convert DataFrame to a dictionary
    else:
        features = features_df

    features["Participant"] = participant
    features["Task"] = task
    features["Date"] = date
    return features

def process_filtered_files(filtered_folder, output_folder):
    """Processes all files in a folder and saves the features."""
    os.makedirs(output_folder, exist_ok=True)
    gsr_features = []

    for root, _, files in os.walk(filtered_folder):
        for file in files:
            if file.endswith("_GSR.csv"):  # Process only files ending with '_GSR.csv'
                file_path = os.path.join(root, file)
                try:
                    features = process_single_file(file_path)
                    gsr_features.append(features)
                except KeyError as e:
                    print(e)  # Inform the user if a file is skipped due to missing 'GSR'

    # Save results
    output_file = os.path.join(output_folder, "gsr_features.csv")
    df = pd.DataFrame(gsr_features)

    columns_order = ["Participant", "Task", "Date"] + [col for col in df.columns if col not in ["Participant", "Task", "Date"]]
    df = df[columns_order]

    # Sort rows by Participant, Task, and Date
    df = df.sort_values(by=["Participant", "Task", "Date"])

    df.to_csv(output_file, index=False)

# Start processing
process_filtered_files("filtered_data", "agg_data/dataset7")
