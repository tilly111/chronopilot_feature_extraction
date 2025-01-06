import os
import pandas as pd
import neurokit2 as nk

SAMPLING_RATE = 40  # Sampling Rate (Hz)

def extract_file_info(file_name):
    """
    Extracts Participant, Task, and Date from the file name.
    
    Parameters:
        file_name (str): Name of the file.
    
    Returns:
        tuple: A tuple containing (participant, task, date).
    """
    try:
        parts = file_name.split('_')[1].split('-')
        participant = parts[0]
        date = parts[1]
        task = parts[2] if len(parts) > 2 else "Unknown"
        return participant, task, date
    except IndexError:
        return "Unknown", "Unknown", "Unknown"

def process_single_file(file_path):
    """
    Processes a single GSR file and extracts features.
    
    Parameters:
        file_path (str): Path to the GSR file.
    
    Returns:
        dict: A dictionary containing extracted features and metadata.
    """
    file_name = os.path.basename(file_path)
    print(f"Verarbeite Datei: {file_name}")  # Debug-Output

    participant, task, date = extract_file_info(file_name)

    # Load the data
    data = pd.read_csv(file_path)

    # Ensure the 'GSR' column exists
    if "GSR" not in data.columns:
        raise KeyError(f"Die Datei '{file_name}' enthält keine 'GSR'-Spalte.")

    # Extract the GSR signal
    raw_signal = data["GSR"]

    # Process the GSR signal
    processed_signal, info = nk.eda_process(raw_signal, sampling_rate=SAMPLING_RATE, method="neurokit")

    # Extract features
    features_df = nk.eda_analyze(processed_signal, sampling_rate=SAMPLING_RATE, method="interval-related")

    # Convert features to dictionary and add metadata
    if isinstance(features_df, pd.DataFrame):
        features = features_df.to_dict(orient="records")[0]
    else:
        features = features_df

    # Add metadata
    features["Participant"] = participant
    features["Task"] = task
    features["Date"] = date

    return features

def process_filtered_files(filtered_folder, output_folder):
    """
    Processes all GSR files in the specified folder and saves the extracted features.
    
    Parameters:
        filtered_folder (str): Path to the folder containing GSR files.
        output_folder (str): Path to the folder where results will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    gsr_features = []

    # Walk through the folder and process each file
    for root, _, files in os.walk(filtered_folder):
        for file in files:
            if file.endswith("_GSR.csv"):  # Process only files ending with '_GSR.csv'
                file_path = os.path.join(root, file)
                try:
                    features = process_single_file(file_path)
                    gsr_features.append(features)
                except KeyError as e:
                    print(f"Warnung: {e}")  # Warn about missing 'GSR' column
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung der Datei {file}: {e}")

    # Save the results
    output_file = os.path.join(output_folder, "gsr_features.csv")
    df = pd.DataFrame(gsr_features)

    # Sort columns and rows
    columns_order = ["Participant", "Task", "Date"] + [col for col in df.columns if col not in ["Participant", "Task", "Date"]]
    df = df[columns_order]
    df = df.sort_values(by=["Participant", "Task", "Date"])

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Alle Features wurden gespeichert unter: {output_file}")

# Main execution
if __name__ == "__main__":
    # Define input and output folders
    filtered_folder = "filtered_data"
    output_folder = "agg_data/dataset7"

    # Process files
    process_filtered_files(filtered_folder, output_folder)
