import os
import pandas as pd
import neurokit2 as nk
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("gsr_processing.log"),
        logging.StreamHandler()
    ]
)

SAMPLING_RATE = 40  # Hz
WINDOW_LENGTH = const.INTERVAL * SAMPLING_RATE
STEP_SIZE = const.STEP * SAMPLING_RATE

def extract_file_info(file_name):
    try:
        parts = file_name.split('_')[1].split('-')
        participant = parts[0]
        date = parts[1]
        task = parts[2] if len(parts) > 2 else "Unknown"
        return participant, task, date
    except IndexError:
        return "Unknown", "Unknown", "Unknown"

def process_single_file(file_path):
    file_name = os.path.basename(file_path)
    logging.info(f"Processing file: {file_name}")

    participant, task, date = extract_file_info(file_name)

    data = pd.read_csv(file_path)
    if "GSR" not in data.columns:
        raise KeyError(f"The file '{file_name}' does not contain a 'GSR' column.")

    raw_signal = data["GSR"].dropna().values

    if len(raw_signal) < WINDOW_LENGTH:
        logging.warning(f"Signal too short for interval processing: {file_name}")
        return []

    results = []
    slice_count = 1
    for start in range(0, len(raw_signal) - WINDOW_LENGTH + 1, STEP_SIZE):
        end = start + WINDOW_LENGTH
        window = raw_signal[start:end]

        try:
            processed_signal, _ = nk.eda_process(window, sampling_rate=SAMPLING_RATE)
            features_df = nk.eda_analyze(processed_signal, sampling_rate=SAMPLING_RATE, method="interval-related")

            if isinstance(features_df, pd.DataFrame):
                features = features_df.to_dict(orient="records")[0]
            else:
                features = features_df

            features["Participant"] = participant
            features["Task"] = task
            features["Date"] = date
            features["Slice"] = slice_count

            results.append(features)
            slice_count += 1

        except Exception as e:
            logging.error(f"Error processing slice {slice_count} in file {file_name}: {e}")
            continue

    return results

def process_filtered_files(filtered_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    all_features = []

    for root, _, files in os.walk(filtered_folder):
        for file in files:
            if file.endswith("_GSR.csv"):
                file_path = os.path.join(root, file)
                try:
                    features_list = process_single_file(file_path)
                    all_features.extend(features_list)
                except KeyError as e:
                    logging.warning(e)
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")

    output_file = os.path.join(output_folder, "gsr_features.csv")
    df = pd.DataFrame(all_features)

    if not df.empty:
        columns_order = ["Participant", "Task", "Date", "Slice"] + [col for col in df.columns if col not in ["Participant", "Task", "Date", "Slice"]]
        df = df[columns_order]
        df = df.sort_values(by=["Participant", "Task", "Date", "Slice"])
        df.to_csv(output_file, index=False)
        logging.info(f"All GSR features have been saved to: {output_file}")
    else:
        logging.warning("No GSR features were extracted.")

# Main execution
if __name__ == "__main__":
    filtered_folder = const.FILTERED_DIR
    output_folder = os.path.join(const.OUTPUT_DIR, "dataset7")

    logging.info("Starting GSR feature extraction with sliding window...")
    process_filtered_files(filtered_folder, output_folder)
    logging.info("Processing complete.")
