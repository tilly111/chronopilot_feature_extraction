import pandas as pd
import neurokit2 as nk
import os
import glob
import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)



# Parameters
block_details_dir = "dataset5_Markova2021/CLAS/Block_details"
data_dir_base = "dataset5_Markova2021/CLAS/Participants"
output_dir = "agg_data/dataset5"
output_file = os.path.join(output_dir, "ecg_features.csv")
SAMPLING_RATE = 256  
os.makedirs(output_dir, exist_ok=True)

# Function to extract ECG features
def extract_ecg_features(ecg_signal, sampling_rate=SAMPLING_RATE, method="interval-related"):
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        processed_ecg, _ = nk.ecg_process(ecg_cleaned, sampling_rate=sampling_rate)
        features = nk.ecg_analyze(processed_ecg, sampling_rate=sampling_rate, method=method)

        if isinstance(features, pd.DataFrame):
            features = features.to_dict(orient="records")[0]
            features.pop("DFA_alpha2", None)  # Remove DFA_alpha2 feature if present
        return features if isinstance(features, dict) else {}
    except Exception as e:
        print(f"Error in extract_ecg_features: {e}")
        return {}

# Collect all results in a single list
all_results = []

# Iterate through all participants
all_participants = [d for d in os.listdir(data_dir_base) if d.startswith("Part")]

for participant_folder in sorted(all_participants, key=lambda x: int(x.replace("Part", ""))):
    try:
        participant_id = int(participant_folder.replace("Part", ""))
        print(f"Processing participant {participant_id}...")

        block_details_path = os.path.join(block_details_dir, f"Part{participant_id}_Block_Details.csv")
        if not os.path.exists(block_details_path):
            print(f"Block details file not found for participant {participant_id}. Skipping.")
            continue

        block_df = pd.read_csv(block_details_path)
        ecg_file_column = next((col for col in block_df.columns if 'ECG' in col and 'File' in col), None)

        if not ecg_file_column:
            print(f"No column with ECG file names found in {block_details_path}")
            continue

        data_dir = os.path.join(data_dir_base, participant_folder, "by_block")
        baseline_row = block_df[block_df['Block Type'] == 'Baseline'].iloc[0]

        baseline_ecg_pattern = os.path.join(data_dir, baseline_row[ecg_file_column].replace('.csv', '*'))
        baseline_ecg_file = glob.glob(baseline_ecg_pattern)
        baseline_ecg_file = baseline_ecg_file[0] if baseline_ecg_file else None

        baseline_features = {}
        if baseline_ecg_file:
            try:
                baseline_ecg_data = pd.read_csv(baseline_ecg_file)
                if 'ecg2' not in baseline_ecg_data.columns:
                    raise ValueError(f"'ecg2' column not found in {baseline_ecg_file}")
                baseline_ecg_signal = baseline_ecg_data['ecg2']
                baseline_features = extract_ecg_features(baseline_ecg_signal)
            except Exception as e:
                print(f"Error processing baseline ECG for participant {participant_id}: {e}")

        for _, row in block_df.sort_values('Block').iterrows():
            block = row['Block']
            block_type = row['Block Type']

            if block_type in ["Baseline", "Neutral", "IQ Test Response","IQ Test","Math Test Response","Math Test","Stroop Test Response","Stroop Test"]:
                continue

            ecg_file = row[ecg_file_column]
            ecg_pattern = os.path.join(data_dir, ecg_file.replace('.csv', '*'))
            ecg_path = glob.glob(ecg_pattern)
            ecg_path = ecg_path[0] if ecg_path else None

            if not ecg_path or not os.path.exists(ecg_path):
                print(f"ECG file for block {block} not found. Skipping.")
                continue

            try:
                ecg_data = pd.read_csv(ecg_path)
                if 'ecg2' not in ecg_data.columns or ecg_data['ecg2'].isnull().all():
                    print(f"'ecg2' column missing or null in {ecg_path}. Skipping block {block}.")
                    continue

                ecg_signal = ecg_data['ecg2']
                if len(ecg_signal) < 3 * SAMPLING_RATE:  # At least 3 seconds
                    print(f"Signal too short for block {block}. Skipping.")
                    continue

                features_task = extract_ecg_features(ecg_signal)
                if not features_task:
                    print(f"No features extracted for block {block}. Skipping.")
                    continue

                combined_features = {**features_task}
                for key, value in baseline_features.items():
                    combined_features[f"Baseline_{key}"] = value

                combined_features.update({
                    'Participant': participant_id,
                    'Task': block,
                    
                })
                all_results.append(combined_features)
            except Exception as e:
                print(f"Error processing block {block} for participant {participant_id}: {e}")
    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")

# Save all results into a single CSV file
if all_results:
    final_ecg_df = pd.DataFrame(all_results)

    # Adjust column order
    required_columns = ['Participant', 'Task']
    feature_columns = [col for col in final_ecg_df.columns if col not in required_columns]
    baseline_columns = [col for col in feature_columns if col.startswith("Baseline_")]
    task_columns = [col for col in feature_columns if not col.startswith("Baseline_")]
    columns_order = required_columns + sorted(task_columns) + sorted(baseline_columns)
    final_ecg_df = final_ecg_df[columns_order]
    final_ecg_df = final_ecg_df.sort_values(by=['Participant', 'Task'])

    final_ecg_df.to_csv(output_file, index=False)
    print(f"All participant ECG feature table saved to {output_file}.")
else:
    print("No valid data to process.")
