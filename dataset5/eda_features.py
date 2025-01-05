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
output_file = os.path.join(output_dir, "eda_features.csv")
SAMPLING_RATE = 256  # Predefined sampling rate
os.makedirs(output_dir, exist_ok=True)

# Function to extract EDA features
def extract_eda_features(eda_signal, sampling_rate=256, method="neurokit"):
    try:
        # Signal reinigen und analysieren
        eda_process, info = nk.eda_process(eda_signal, sampling_rate=sampling_rate, method=method)
        features = nk.eda_analyze(eda_process, sampling_rate=sampling_rate, method="interval-related")

        if isinstance(features, pd.DataFrame):
            features = features.to_dict(orient="records")[0]
        return features if isinstance(features, dict) else {}
    except Exception as e:
        print(f"Error in extract_eda_features: {e}")
        return {}

# Function to process EDA and PPG files for participants
def process_participant_files(participant_id):
    try:
        participant_folder = f"Part{participant_id}"
        block_details_path = os.path.join(block_details_dir, f"{participant_folder}_Block_Details.csv")

        if not os.path.exists(block_details_path):
            print(f"Block details file not found for participant {participant_id}. Skipping.")
            return []

        block_df = pd.read_csv(block_details_path)
        eda_file_column = next((col for col in block_df.columns if 'EDA&PPG' in col), None)

        if not eda_file_column:
            print(f"No column with EDA&PPG file names found in {block_details_path}")
            return []

        data_dir = os.path.join(data_dir_base, participant_folder, "by_block")
        results = []

        baseline_row = block_df[block_df['Block Type'] == 'Baseline'].iloc[0]
        baseline_file_pattern = os.path.join(data_dir, baseline_row[eda_file_column].replace('.csv', '*'))
        baseline_file = glob.glob(baseline_file_pattern)
        baseline_file = baseline_file[0] if baseline_file else None

        baseline_features = {}
        if baseline_file:
            try:
                baseline_data = pd.read_csv(baseline_file)
                if 'gsr' not in baseline_data.columns:
                    raise ValueError(f"'gsr' column not found in {baseline_file}")

                baseline_signal = baseline_data['gsr']
                baseline_features = extract_eda_features(baseline_signal, sampling_rate=SAMPLING_RATE)
            except Exception as e:
                print(f"Error processing baseline EDA for participant {participant_id}: {e}")

        for _, row in block_df.sort_values('Block').iterrows():
            block = row['Block']
            block_type = row['Block Type']

            # Skip Baseline, Neutral blocks, and blocks with Length < 10 seconds
            if block_type in ["Baseline", "Neutral", "IQ Test Response","IQ Test","Math Test Response","Math Test","Stroop Test Response","Stroop Test"]:
                continue

            eda_file = row[eda_file_column]
            eda_pattern = os.path.join(data_dir, eda_file.replace('.csv', '*'))
            eda_path = glob.glob(eda_pattern)
            eda_path = eda_path[0] if eda_path else None

            if not eda_path or not os.path.exists(eda_path):
                print(f"EDA file for block {block} not found. Skipping.")
                continue

            try:
                eda_data = pd.read_csv(eda_path)
                if 'gsr' not in eda_data.columns or eda_data['gsr'].isnull().all():
                    print(f"'gsr' column missing or null in {eda_path}. Skipping block {block}.")
                    continue

                eda_signal = eda_data['gsr']
                features_task = extract_eda_features(eda_signal, sampling_rate=SAMPLING_RATE)

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
                results.append(combined_features)
            except Exception as e:
                print(f"Error processing block {block} for participant {participant_id}: {e}")

        return results
    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")
        return []

# Process all participants
all_results = []
all_participants = [d for d in os.listdir(data_dir_base) if d.startswith("Part")]

for participant_folder in sorted(all_participants, key=lambda x: int(x.replace("Part", ""))):
    participant_id = int(participant_folder.replace("Part", ""))
    print(f"Processing participant {participant_id}...")
    results = process_participant_files(participant_id)
    all_results.extend(results)

# Save all results into a single CSV file
if all_results:
    final_eda_df = pd.DataFrame(all_results)

    # Adjust column order
    required_columns = ['Participant', 'Task']
    feature_columns = [col for col in final_eda_df.columns if col not in required_columns]
    baseline_columns = [col for col in feature_columns if col.startswith("Baseline_")]
    task_columns = [col for col in feature_columns if not col.startswith("Baseline_")]
    columns_order = required_columns + sorted(task_columns) + sorted(baseline_columns)
    final_eda_df = final_eda_df[columns_order]
    final_eda_df = final_eda_df.sort_values(by=['Participant', 'Task'])

    final_eda_df.to_csv(output_file, index=False)
    print(f"All participant EDA feature table saved to {output_file}.")
else:
    print("No valid data to process.")
