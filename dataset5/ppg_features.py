import pandas as pd
import neurokit2 as nk
import os
import glob
import warnings

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)



block_details_dir = "dataset5_Markova2021/CLAS/Block_details"
data_dir_base = "dataset5_Markova2021/CLAS/Participants"
output_dir = "agg_data/dataset5"
output_ppg_file = os.path.join(output_dir, "ppg_features.csv")
SAMPLING_RATE = 256  # Predefined sampling rate
os.makedirs(output_dir, exist_ok=True)

# Function to extract PPG features
def extract_ppg_features(ppg_signal):
    try:
        ppg_process, info = nk.ppg_process(ppg_signal, sampling_rate=SAMPLING_RATE)
        features = nk.ppg_analyze(ppg_process, sampling_rate=SAMPLING_RATE, method="interval-related")

        if isinstance(features, pd.DataFrame):
            features = features.to_dict(orient="records")[0]
        return features if isinstance(features, dict) else {}
    except Exception as e:
        print(f"Error in extract_ppg_features: {e}")
        return {}

# Function to process PPG files for participants
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
        ppg_results = []

        for _, row in block_df.sort_values('Block').iterrows():
            block = row['Block']
            block_type = row['Block Type']

            # Skip 
            if block_type in ["Baseline", "Neutral", "IQ Test Response","IQ Test","Math Test","Math Test Response","Stroop Test Response","Stroop Test"]:
                continue

            eda_file = row[eda_file_column]
            eda_pattern = os.path.join(data_dir, eda_file.replace('.csv', '*'))
            eda_path = glob.glob(eda_pattern)
            eda_path = eda_path[0] if eda_path else None

            if not eda_path or not os.path.exists(eda_path):
                print(f"PPG file for block {block} not found. Skipping.")
                continue

            try:
                eda_data = pd.read_csv(eda_path)
                if 'ppg' not in eda_data.columns or eda_data['ppg'].isnull().all():
                    print(f"'ppg' column missing or null in {eda_path}. Skipping block {block}.")
                    continue

                ppg_signal = eda_data['ppg']

                # Extract PPG features
                ppg_features_task = extract_ppg_features(ppg_signal)
                if ppg_features_task:
                    ppg_combined_features = {**ppg_features_task}
                    ppg_combined_features.update({
                        'Participant': participant_id,
                        'Task': block,
                    })
                    ppg_results.append(ppg_combined_features)

            except Exception as e:
                print(f"Error processing block {block} for participant {participant_id}: {e}")

        return ppg_results
    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")
        return []

# Process all participants
all_ppg_results = []
all_participants = [d for d in os.listdir(data_dir_base) if d.startswith("Part")]

for participant_folder in sorted(all_participants, key=lambda x: int(x.replace("Part", ""))):
    participant_id = int(participant_folder.replace("Part", ""))
    print(f"Processing participant {participant_id}...")
    ppg_results = process_participant_files(participant_id)
    all_ppg_results.extend(ppg_results)

# Save PPG results into a CSV file
if all_ppg_results:
    final_ppg_df = pd.DataFrame(all_ppg_results)
    required_columns = ['Participant', 'Task']
    feature_columns = [col for col in final_ppg_df.columns if col not in required_columns]
    columns_order = required_columns + sorted(feature_columns)
    final_ppg_df = final_ppg_df[columns_order]
    final_ppg_df = final_ppg_df.sort_values(by=['Participant', 'Task'])
    final_ppg_df.to_csv(output_ppg_file, index=False)
    print(f"All participant PPG feature table saved to {output_ppg_file}.")
else:
    print("No valid PPG data to process.")
