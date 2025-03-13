import pandas as pd
import neurokit2 as nk
import os
import glob
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

block_details_dir = os.path.join(const.BASE_DIR, "dataset5_Markova2021/CLAS/Block_details")
data_dir_base = os.path.join(const.BASE_DIR, "dataset5_Markova2021/CLAS/Participants")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset5")
output_ppg_file = os.path.join(output_folder, "ppg_features.csv")
SAMPLING_RATE = 256  # Predefined sampling rate
os.makedirs(output_folder, exist_ok=True)

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
        ppg_file_column = next((col for col in block_df.columns if 'PPG' in col), None)

        if not ppg_file_column:
            print(f"No column with PPG file names found in {block_details_path}")
            return []

        data_dir = os.path.join(data_dir_base, participant_folder, "by_block")
        baseline_row = block_df[block_df['Block Type'] == 'Baseline'].iloc[0]

        baseline_ppg_pattern = os.path.join(data_dir, baseline_row[ppg_file_column].replace('.csv', '*'))
        baseline_ppg_file = glob.glob(baseline_ppg_pattern)
        baseline_ppg_file = baseline_ppg_file[0] if baseline_ppg_file else None

        baseline_features = {}

        if baseline_ppg_file:
            try:
                baseline_ppg_data = pd.read_csv(baseline_ppg_file)
                if 'ppg' not in baseline_ppg_data.columns:
                    raise ValueError(f"'ppg' column not found in {baseline_ppg_file}")
                baseline_ppg_signal = baseline_ppg_data['ppg']
                baseline_features = extract_ppg_features(baseline_ppg_signal)
                baseline_features = {f"Baseline_{key}": value for key, value in baseline_features.items()}
            except Exception as e:
                print(f"Error processing baseline PPG for participant {participant_id}: {e}")

        ppg_results = []
        for _, row in block_df.sort_values('Block').iterrows():
            block = row['Block']
            block_type = row['Block Type']

            if block_type in ["Baseline", "Neutral", "IQ Test Response", "IQ Test", "Math Test Response", "Math Test", "Stroop Test Response", "Stroop Test"]:
                continue

            ppg_file = row[ppg_file_column]
            ppg_path = os.path.join(data_dir, ppg_file) if isinstance(ppg_file, str) else None

            if not ppg_path or not os.path.exists(ppg_path):
                print(f"PPG file for block {block} not found. Skipping.")
                continue

            try:
                ppg_data = pd.read_csv(ppg_path)
                if 'ppg' not in ppg_data.columns or ppg_data['ppg'].isnull().all():
                    print(f"'ppg' column missing or null in {ppg_path}. Skipping block {block}.")
                    continue

                ppg_signal = ppg_data['ppg']

                # Extract PPG features
                ppg_features_task = extract_ppg_features(ppg_signal)
                if ppg_features_task:
                    ppg_combined_features = {**baseline_features, **ppg_features_task}
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

if all_ppg_results:
    final_ppg_df = pd.DataFrame(all_ppg_results)
    required_columns = ['Participant', 'Task']
    feature_columns = [col for col in final_ppg_df.columns if col not in required_columns]
    baseline_columns = [col for col in feature_columns if col.startswith("Baseline_")]
    task_columns = [col for col in feature_columns if not col.startswith("Baseline_")]
    columns_order = required_columns + sorted(task_columns) + sorted(baseline_columns)
    final_ppg_df = final_ppg_df[columns_order]
    final_ppg_df = final_ppg_df.sort_values(by=['Participant', 'Task'])
    final_ppg_df.to_csv(output_ppg_file, index=False)
    print(f"All participant PPG feature table saved to {output_ppg_file}.")
else:
    print("No valid PPG data to process.")
