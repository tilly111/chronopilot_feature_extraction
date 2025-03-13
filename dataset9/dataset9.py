import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const


file_path = os.path.join(const.BASE_DIR, "dataset9_Pires2023/data/physiol_processed_short.csv")
data = pd.read_csv(file_path)

output_dir = os.path.join(const.OUTPUT_DIR, "dataset9")
os.makedirs(output_dir, exist_ok=True)

# Add numeric IDs for Participant and Sound
data["Participant"] = data["Subject"].astype("category").cat.codes + 1  # Numeric IDs start at 1
data["Sound_ID"] = data["Sound"].astype("category").cat.codes + 1

# Add Well-being Labels
data["Well-being"] = data["vocAuth"].apply(lambda x: 1 if x == "Laughter_Real" else 0)

# Save Labels
labels_df = data[
    ["Participant", "Subject", "Sound", "Sound_ID", "Well-being", "vocAuth"]
]
labels_file = os.path.join(output_dir, "labels.csv")
labels_df.to_csv(labels_file, index=False)
print(f"Labels file saved to {labels_file}.")

# Feature Calculation Functions
def calculate_features(data, prefix, bins=18):
    """
    Calculate features for bin data similar to NeuroKit2.
    Parameters:
        - data: DataFrame containing the bins.
        - prefix: Prefix for bin columns (e.g., 'eda', 'emg_orb').
        - bins: Number of bins.
    Returns:
        - feature_dict: Dictionary containing calculated features.
    """
    bin_columns = [f"{prefix}_bin{i}" for i in range(1, bins + 1)]
    bin_values = data[bin_columns].values
    return {
        f"{prefix}_Mean": np.mean(bin_values, axis=1),
        f"{prefix}_SD": np.std(bin_values, axis=1),
        f"{prefix}_Min": np.min(bin_values, axis=1),
        f"{prefix}_Max": np.max(bin_values, axis=1),
        f"{prefix}_Range": np.max(bin_values, axis=1) - np.min(bin_values, axis=1),
        f"{prefix}_Sum": np.sum(bin_values, axis=1),
    }

def calculate_change_to_baseline(data, feature_column, baseline_column):
    """Calculate change to baseline."""
    return data[feature_column] - data[baseline_column]

def calculate_relative_change(data, feature_column, baseline_column):
    """Calculate relative change to baseline."""
    return data[feature_column] / (data[baseline_column] + 1e-9)

# 3. EDA Features
eda_features_dict = calculate_features(data, prefix="eda")
eda_features = pd.DataFrame(eda_features_dict)
eda_features["Participant"] = data["Participant"]
eda_features["Sound_ID"] = data["Sound_ID"]
eda_features["EDA_Tonic_Mean"] = eda_features_dict["eda_Mean"]
eda_features["EDA_Tonic_SD"] = data[["eda_bin1", "eda_bin2", "eda_bin3", "eda_bin4", "eda_bin5",
                                      "eda_bin6", "eda_bin7", "eda_bin8", "eda_bin9", "eda_bin10",
                                      "eda_bin11", "eda_bin12", "eda_bin13", "eda_bin14", "eda_bin15",
                                      "eda_bin16", "eda_bin17", "eda_bin18"]].std(axis=1)
eda_features["EDA_Phasic_Mean"] = data["eda_av"]
eda_features["EDA_Phasic_SD"] = eda_features["EDA_Tonic_SD"]
eda_features["SCR_Peaks"] = (data["eda_peak"] > 0).sum(axis=0)
eda_features["SCR_Amplitude_Max"] = data["eda_peak"]
eda_features["SCR_Amplitude_Mean"] = eda_features_dict["eda_Mean"]
eda_features["Baseline_Tonic_Mean"] = data["base_trial_eda"]
eda_features["Baseline_Phasic_Mean"] = data["base_trial_eda"]
eda_features["Change_Tonic_Mean"] = calculate_change_to_baseline(data, "eda_peak", "base_trial_eda")
eda_features["Change_Phasic_Mean"] = calculate_change_to_baseline(data, "eda_peak", "base_trial_eda")
eda_features["Relative_Change_Tonic"] = calculate_relative_change(data, "eda_peak", "base_trial_eda")
eda_features["Relative_Change_Phasic"] = calculate_relative_change(data, "eda_peak", "base_trial_eda")

# Reorder Columns
eda_features = eda_features[
    ["Participant", "Sound_ID"] + [col for col in eda_features.columns if col not in ["Participant", "Sound_ID"]]
]

# EMG Features
emg_features = pd.DataFrame({
    "Participant": data["Participant"],
    "Sound_ID": data["Sound_ID"]
})
for muscle in ["emg_orb", "emg_cor", "emg_zyg"]:
    muscle_features = calculate_features(data, prefix=muscle)
    for feature_name, values in muscle_features.items():
        feature_type = feature_name.split("_")[1]
        emg_features[f"{muscle.upper()}_{feature_type}"] = values
    emg_features[f"{muscle.upper()}_Baseline_Mean"] = data[f"base_trial_{muscle.split('_')[1]}"]

# Reorder Columns
emg_features = emg_features[
    ["Participant", "Sound_ID"] + [col for col in emg_features.columns if col not in ["Participant", "Sound_ID"]]
]

# Save Feature Results
eda_output_path = os.path.join(output_dir, "eda_features.csv")
emg_output_path = os.path.join(output_dir, "emg_features.csv")

eda_features.to_csv(eda_output_path, index=False)
emg_features.to_csv(emg_output_path, index=False)

print(f"EDA features saved to {eda_output_path}.")
print(f"EMG features saved to {emg_output_path}.")
