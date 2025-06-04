import pandas as pd
import neurokit2 as nk
import os
import warnings
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants as const
from utils import extract_scalar_features

warnings.filterwarnings("ignore")

base_dir = os.path.join(const.BASE_DIR, "dataset2_RobotBehaviour", "Measurements_fixed")
output_folder = os.path.join(const.OUTPUT_DIR, "dataset2")
os.makedirs(output_folder, exist_ok=True)

SIGNALS = {
    "EDA": {"sampling_rate": 4, "ext": "EDA", "method": "eda"},
    "ECG": {"sampling_rate": 130, "ext": "ECG", "method": "ecg"},
    "BVP": {"sampling_rate": 64, "ext": "BVP", "method": "ppg"},
}

participants = range(1, 26)
speeds = [1, 2]
robots = [1, 2, 3]

def process_signal(signal_name, config):
    results = []
    sampling_rate = config["sampling_rate"]
    method = config["method"]
    ext = config["ext"]
    
    window_length = int(const.INTERVAL * sampling_rate)
    step = int(const.STEP * sampling_rate)

    for participant in participants:
        for speed in speeds:
            for robot in robots:
                file_path = os.path.join(base_dir, f"p_{participant}", f"{speed}_{robot}", f"{ext}.csv")
                if not os.path.exists(file_path):
                    print(f"Missing file: {file_path}")
                    continue

                try:
                    df = pd.read_csv(file_path, sep=";")
                    if ext not in df.columns:
                        for col in df.columns:
                            if ext.lower() in col.lower():
                                df = df.rename(columns={col: ext})

                    if ext not in df.columns:
                        print(f"Missing column '{ext}' in: {file_path}")
                        continue

                    df[ext] = df[ext].apply(lambda x: float(str(x).replace(",", ".")) if pd.notnull(x) else x)
                    df = df.dropna(subset=[ext])

                    signal = df[ext].values
                    total_samples = len(signal)
                    start_idx = 0
                    slice_count = 1

                    while start_idx + window_length <= total_samples:
                        window = signal[start_idx:start_idx + window_length]
                        if len(window) < 10:
                            start_idx += step
                            continue

                        process_fn = getattr(nk, f"{method}_process")
                        analyze_fn = getattr(nk, f"{method}_analyze")

                        processed, _ = process_fn(window, sampling_rate=sampling_rate)
                        analyzed = analyze_fn(processed, sampling_rate=sampling_rate, method="interval-related")
                        features = extract_scalar_features(analyzed)

                        row = {
                            "Participant": participant,
                            "Task": (speed - 1) * 3 + robot,
                            "Slice": slice_count,
                            "Speed": speed,
                            "Robots": robot
                        }
                        row.update(features)
                        results.append(row)

                        slice_count += 1
                        start_idx += step

                except Exception as e:
                    print(f"Error in file {file_path}: {e}")
    return pd.DataFrame(results)

for signal_name, config in SIGNALS.items():
    print()
    df = process_signal(signal_name, config)

    if not df.empty:
        output_name = "ppg_features.csv" if signal_name == "BVP" else f"{signal_name.lower()}_features.csv"
        output_path = os.path.join(output_folder, output_name)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    else:
        print(f"No valid data for {signal_name}")
